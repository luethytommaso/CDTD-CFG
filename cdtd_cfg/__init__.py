import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_ema import ExponentialMovingAverage
from tqdm.notebook import tqdm

from .layers import MLP, CatEmbedding, Timewarp_Logistic, WeightNetwork
from .utils import (
    FastTensorDataLoader,
    LinearScheduler,
    cycle,
    low_discrepancy_sampler,
    set_seeds,
)


class MixedTypeDiffusion(nn.Module):
    """
    MixedTypeDiffusion is the core CDTD-CFG module, defining the entiry diffusion pipeline:
    It embeds categorical columns, maps uniform times u to per-feature noise σ based on the adaptive noise schedule (Timewarp_Logistic), 
    adds noise to both types of features, and uses the score network to predict the clean targets (class logits for categoricals, denoised values for continuous).
    
    During training, it builds a calibrated loss, uses a small WeightNetwork to rebalance losses across noise levels, 
    fits the adaptive noise schedule so the model spends effort training where improved denoising is most useful. 
    During sampling, sampler() runs the reverse process from random noise to generate synthetic rows.

    CDTD-CFG extension: Across all steps of the diffusion pipeline the conditioining labels are passed in order to learn conditional score fucntions.
    These conditional score functions combined during sampling to perform classifier free guidance sampling. 
    The exact additions are highlighted in each method.
    """
    def __init__(
        self,
        model, 
        dim, 
        categories,
        proportions,
        num_features,
        sigma_data_cat,
        sigma_data_cont,
        sigma_min_cat,
        sigma_max_cat,
        sigma_min_cont,
        sigma_max_cont,
        cat_emb_init_sigma,
        timewarp_type="bytype", 
        timewarp_weight_low_noise=1.0,
    ):
        """
        Initializes the MixedTypeDiffusion: 
        Record the table shape (how many categorical/continuous features), and builds a per-feature categorical embedding layer (with optional bias). 
        We also create the WeightNetwork (to reweight loss across noise levels) and set the noise bounds (sigma_min/max) for each feature. 
        Lastly, we instantiate the adaptive noise scedule (Timewarp_Logistic) with the chosen hyperparamams (single/bytype/all and `weight_low_noise`).
        """
        super(MixedTypeDiffusion, self).__init__()

        self.dim = dim
        self.num_features = num_features
        self.num_cat_features = len(categories)
        self.num_cont_features = num_features - self.num_cat_features
        self.num_unique_cats = sum(categories)
        self.categories = categories
        self.model = model

        self.cat_emb = CatEmbedding(dim, categories, cat_emb_init_sigma, bias=True) 
        
        self.register_buffer("sigma_data_cat", torch.tensor(sigma_data_cat)) 
        self.register_buffer("sigma_data_cont", torch.tensor(sigma_data_cont))

        entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
        self.register_buffer(
            "normal_const",
            torch.cat((entropy, torch.ones((self.num_cont_features,)))), 
        )

        self.weight_network = WeightNetwork(1024) 
 
        self.timewarp_type = timewarp_type
        self.sigma_min_cat = torch.tensor(sigma_min_cat) 
        self.sigma_max_cat = torch.tensor(sigma_max_cat)
        self.sigma_min_cont = torch.tensor(sigma_min_cont)
        self.sigma_max_cont = torch.tensor(sigma_max_cont)

        sigma_min = torch.cat(
            (
                torch.tensor(sigma_min_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_min_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        sigma_max = torch.cat(
            (
                torch.tensor(sigma_max_cat).repeat(self.num_cat_features),
                torch.tensor(sigma_max_cont).repeat(self.num_cont_features),
            ),
            dim=0,
        )
        self.register_buffer("sigma_max", sigma_max) 
        self.register_buffer("sigma_min", sigma_min)

        self.timewarp_cdf = Timewarp_Logistic(
            self.timewarp_type,
            self.num_cat_features,
            self.num_cont_features,
            sigma_min,
            sigma_max,
            weight_low_noise=timewarp_weight_low_noise,
            decay=0.0,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device 

    def diffusion_loss(self, x_cat_0, x_cont_0, cat_logits, cont_preds):
        assert len(cat_logits) == self.num_cat_features 
        assert cont_preds.shape == x_cont_0.shape

        ce_losses = torch.stack(
            [
                F.cross_entropy(cat_logits[i], x_cat_0[:, i], reduction="none")
                for i in range(self.num_cat_features)
            ],
            dim=1,
        )
        mse_losses = (cont_preds - x_cont_0) ** 2

        return ce_losses, mse_losses

    def add_noise(self, x_cat_emb_0, x_cont_0, sigma):
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]

        x_cat_emb_t = x_cat_emb_0 + torch.randn_like(x_cat_emb_0) * sigma_cat.unsqueeze(
            2
        ) 
        x_cont_t = x_cont_0 + torch.randn_like(x_cont_0) * sigma_cont

        return x_cat_emb_t, x_cont_t

    def loss_fn(self, x_cat, x_cont, u=None, cfg = False, y_condition_1=None, y_condition_2=None, dropout=1.0):
        """
        This method is exclusively called during training. computes the training losses for one batch. 
        It samples (or takes given) times u, maps them to per-feature noise σ via the learned timewarp, 
        and adds Gaussian noise to categorical embeddings and continuous values for noise level σ.
        
        It then runs the MLP with EDM preconditioning to predict per-feature targets
        (class logits for categorical features, denoised values for continuous) and computes the CE/MSE losses.
        
        Next, it calibrates losses (entropy for categorical features) and applies the EDM weight to the continuous MSE.
        
        It feeds log(u) into the WeightNetwork to produce a positive weight that matches the observed average loss. 
        This weight reweights the main loss, and the network’s own regression error is added.
        
        Finally, it adds a timewarp fitting loss that regresses the CDF to the empirical loss profile. 
        The overall objective is the mean reweighted loss + timewarp loss + weight-net loss. Returns the loss dictionary and the per-sample σ.

        These losses are used in multiple ways to guide the diffusion process. Through backpropagation each step we train the score network and, 
        via the auxiliary terms, update the timewarp and weight network. In this manner the model learns to denoise while the noise schedule adapts to the data.

        CDTD-CFG Extensions:
        If cfg=True, we pass the conditioning labels but then randomly mask them using a dropout mechanism. For each sample in the batch, 
        we draw a uniform random value and, based on the dropout ratio, decide whether to mask both labels (unconditional), only label 1, or only label 2.
        Masked labels are set to -1 before calling the precondition method, ensuring the model learns unconditional and single-label score functions.        
        """
        batch = x_cat.shape[0] if x_cat is not None else x_cont.shape[0]

        x_cat_emb_0 = self.cat_emb(x_cat) 
        x_cont_0 = x_cont
        x_cat_0 = x_cat

        with torch.no_grad():
            if u is None:
                u = low_discrepancy_sampler(batch, device=self.device)  
            sigma = self.timewarp_cdf(u, invert=True).detach().to(torch.float32) 
            u = u.to(torch.float32)
            assert sigma.shape == (batch, self.num_features) 

        x_cat_emb_t, x_cont_t = self.add_noise(x_cat_emb_0, x_cont_0, sigma)

        if cfg:
            random_vals = torch.rand(batch, device=x_cat_emb_t.device)
            unconditional_mask = random_vals < dropout

            label_1_mask = (random_vals >= dropout) & (random_vals < dropout + 0.5 * (1 - dropout))
            label_2_mask = (random_vals >= dropout + 0.5 * (1 - dropout))

            y_condition_1[unconditional_mask] = -1
            y_condition_2[unconditional_mask] = -1

            y_condition_2[label_1_mask] = -1
            y_condition_1[label_2_mask] = -1
      
        cat_logits, cont_preds = self.precondition(x_cat_emb_t, x_cont_t, u, sigma, 
                                                   cfg, y_condition_1, y_condition_2,dropout_ratio=dropout) 
        ce_losses, mse_losses = self.diffusion_loss( 
            x_cat_0, x_cont_0, cat_logits, cont_preds 
        ) 

        sigma_cont = sigma[:, self.num_cat_features :] 
        cont_weight = (sigma_cont**2 + self.sigma_data_cont**2) / (
            (sigma_cont * self.sigma_data_cont) ** 2 + 1e-7
        )

        losses = {}
        losses["unweighted"] = torch.cat((ce_losses, mse_losses), dim=1)
        losses["unweighted_calibrated"] = losses["unweighted"] / self.normal_const
        weighted_calibrated = (     
            torch.cat((ce_losses, cont_weight * mse_losses), dim=1) / self.normal_const
        )
        
        c_noise = torch.log(u.to(torch.float32) + 1e-8) * 0.25 
        time_reweight = self.weight_network(c_noise).unsqueeze(1) 
        
        losses["timewarping"] = self.timewarp_cdf.loss_fn(
            sigma.detach(), losses["unweighted_calibrated"].detach()
        )

        weightnet_loss = (
            time_reweight.exp() - weighted_calibrated.detach().mean(1)
        ) ** 2
        
        losses["weighted_calibrated"] = (
            weighted_calibrated / time_reweight.exp().detach()
        )

        train_loss = (
            losses["weighted_calibrated"].mean() 
            + losses["timewarping"].mean()
            + weightnet_loss.mean()
        )

        losses["train_loss"] = train_loss

        return losses, sigma

    def precondition(self, x_cat_emb_t, x_cont_t, u, sigma, 
                     cfg = False, y_condition_1=None, y_condition_2=None, dropout_ratio=1.0, sample=False):
        """
        Improved preconditioning proposed in the paper "Elucidating the Design
        Space of Diffusion-Based Generative Models" (EDM) adjusted for categorical data

        The model applies EDM-style preconditioning so the network sees inputs on a comparable scale at any noise level, 
        and produces categorical logits and a denoised continuous estimate.

        The scaled categorical embeddings and continuous values, together with `c_noise`, are inputted to the MLP. 
        For the continuous outputs we don’t use the raw prediction directly, we blend it with the current noisy input via
        D_x = c_skip * x_t + c_out * pred, where c_skip/c_ou` depend on the signal-to-noise ratio.
        The intuition is that at a low noise our prediction depends more on the input, at high noise more on the network’s output. 

        CDTD-CFG changes:
        This method now accepts cfg, y_condition_1, y_condition_2, dropout_ratio, and sample.
        When CFG is True, loss_fn masks labels by setting them to −1 during training, and the MLP treats −1 as no conditioning. 
        During sampling we call precondition three times externally for different combinations of y_condition_1 and y_condition_2.
        """                         
        sigma_cat = sigma[:, : self.num_cat_features]
        sigma_cont = sigma[:, self.num_cat_features :]
        
        c_in_cat = (
            1 / (self.sigma_data_cat**2 + sigma_cat.unsqueeze(2) ** 2).sqrt()
        ) 
        c_in_cont = 1 / (self.sigma_data_cont**2 + sigma_cont**2).sqrt()
        c_noise = torch.log(u + 1e-8) * 0.25 * 1000  

        cat_logits, cont_preds = self.model(
            c_in_cat * x_cat_emb_t,
            c_in_cont * x_cont_t,
            c_noise,
            cfg,
            y_condition_1,
            y_condition_2,
            dropout_ratio=dropout_ratio,
            sample=sample
        )
        
        assert len(cat_logits) == self.num_cat_features
        assert cont_preds.shape == x_cont_t.shape

        c_skip = self.sigma_data_cont**2 / (sigma_cont**2 + self.sigma_data_cont**2)
        c_out = (
            sigma_cont
            * self.sigma_data_cont
            / (sigma_cont**2 + self.sigma_data_cont**2).sqrt()
        )
        D_x = c_skip * x_cont_t + c_out * cont_preds
        
        return cat_logits, D_x

    def score_interpolation(self, x_cat_emb_t, cat_logits, sigma, return_probs=False):
        """
        Estimates the categorical score (to denoise) or returns class probabilities, depending on return_probs.
        The categorical score is given by (x_cat_emb_t - x̂_0) / σ_cat.
        """
        if return_probs:
            probs = []
            for logits in cat_logits:
                probs.append(F.softmax(logits.to(torch.float64), dim=1))
            return probs 

        def interpolate_emb(i):
            p = F.softmax(cat_logits[i].to(torch.float64), dim=1)
            true_emb = self.cat_emb.get_all_feat_emb(i).to(torch.float64) 
            return torch.matmul(p, true_emb)

        x_cat_emb_0_hat = torch.zeros_like( 
            x_cat_emb_t, device=self.device, dtype=torch.float64
        )
        for i in range(self.num_cat_features):
            x_cat_emb_0_hat[:, i, :] = interpolate_emb(i) 

        sigma_cat = sigma[:, : self.num_cat_features] 
        interpolated_score = (x_cat_emb_t - x_cat_emb_0_hat) / sigma_cat.unsqueeze(2) 
        
        return interpolated_score, x_cat_emb_0_hat 

    @torch.inference_mode()
    def sampler(self, cat_latents, cont_latents, num_steps=200, cfg = False, y_condition_1=None, y_condition_2=None, cfg_scale_1 = 0.0, cfg_scale_2 = 0.0):
        """
        Generates synthetic samples via reverse diffusion. It defines a sequence of times u going from 1 to 0 over num_steps.
        These are mapped them to noise levels σ with the learned timewarp (inverse CDF), and random latents are initialized at the maximum noise.
        
        At each step we run `precondition` to get categorical logits and a denoised continuous estimate. 
        We compute scores: for categoricals via score_interpolation, for continuous as (x_t - x_denoised)/σ_t. 
        An Euler update is performed to generate partially denoised x for the next timestep x_{t+1} = x_t + h * score, where h = σ_next - σ_cur.
        
        After the loop, we make one final forward pass to get categorical probabilities and pick the most likely class per feature (argmax). 
        
        CDTD-CFG extension: If `cfg=True`, each denoising step runs three passes of `precondition`:
        an unconditional one (all labels set to None), label-1 only, label-2 only.
        We turn each pass into scores and blend them via classifier-free guidance score function: d = d_unc + cfg_scale_1*(d_1 - d_unc) + cfg_scale_2*(d_2 - d_unc).
        After the last loop, we combine the three categorical probabilities before taking argmax per feature.
        """
        B = (
            cont_latents.shape[0]
            if self.num_cont_features > 0
            else cat_latents.shape[0]
        )
        
        u_steps = torch.linspace(
            1, 0, num_steps + 1, device=self.device, dtype=torch.float64
        )
        t_steps = self.timewarp_cdf(u_steps, invert=True) 
        
        assert torch.allclose(t_steps[0].to(torch.float32), self.sigma_max.float())
        assert torch.allclose(t_steps[-1].to(torch.float32), self.sigma_min.float())

        t_cat_next = t_steps[0, : self.num_cat_features] 
        t_cont_next = t_steps[0, self.num_cat_features :]

        x_cat_next = cat_latents.to(torch.float64) * t_cat_next.unsqueeze(1) 
        x_cont_next = cont_latents.to(torch.float64) * t_cont_next

        for i, (t_cur, t_next, u_cur) in enumerate( 
            zip(t_steps[:-1], t_steps[1:], u_steps[:-1])
        ): 
            
            t_cur = t_cur.repeat((B, 1)) 
            t_next = t_next.repeat((B, 1)) 
            t_cont_cur = t_cur[:, self.num_cat_features :] 
            
            if cfg: 
                cat_logits, x_cont_denoised = self.precondition( # unconditional setting both labels to 0
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=None,
                    y_condition_2=None,
                    sample=True
                )
                d_cat_unc, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur)
                d_cont_unc = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur 

                cat_logits_1, x_cont_denoised_1 = self.precondition( # Conditoinal on only label 1
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=y_condition_1,
                    y_condition_2=None,
                    sample=True
                )
                d_cat_con_1, _ = self.score_interpolation(x_cat_next, cat_logits_1, t_cur) 
                d_cont_con_1 = (x_cont_next - x_cont_denoised_1.to(torch.float64)) / t_cont_cur
                
                cat_logits_2, x_cont_denoised_2 = self.precondition( # Conditional on only label 2
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=None,
                    y_condition_2=y_condition_2,
                    sample=True
                )
                d_cat_con_2, _ = self.score_interpolation(x_cat_next, cat_logits_2, t_cur) 
                d_cont_con_2 = (x_cont_next - x_cont_denoised_2.to(torch.float64)) / t_cont_cur 
                                
                d_cat_cur = d_cat_unc + cfg_scale_1 * (d_cat_con_1 - d_cat_unc) + cfg_scale_2 * (d_cat_con_2 - d_cat_unc) 
                d_cont_cur = d_cont_unc + cfg_scale_1 * (d_cont_con_1 - d_cont_unc) + cfg_scale_2 * (d_cont_con_2 - d_cont_unc) 
                
            else:
                cat_logits, x_cont_denoised = self.precondition(
                    x_cat_emb_t=x_cat_next.to(torch.float32),
                    x_cont_t=x_cont_next.to(torch.float32),
                    u=u_cur.to(torch.float32).repeat((B,)),
                    sigma=t_cur.to(torch.float32),
                    cfg = cfg,
                    y_condition_1=None,
                    y_condition_2=None,
                    sample=True
                )

                d_cat_cur, _ = self.score_interpolation(x_cat_next, cat_logits, t_cur) 
                d_cont_cur = (x_cont_next - x_cont_denoised.to(torch.float64)) / t_cont_cur 
                
            h = t_next - t_cur 
            x_cat_next = (
                x_cat_next + h[:, : self.num_cat_features].unsqueeze(2) * d_cat_cur
            )
            x_cont_next = x_cont_next + h[:, self.num_cat_features :] * d_cont_cur

        u_final = u_steps[:-1][-1]
        t_final = t_steps[:-1][-1].repeat(B, 1)

        if cfg: 
            cat_logits_unc, _ = self.precondition(
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
                cfg = cfg,
                y_condition_1=None,
                y_condition_2=None,
                sample=True
            )
            probs_unc = self.score_interpolation(x_cat_next, cat_logits_unc, t_final, return_probs=True)
                    
            cat_logits_1, _ = self.precondition( 
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
                cfg = cfg,
                y_condition_1=y_condition_1,
                y_condition_2=None,
                sample=True                
            )
            probs_1 = self.score_interpolation(x_cat_next, cat_logits_1, t_final, return_probs=True)

            cat_logits_2, _ = self.precondition(
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
                cfg = cfg,
                y_condition_1=None,
                y_condition_2=y_condition_2,
                sample=True                
            )
            probs_2 = self.score_interpolation(x_cat_next, cat_logits_2, t_final, return_probs=True)

            probs = []
            for p_unc, p_1, p_2 in zip(probs_unc, probs_1, probs_2):
                combined = p_unc + cfg_scale_1 * (p_1 - p_unc) + cfg_scale_2 * (p_2 - p_unc)
                probs.append(combined)
            
        else: 
            cat_logits, _ = self.precondition(
                x_cat_emb_t=x_cat_next.to(torch.float32),
                x_cont_t=x_cont_next.to(torch.float32),
                u=u_cur.to(torch.float32).repeat((B,)),
                sigma=t_cur.to(torch.float32),
                cfg = cfg,
                y_condition_1=None,
                y_condition_2=None,
                sample=True
            )

            probs = self.score_interpolation(
                x_cat_next, cat_logits, t_final, return_probs=True
            )
            
        x_cat_gen = torch.empty(B, self.num_cat_features, device=self.device)
        for i in range(self.num_cat_features):
            x_cat_gen[:, i] = probs[i].argmax(1)

        return x_cat_gen.cpu(), x_cont_next.cpu()


class CDTD:
    """ 
    CDTD is the overall class used for both CDTD-CFG and CDTD. It builds the score network (MLP) and the
    MixedTypeDiffusion module with the right feature information (categorical cardinalities, proportions, embedding sizes).
    This class is called from the main model, and is tasked with both training and sampling in their entirety.
    """
    def __init__(
        self,
        X_cat_train,
        X_cont_train,
        cat_emb_dim=16, 
        mlp_emb_dim=256, 
        mlp_n_layers=5,
        mlp_n_units=1024,
        sigma_data_cat=1.0, 
        sigma_data_cont=1.0,
        sigma_min_cat=0.0, 
        sigma_min_cont=0.0,
        sigma_max_cat=100.0,
        sigma_max_cont=80.0,
        cat_emb_init_sigma=0.001, 
        timewarp_type="bytype",  
        timewarp_weight_low_noise=1.0,
        cfg = False,
        y_condition_1=None, 
        y_condition_2=None
    ):
        """
        From the training tensors it computes how many categorical and continuous features there are, the category cardinalities, 
        and per-class proportions used to calibrate the outputs.  It subsequently constructs the score network (MLP)
        with the chosen hyperparams and the categorical embedding size, and inputs this into MixedTypeDiffusion.
        
        CDTD-CFG extension: If cfg=True, we also compute the number of classes for each conditioning label (num_classes_1/2) 
        and pass these into the MLP so it allocates label-embedding layers.
        """
        super().__init__()

        self.num_cat_features = X_cat_train.shape[1]
        self.num_cont_features = X_cont_train.shape[1]
        self.num_features = self.num_cat_features + self.num_cont_features
        self.cat_emb_dim = cat_emb_dim

        self.categories = []
        for i in range(self.num_cat_features): 
            uniq_vals = np.unique(X_cat_train[:, i])
            self.categories.append(len(uniq_vals))

        self.proportions = []
        n_sample = X_cat_train.shape[0]
        for i in range(len(self.categories)): 
            _, counts = X_cat_train[:, i].unique(return_counts=True)
            self.proportions.append(counts / n_sample)

        if cfg:     
            num_classes_1 = len(np.unique(y_condition_1.numpy()))
            num_classes_2 = len(np.unique(y_condition_2.numpy()))
        else:
            num_classes_1 = None
            num_classes_2 = None
            
        score_model = MLP(
            self.num_cont_features,
            self.cat_emb_dim,
            self.categories,
            self.proportions, 
            mlp_emb_dim,
            mlp_n_layers,
            mlp_n_units,
            num_classes_1 = num_classes_1, 
            num_classes_2 = num_classes_2
        )

        self.diff_model = MixedTypeDiffusion(
            model=score_model,
            dim=self.cat_emb_dim,
            categories=self.categories,
            num_features=self.num_features,
            sigma_data_cat=sigma_data_cat,
            sigma_data_cont=sigma_data_cont,
            sigma_min_cat=sigma_min_cat,
            sigma_max_cat=sigma_max_cat,
            sigma_min_cont=sigma_min_cont,
            sigma_max_cont=sigma_max_cont,
            proportions=self.proportions, 
            cat_emb_init_sigma=cat_emb_init_sigma,
            timewarp_type=timewarp_type, 
            timewarp_weight_low_noise=timewarp_weight_low_noise,
        )

    def fit(  
        self,
        X_cat_train,
        X_cont_train,
        num_steps_train=30_000, 
        num_steps_warmup=1000, 
        batch_size=4096, 
        lr=1e-3, 
        seed=42,
        ema_decay=0.999,
        log_steps=100,
        cfg = False,  
        y_condition_1=None, 
        y_condition_2=None,
        dropout_ratio = 1.0
    ):
        """
        Trains the CDTD(-CFG). It starts by creating the fast data loader, and setting all seeds for reproducability.
        It sets up additional training utilities: an Exponential Moving Average (EMA) over the model’s parameters (for a smoother “target” model at the end), 
        the AdamW optimizer, and a linear learning-rate schedule with warmup and optional annealing. It also starts a progress bar and an average of the loss.

        Each training step looks as follows: the gradients of the previous setp are set to zero. 
        We select a batch, and we call diff_model.loss_fn to compute the batch loss. The loss is backpropagated to update the score function, 
        and call an optimizer to update the parameters. Finally we update the adaptive timewarp parameters (timewarp_cdf.update_ema()) and the model weights. 

        CDTD-CFG extensions: If cfg=True, the dataloader also returns y_condition_1/2, and loss_fn is called with those labels and dropout_ratio. 
        Inside loss_fn, labels are randomly masked per sample to mix unconditional and single-label conditions.    
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = FastTensorDataLoader(
            X_cat_train,
            X_cont_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            y_condition_1=y_condition_1, 
            y_condition_2=y_condition_2
        ) 
        train_iter = cycle(train_loader)

        set_seeds(seed, cuda_deterministic=True) 
        self.diff_model = self.diff_model.to(self.device) 
        self.diff_model.train() 

        self.ema_diff_model = ExponentialMovingAverage(
            self.diff_model.parameters(), decay=ema_decay 
        ) 

        self.optimizer = torch.optim.AdamW(
            self.diff_model.parameters(), lr=lr, weight_decay=0 
        )
        self.scheduler = LinearScheduler(
            num_steps_train,
            base_lr=lr,
            final_lr=1e-6,
            warmup_steps=num_steps_warmup,
            warmup_begin_lr=1e-6,
            anneal_lr=True,
        ) 

        self.current_step = 0
        n_obs = sum_loss = 0
        train_start = time.time()

        with tqdm(
            initial=self.current_step,
            total=num_steps_train,
        ) as pbar: 
            while self.current_step < num_steps_train:
                self.optimizer.zero_grad() 
                
                inputs = next(train_iter)
                x_cat, x_cont, y_cond_1_batch, y_cond_2_batch = (
                    input.to(self.device) if input is not None else None
                    for input in inputs
                )  
                
                losses, _ = self.diff_model.loss_fn(x_cat, x_cont, None,
                                                    cfg, y_condition_1=y_cond_1_batch, y_condition_2=y_cond_2_batch,dropout=dropout_ratio) 
                losses["train_loss"].backward() 

                self.optimizer.step()
                self.diff_model.timewarp_cdf.update_ema()
                self.ema_diff_model.update()

                sum_loss += losses["train_loss"].detach().mean().item() * x_cat.shape[0]
                n_obs += x_cat.shape[0]
                self.current_step += 1
                pbar.update(1)

                if self.current_step % log_steps == 0: 
                    pbar.set_description(
                        f"Loss (last {log_steps} steps): {(sum_loss / n_obs):.3f}"
                    )
                    n_obs = sum_loss = 0

                if self.scheduler:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.scheduler(self.current_step)

        train_duration = time.time() - train_start
        print(f"Training took {(train_duration / 60):.2f} min.")

        self.ema_diff_model.copy_to()
        self.diff_model.eval()

        return self.diff_model 

    def sample(self, num_samples, num_steps=200, batch_size=4096, seed=42, 
               cfg=False, probs_matrix=None, cfg_scale_1 =0.0, cfg_scale_2 =0.0, return_latents=False):
        """
        Generates num_samples synthetic rows by running the learned reverse diffusion process.
        We split the process into batches and sample Gaussian latents for categorical and continuous features, 
        and denoise them with diff_model.sampler` over num_steps. The cleaned outputs are returned as numpy arrays.

        CDTD-CFG extensions: If cfg=True, we first draw a conditioning pair (y_condition_1, y_condition_2) for every sample using the probs_matrix.
        probs_matrix is transformed into a matrix containing the joint CDF distribution over the two labels: we subsequently flatten the CDF, 
        draw a uniform random number, and pick the pair whose CDF interval contains it. This lets the model control how often each label-pair appears.        

        To illustarte if we have two binary features, where are equally interested in every combination, this would look like:
        [0.25, 0.5]
        [0.75, 1.0]
        This allows the CDTD-CFG to be called ID as well, where the matrix simply follows the distributions seen in the training data.
        All usecases tested in this paper apply OOD sampling for a single combination, resulting in a CDF matrix like:
        [0, 1]
        [1, 1]

        These per-sample labels, together with guidance weights cfg_scale_1 and cfg_scale_2, are passed into diff_model.sampler.
        Inside, the sampler computes three score trajectories and blends them.
        Setting both scales to 0 yields unconditional sampling, allowing for simple CDTD sampling.
        
        If return_latents=True, the function also returns a concatenated view of the Gaussian latents used to generate each row.
        While this is currently not used in the papers final diagnostics, there is a clear correlation between latent space position and OOD data quality,
        which could be interesting to further investigate as it might allow for potential model improvements.
        """           
        set_seeds(seed, cuda_deterministic=True)
        n_batches, remainder = divmod(num_samples, batch_size) 
        sample_sizes = (
            n_batches * [batch_size] + [remainder]
            if remainder != 0
            else n_batches * [batch_size]
        ) 

        x_cat_list = []
        x_cont_list = []
        cat_latent_batch_list = []
        cont_latent_batch_list = []

        for num_samples in tqdm(sample_sizes):
            cat_latents = torch.randn(
                (num_samples, self.num_cat_features, self.cat_emb_dim),
                device=self.device,
            ) 
            cont_latents = torch.randn(
                (num_samples, self.num_cont_features), device=self.device
            )

            if cfg:
                flat_probs = probs_matrix.flatten()
                flat_cdf = np.cumsum(flat_probs)
                flat_cdf[-1] = 1.0  
                
                rand_vals = np.random.rand(num_samples)
            
                sampled_indices = np.searchsorted(flat_cdf, rand_vals, side="right")
            
                y_condition_1, y_condition_2 = np.unravel_index(sampled_indices, probs_matrix.shape)
            
                y_condition_1 = torch.tensor(y_condition_1, device=self.device).long()
                y_condition_2 = torch.tensor(y_condition_2, device=self.device).long()
                                
            else:
                y_condition_1 = None
                y_condition_2 = None
                        
            x_cat_gen, x_cont_gen = self.diff_model.sampler(
                cat_latents, cont_latents, num_steps, cfg, y_condition_1=y_condition_1, y_condition_2=y_condition_2, cfg_scale_1=cfg_scale_1, cfg_scale_2=cfg_scale_2
            ) 
            x_cat_list.append(x_cat_gen)
            x_cont_list.append(x_cont_gen)

            if return_latents: 
                cat_latent_batch_list.append(cat_latents.detach().cpu())
                cont_latent_batch_list.append(cont_latents.detach().cpu())

        x_cat = torch.cat(x_cat_list).cpu()
        x_cont = torch.cat(x_cont_list).cpu()
        latents_all = None

        if return_latents:            
            cat_latent_all = torch.cat(cat_latent_batch_list, dim=0)           
            cont_latent_all = torch.cat(cont_latent_batch_list, dim=0)        
            
            cat_latent_all_flat = cat_latent_all.reshape(cat_latent_all.shape[0], -1)  
    
            latents_all = torch.cat([cat_latent_all_flat, cont_latent_all], dim=1).numpy()  

        return x_cat.long().numpy(), x_cont.numpy(), latents_all
