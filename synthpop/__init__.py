from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
import numpy as np
import random

class Synthpop:
    """
    Synthpop is a CART-based tabular synthesizer inspired by the synthpop R package.
    It sequentially generates synhetic features with simple decision trees (CART), conditioning on the fixed inputs and any already generated columns.
    Our version includes both the plain in-distribution mode (cfg=False) and an extension for OOD synthesis to synthesize missing class combinations.
    
    For OOD, two conditioning variables are fixed and the target classes are passed at sampling time.
    We train two trees (each omits one of the two conditions) so each can propose a value even if the pair never appeared in training.
    Their outputs are then blended using simple weighting, which steers results away from impossible combinations.
    """
    def __init__(self, cat_train, cont_train, x_cat_indices, x_cont_indices=None,
                 order=None, cfg=False, max_depth=5):
        """
        The initailization takes in the training arrays (categorical and continuous), and records which columns are fixed beforehand (conditions). 
        Furthermore, a user generated order can be inputted (if no order is given, all categorical variables are constructed first, and continuous ones last).
        
        This method furthermore sets up dictionaries for the main trees if cfg = False (and per-leaf mean/std for continuous targets),
        and the variable omitting trees for OOD if cfg = True (cfg_1_*, cfg_2_*) that omit one conditioning variable each.
        """
        self.cat_train = cat_train.numpy() if hasattr(cat_train, 'numpy') else cat_train
        self.cont_train = cont_train.numpy() if hasattr(cont_train, 'numpy') else cont_train

        self.x_cat_indices = x_cat_indices
        self.x_cont_indices = x_cont_indices if x_cont_indices else []
        self.cfg = cfg
        self.max_depth = max_depth

        self.main_model = {}
        self.main_leaf_info = {}
        self.cfg_1_model = {}
        self.cfg_2_model = {}
        self.cfg_1_leaf_info = {}
        self.cfg_2_leaf_info = {}

        self.context_sizes = {}

        all_cat_indices = list(range(self.cat_train.shape[1]))
        all_cont_indices = list(range(self.cont_train.shape[1])) if self.cont_train is not None else []
        self.y_cat_indices = [i for i in all_cat_indices if i not in self.x_cat_indices]
        self.y_cont_indices = [i for i in all_cont_indices if i not in self.x_cont_indices]

        default_order = [f"v{i}" for i in self.y_cat_indices] + [f"c{i}" for i in self.y_cont_indices]
        self.order = order if order else default_order

    def build_cart(self, var, X, y, model_dict, leaf_info_dict=None):
        """
        Fit a single CART for one target variable. If var is continuous ('c*') it trains a DecisionTreeRegressor,
        and stores the per-leaf mean and std so we can sample Gaussian values at inference. 
        If var is categorical, train a DecisionTreeClassifier whose leaf probabilities will be used to sample categories. 
        """
        if isinstance(var, str) and var.startswith('c'):
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, y)
            model_dict[var] = model

            if leaf_info_dict is not None:
                leaf_ids = model.apply(X)
                leaf_stats = {}
                for leaf in np.unique(leaf_ids):
                    values = y[leaf_ids == leaf]
                    mean = values.mean()
                    std = values.std() if len(values) > 1 else 1e-6
                    leaf_stats[leaf] = (mean, std)
                leaf_info_dict[var] = leaf_stats
        else:
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X, y)
            model_dict[var] = model

    def fit(self):
        """
        Sequentially trains one CART per target variable in an autoregressive manner using a growing "context" matrix. 
        At the start this context exclusively contains the prior conditioning variables, and their values observed in the training data.

        For the first target variable in self.order, we build X from those fixed columns and fit a tree to predict that target from X. 
        Once that tree is trained, we append the target column itself to the context, 
        so the next tree uses both the fixed variables and everything we’ve already modeled. 
        
        When cfg = true, all we do is train two distinct trees, excluding either the first or second column from the context matrix.
        This works since the first two columns of X are the fixed/conditioning vars. 
        This way no single tree depends on both conditions at once, which lets us produce values even for unseen class combinations.        
        """
        self.main_model = {}
        self.main_leaf_info = {}
        self.cfg_1_model = {}
        self.cfg_2_model = {}
        self.cfg_1_leaf_info = {}
        self.cfg_2_leaf_info = {}

        context_cat = list(self.x_cat_indices)
        context_cont = list(self.x_cont_indices)

        for var in self.order:
            if var.startswith("c"):
                idx = int(var[1:])
                y = self.cont_train[:, idx]
            else:
                idx = int(var[1:])
                y = self.cat_train[:, idx]

            X_cat = self.cat_train[:, context_cat] if context_cat else np.empty((self.cat_train.shape[0], 0))
            X_cont = self.cont_train[:, context_cont] if context_cont else np.empty((self.cat_train.shape[0], 0))
            X = np.hstack((X_cat, X_cont)) if X_cat.size and X_cont.size else (X_cat if X_cat.size else X_cont)

            self.build_cart(var, X, y, self.main_model, self.main_leaf_info)
            self.context_sizes[var] = X.shape[1]

            if self.cfg:
                for i in range(2): 
                    if X.shape[1] == 0:
                        raise ValueError("Cannot train CFG model, context has no features.")
                    X_cfg = np.delete(X, i, axis=1)
                    if i == 0:
                        self.build_cart(var, X_cfg, y, self.cfg_1_model, self.cfg_1_leaf_info)
                    else:
                        self.build_cart(var, X_cfg, y, self.cfg_2_model, self.cfg_2_leaf_info)

            if var.startswith("c"):
                context_cont.append(idx)
            else:
                context_cat.append(idx)

        return self.main_model, self.main_leaf_info

    def sample(self, X_fixed_cat, X_fixed_cont=None, targeted_sampling=False):
        """
        Generates new rows given fixed conditioning values. The first variable is generated exclusively using 
        the prior fixed variables, after which the generated values are added to the context and the process is repeated.
        
        For both ID and OOD modes, X_fixed_cat and X_fixed_cont provide the fixed context per row,
        these values are set before starting generation and the trees condition on them at every step.
        ID sampling: we fill X_fixed_ by copying the training distribution, so the ID samples match train distributions.
        OOD sampling: we set the two conditioning variables in X_fixed_cat to the target class pair for every row,
        ensuring every generated sample belongs to that specific missing class-combination.
        
        If cfg and targeted_sampling are both True, we use a dual-tree strategy: two trees are trained, each omitting one conditioning variable. 
        At sampling time, both trees propose a value for the current feature. We then give a probability to each proposal under the other trees leaf.
        For categoricals targets we look at the other tree’s probability of that class, for continuous variables the other leaf’s Gaussian pdf using its mean/std.
        We pick stochastically, proportional to these probabilities, so the option that’s more likely under both conditions is preferred.
        If both proposals are implausible (zero/near-zero probability or >4σ away), we retry up to 5 times. 
        If it still fails, the row is marked invalid. This keeps OOD samples feasible while allowing unseen class combinations.
        
        If sampling ID, we use the main tree only: sampling from the leaf Gaussian distribution for continuous variables 
        or from the predicted class probabilities for categorical variables.
        
        Finally, we stack the generated columns and drop any invalid rows (removing the same rows from the fixed inputs to keep rows aligned)
        then return the generated categorical/continuous arrays along with the fixed inputs.
        """
        n_samples = X_fixed_cat.shape[0]
        context_cat = X_fixed_cat.tolist()
        context_cont = X_fixed_cont.tolist() if X_fixed_cont is not None else [[] for _ in range(n_samples)]

        generated_cat = {i: [] for i in self.y_cat_indices}
        generated_cont = {i: [] for i in self.y_cont_indices}

        invalid_observations = []

        for var in self.order:
            if var.startswith("v") and int(var[1:]) in self.x_cat_indices:
                continue
            if var.startswith("c") and int(var[1:]) in self.x_cont_indices:
                continue

            context_size = self.context_sizes[var]
            X_context = np.array([cc + ct for cc, ct in zip(context_cat, context_cont)])
            X_context = X_context[:, :context_size]

            idx = int(var[1:])
            is_cont = var.startswith("c")

            if self.cfg and targeted_sampling:
                income_pos = 0
                gender_pos = 1

                X_context_cfg1 = np.delete(X_context, income_pos, axis=1)
                X_context_cfg2 = np.delete(X_context, gender_pos, axis=1)

                if is_cont:
                    model_1 = self.cfg_1_model[var]
                    model_2 = self.cfg_2_model[var]

                    leaf_ids_1 = model_1.apply(X_context_cfg1)
                    leaf_ids_2 = model_2.apply(X_context_cfg2)

                    for i in range(n_samples):
                        mean_1, std_1 = self.cfg_1_leaf_info[var][leaf_ids_1[i]]
                        mean_2, std_2 = self.cfg_2_leaf_info[var][leaf_ids_2[i]]

                        draw = 0

                        while draw < 5:
                            pred_1 = np.random.normal(mean_1, std_1)
                            pred_2 = np.random.normal(mean_2, std_2)

                            z_1_in_2 = abs(pred_1 - mean_2) / std_2
                            z_2_in_1 = abs(pred_2 - mean_1) / std_1

                            p_1 = np.exp(-0.5 * z_1_in_2**2)
                            p_2 = np.exp(-0.5 * z_2_in_1**2)

                            if z_1_in_2 > 4 and z_2_in_1 > 4:
                                draw += 1
                                if draw == 5:
                                    p_1 = p_2 = 0.5
                                    invalid_observations.append(i)

                            else:
                                break

                        prob_pick_1 = p_1 / (p_1 + p_2)
                        chosen = pred_1 if random.random() < prob_pick_1 else pred_2
                        context_cont[i].append(chosen)
                        generated_cont[idx].append(chosen)

                else:
                    model_1 = self.cfg_1_model[var]
                    model_2 = self.cfg_2_model[var]

                    X_cfg1 = X_context[:, 1:]
                    X_cfg2 = X_context[:, [0] + list(range(2, X_context.shape[1]))]

                    probs_1 = model_1.predict_proba(X_cfg1)
                    probs_2 = model_2.predict_proba(X_cfg2)

                    for i in range(n_samples):
                        draw = 0

                        while draw < 5:
                            pred_1 = np.random.choice(model_1.classes_, p=probs_1[i])
                            pred_2 = np.random.choice(model_2.classes_, p=probs_2[i])

                            idx_1_in_2 = list(model_2.classes_).index(pred_1)
                            idx_2_in_1 = list(model_1.classes_).index(pred_2)

                            p_1 = probs_2[i][idx_1_in_2]
                            p_2 = probs_1[i][idx_2_in_1]

                            if p_1 + p_2 == 0:
                                draw += 1
                                if draw == 5:
                                    p_1 = p_2 = 0.5
                                    invalid_observations.append(i)

                            else:
                                break

                        prob_pick_1 = p_1 / (p_1 + p_2)
                        chosen = pred_1 if random.random() < prob_pick_1 else pred_2

                        context_cat[i].append(chosen)
                        generated_cat[idx].append(chosen)

            else:
                if is_cont:
                    model = self.main_model[var]
                    leaf_ids = model.apply(X_context)
                    preds = [np.random.normal(*self.main_leaf_info[var][lid]) for lid in leaf_ids]
                    for i in range(n_samples):
                        context_cont[i].append(preds[i])
                        generated_cont[idx].append(preds[i])
                else:
                    model = self.main_model[var]
                    probs = model.predict_proba(X_context)
                    preds = [np.random.choice(model.classes_, p=prob) for prob in probs]
                    for i in range(n_samples):
                        context_cat[i].append(preds[i])
                        generated_cat[idx].append(preds[i])

        x_cat_gen = (
            np.column_stack([generated_cat[i] for i in self.y_cat_indices])
            if self.y_cat_indices else np.empty((n_samples, 0))
        )
        x_cont_gen = (
            np.column_stack([generated_cont[i] for i in self.y_cont_indices])
            if self.y_cont_indices else np.empty((n_samples, 0))
        )

        if invalid_observations:
            x_cat_gen = np.delete(x_cat_gen, invalid_observations, axis=0)
            x_cont_gen = np.delete(x_cont_gen, invalid_observations, axis=0)
            X_fixed_cat = np.delete(X_fixed_cat, invalid_observations, axis=0)
            if X_fixed_cont is not None:
                X_fixed_cont = np.delete(X_fixed_cont, invalid_observations, axis=0)

        return x_cat_gen, x_cont_gen, X_fixed_cat, X_fixed_cont

