import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def cycle(dl):
    while True:
        for data in dl:
            yield data
            

def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    Returns a set num_samples values in [0,1) that are evenly spaced with a random shift.
    This covers timesteps and noise more evenly than plain random draws as well as avoiding repeating patterns.
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float64)
    return (
        single_u
        + torch.arange(
            0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False
        )
    ) % 1


class LinearScheduler:
    """
    The LinearScheduler returns the learning rate for a given training step. It linearly increases from warmup_begin_lr to base_lr over warmup_steps.
    If anneal_lr=True it linearly decays from base_lr to final_lr until max_update, otherwise it stays at base_lr.
    """
    def __init__(
        self,
        max_update, 
        base_lr=0.1, 
        final_lr=0.0,
        warmup_steps=0,
        warmup_begin_lr=0, 
        anneal_lr=False, 
    ):
        self.anneal_lr = anneal_lr
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, step):
        increase = (
            (self.base_lr_orig - self.warmup_begin_lr)
            * float(step)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
        if (step <= self.max_update) and self.anneal_lr:
            decrease = (
                (self.final_lr - self.base_lr_orig)
                / (self.max_update - self.warmup_steps)
                * (step - self.warmup_steps)
            )
        return decrease + self.base_lr_orig if self.anneal_lr else self.base_lr_orig


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Adapted from: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

    The FastTensorDataLoader supports shuffling (one permutation per epoch) and drop_last (truncate to a multiple of batch_size). When drop_last is False, the last batch may be smaller.
    With shuffle=True it gathers shuffled rows via their index using index_select(dim=0, indices), otherwise it returns slices [i:i+batch_size]. 
    Output: each iteration yields a tuple (X_cat_batch, X_cont_batch[, y_cond_1_batch, y_cond_2_batch]), where missing inputs are returned as None. len returns the number of batches per epoch.

    CDTD-CFG Extension: the loader now also accepts y_condition_1 and y_condition_2 to create batches, using the same indices/slices as X_cat/X_cont (both shuffled and sequential). 
    They are returned in addition to the feature tensors so the conditioning labels remain alligned.
    """
    def __init__(self, X_cat, X_cont, batch_size=32, shuffle=False, drop_last=False, y_condition_1 = None, y_condition_2 = None,):
        self.dataset_len = X_cat.shape[0] if X_cat is not None else X_cont.shape[0] 
        assert all(
            t.shape[0] == self.dataset_len for t in (X_cat, X_cont) if t is not None
        )
        self.X_cat = X_cat
        self.X_cont = X_cont
        
        self.y_condition_1 = y_condition_1
        self.y_condition_2 = y_condition_2

        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last:
            self.dataset_len = (self.dataset_len // self.batch_size) * self.batch_size

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1 
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len) 
        else:
            self.indices = None
        self.i = 0 
        return self

    def __next__(self):
        if self.i >= self.dataset_len: 
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size] 
            batch = {}
            batch["X_cat"] = (
                torch.index_select(self.X_cat, 0, indices)
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                torch.index_select(self.X_cont, 0, indices)
                if self.X_cont is not None
                else None
            )
            batch["y_cond_1_batch"] = (
                torch.index_select(self.y_condition_1, 0, indices)
                if self.y_condition_1 is not None
                else None
            )
            batch["y_cond_2_batch"] = (
                torch.index_select(self.y_condition_2, 0, indices)
                if self.y_condition_2 is not None
                else None
            )

        else:
            batch = {}
            batch["X_cat"] = (
                self.X_cat[self.i : self.i + self.batch_size]
                if self.X_cat is not None
                else None
            )
            batch["X_cont"] = (
                self.X_cont[self.i : self.i + self.batch_size]
                if self.X_cont is not None
                else None
            )
            batch["y_cond_1_batch"] = (
                self.y_condition_1[self.i : self.i + self.batch_size]
                if self.y_condition_1 is not None
                else None
            )
            batch["y_cond_2_batch"] = (
                self.y_condition_2[self.i : self.i + self.batch_size]
                if self.y_condition_2 is not None
                else None
            )

        self.i += self.batch_size

        batch = tuple(batch.values()) 
        return batch

    def __len__(self):
        return self.n_batches 
