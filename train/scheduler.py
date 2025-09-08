import numpy as np


def assign_learning_rate(optimizer, new_lrs):
   for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
        param_group["lr"] = new_lr
       


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    def _lr_adjuster(step):
        lrs = []
        for base_lr in base_lrs:
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            lrs.append(lr)
        assign_learning_rate(optimizer, lrs)
        return lrs
    return _lr_adjuster
