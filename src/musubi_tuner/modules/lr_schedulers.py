import torch


class RexLR(torch.optim.lr_scheduler.LRScheduler):
    """
    Reflected Exponential (REX) learning rate scheduler (https://arxiv.org/abs/2107.04197)
    Modified from: https://github.com/IvanVassi/REX_LR (Apache-2.0 License)

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for
        max_lr (float): The maximum learning rate
        min_lr (float): The minimum learning rate
        num_steps (int): The total number of training steps
        num_warmup_steps (int): The number of warmup steps
        rex_alpha (float): Constant added to the denominator of the REX factor;
            prevents division-by-zero and softens the initial decay (default: 0.1).
        rex_beta (float): Multiplier of z in the denominator of the REX factor;
            controls how quickly the decay flattens as z increases (default: 0.9).
        last_epoch (int): The index of the last step
    """

    def __init__(self, optimizer, max_lr, min_lr=0.0, num_steps=0, num_warmup_steps=0, rex_alpha=0.1, rex_beta=0.9, last_epoch=-1):
        if min_lr > max_lr:
            raise ValueError(f'Value of "min_lr" should be less than value of "max_lr". Got min_lr={min_lr} and max_lr={max_lr}')
        if num_warmup_steps > num_steps:
            raise ValueError(f"num_warmup_steps ({num_warmup_steps}) must be less than or equal to num_steps ({num_steps})")

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        self.rex_alpha = rex_alpha
        self.rex_beta = rex_beta
        self.last_epoch = last_epoch

        # Ensure each parameter group has an "initial_lr" key to avoid issues when resuming
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Single warmup step
        if self.num_warmup_steps == 1 and self.last_epoch == 1:
            return [self.min_lr for _ in self.base_lrs]
        # Multiple warmup steps; increase lr linearly from min_lr to max_lr
        elif self.num_warmup_steps > 1 and self.last_epoch >= 1 and self.last_epoch <= (self.num_warmup_steps - 1):
            return [
                self.min_lr + (self.max_lr - self.min_lr) * (self.last_epoch - 1) / (self.num_warmup_steps - 1)
                for _ in self.base_lrs
            ]

        # Post-warmup phase: adjust step relative to the end of warmup
        step_after = self.last_epoch - self.num_warmup_steps
        remaining_steps = self.num_steps - self.num_warmup_steps

        # Avoid LR spiking
        if step_after >= remaining_steps or step_after == -1 or remaining_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        # Calculate REX curve for current step
        rex_z = (remaining_steps - (step_after % remaining_steps)) / remaining_steps
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (
            rex_z / (self.rex_alpha + self.rex_beta * rex_z)
        )

        return [base_lr * rex_factor for base_lr in self.base_lrs]
