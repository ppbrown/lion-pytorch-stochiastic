import math
import torch
from torch.optim import Optimizer

def stochastic_round(source_fp32: torch.Tensor) -> torch.Tensor:
    """
    Convert a float32 tensor to bfloat16 with stochastic rounding.
    Each element is randomly rounded up or down such that the expectation
    matches the original float32 value (unbiased rounding).
    """
    assert source_fp32.dtype == torch.float32, "Input must be float32 for stochastic rounding."

    # Create a random int32 tensor with the same shape
    # Each entry is in [0, 2^16), representing the random offset in the lower 16 bits
    r = torch.randint(
        low=0,
        high=(1 << 16),
        size=source_fp32.shape,
        dtype=torch.int32,
        device=source_fp32.device
    )

    # View the float32 values as int32 to manipulate bits
    source_int = source_fp32.view(torch.int32)

    # Add the random offset to the int32 representation
    # This effectively decides the rounding direction
    result_int = source_int + r

    # Mask out the lower 16 bits to truncate to bfloat16 precision
    result_int = result_int & ~0xFFFF  # keep the top 16 bits, zero out the bottom 16

    # Convert back to float32, then to bfloat16
    result_fp32 = result_int.view(torch.float32)
    return result_fp32.to(torch.bfloat16)

class LionStochastic(Optimizer):
    r"""
    Implements Lion optimizer from the paper "Symbolic Discovery of Optimization Algorithms"
    by Chen et al. (2023), with optional stochastic rounding for bfloat16 parameters.

    Reference: https://github.com/lucidrains/lion-pytorch

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its magnitude (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to use decoupled weight decay
            as in AdamW (default: False)
        stochastic_rounding (bool, optional): whether to enable stochastic rounding
            for parameters in bfloat16 precision (default: False)

    Usage:
        from lion_stoch import LionStochastic

        optimizer = LionStochastic(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            decoupled_weight_decay=False,
            stochastic_rounding=True
        )
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        decoupled_weight_decay=False,
        stochastic_rounding=False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            stochastic_rounding=stochastic_rounding
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            decoupled_wd = group['decoupled_weight_decay']
            stoch_round_enabled = group['stochastic_rounding']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer if not present
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']

                # Optional: handle any FP16 -> FP32 conversions if needed,
                # but we specifically handle BF16 here.
                # We'll only do special logic if p.dtype == bfloat16 and stoch_round_enabled is True.
                if stoch_round_enabled and p.dtype == torch.bfloat16:
                    # Convert to float32 for higher precision arithmetic
                    p_fp32 = p.data.float()
                    grad_fp32 = grad.detach().float()
                    exp_avg_fp32 = exp_avg.data.float()

                    # Weight decay
                    if decoupled_wd:
                        # Decoupled weight decay (like AdamW)
                        p_fp32.mul_(1 - lr * wd)
                    elif wd != 0:
                        # L2 weight decay (coupled)
                        grad_fp32.add_(p_fp32, alpha=wd)

                    # Update exponential moving average
                    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    exp_avg_fp32.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)

                    # Compute sign of momentum
                    update_fp32 = exp_avg_fp32.sign()

                    # Apply the update to parameters
                    p_fp32.add_(update_fp32, alpha=-lr)

                    # Update exp_avg for the second momentum
                    # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                    exp_avg_fp32.mul_(beta2).add_(grad_fp32, alpha=1 - beta2)

                    # Stochastic rounding back to bfloat16
                    p.data.copy_(stochastic_round(p_fp32))
                    exp_avg.data.copy_(stochastic_round(exp_avg_fp32))

                else:
                    # Default path (no stochastic rounding or not bfloat16)
                    if decoupled_wd:
                        # Decoupled weight decay
                        p.data.mul_(1 - lr * wd)
                    elif wd != 0:
                        # L2 weight decay
                        grad = grad.add(p, alpha=wd)

                    # Update exponential moving average
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Sign of momentum
                    update = exp_avg.sign()

                    # Weight update
                    p.add_(update, alpha=-lr)

                    # Momentum update
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
