"""Meta learning rate scheduler.

This module implements exactly the same learning rate scheduler as native PyTorch,
see `"torch.optim.lr_scheduler" <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
At present, only`linear_lr`, `polynomial_lr`, `multi_step_lr` and 'cosine_decay_lr' are implemented.
The number, name and usage of the Positional Arguments are exactly the same as those of native PyTorch.

"""
import math
from bisect import bisect_right


def linear_refined_lr(start_factor, end_factor, warmup_steps, *, lr, total_steps):
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(total_steps):
        multiplier = min(i, warmup_steps) / warmup_steps
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def polynomial_refined_lr(decay_steps, power, *, lr, end_lr, total_steps):
    lrs = []
    for i in range(total_steps):
        lrs.append((lr - end_lr) * (1 - min(i, decay_steps) / decay_steps) ** power + end_lr)
    return lrs


def multi_step_lr(milestones, gamma, *, lr, total_steps):
    milestones = sorted(milestones)
    lrs = []
    for i in range(total_steps):
        lrs.append(lr * gamma ** bisect_right(milestones, i))
    return lrs


def cosine_decay_refined_lr(decay_steps, eta_min, *, eta_max, total_steps, num_cycles=1, cycle_decay=1.0):
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(decay_steps):
            t_cur = min(i, decay_steps)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_steps))
            if len(lrs) < total_steps:
                lrs.append(lr_cur)
            else:
                break

    if total_steps > num_cycles * decay_steps:
        for i in range(total_steps - (num_cycles * decay_steps)):
            lrs.append(eta_min)

    return lrs


if __name__ == "__main__":
    # Demonstrate how these schedulers work by printing & visualizing the returned list.
    import matplotlib.pyplot as plt

    table = (
        (
            (
                "linear_refined_lr",
                linear_refined_lr(0.0, 1.0, 20, lr=0.05, total_steps=20),
            ),
        ),
        (
            (
                "polynomial_refined_lr",
                polynomial_refined_lr(4, 1.0, lr=0.05, end_lr=0.01, total_steps=20),
            ),
        ),
        (
            (
                "multi_step_lr",
                multi_step_lr([3, 6], 0.5, lr=0.05, total_steps=20),
            ),
        ),
        (
            (
                "cosine_decay_refined_lr",
                cosine_decay_refined_lr(20, 0.1, eta_max=2.0, total_steps=20),
            ),
        ),
    )
    for variants in table:
        n_variants = len(variants)
        fig = plt.figure(figsize=(4, 3 * n_variants))
        for ax_idx, (title, lrs_ms) in enumerate(variants, start=1):
            print(f"name: {title}\nlrs: {lrs_ms}")
            ax = plt.subplot(n_variants, 1, ax_idx)
            ax.plot(lrs_ms, marker="*")
            ax.set_title(title)
            ax.set_xlim(0, len(lrs_ms))  # n_steps
            ax.set_xlabel("step")
            ax.set_ylabel("lr")
        plt.tight_layout()

    # Compare the difference between cosine_annealing_lr and cosine_annealing_warm_restarts_lr.
    plt.figure()
    lrs_ms = polynomial_refined_lr(4, 1.0, lr=0.05, end_lr=0.01, total_steps=20)
    plt.plot(lrs_ms)
    lrs_ms = cosine_decay_refined_lr(20, 0.1, eta_max=2.0, total_steps=20)
    plt.plot(lrs_ms)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["polynomial_refined_lr", "cosine_decay_refined_lr"], loc="best")
    plt.title("polynomial_refined_lr vs. cosine_decay_refined_lr")
    plt.show()
# fmt: on
