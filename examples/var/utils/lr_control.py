# This code is adapted from https://github.com/FoundationVision/VAR
# with modifications to run on MindSpore.

import math


def lr_wd_annealing(sche_type: str, peak_lr, warmup_it, max_it, wp0=0.005, wpe=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_it = round(warmup_it)
    lr_scheduler = []

    if warmup_it > max_it:
        print("Warning!!! warmup_it should be less than max_it")
        warmup_it = min(warmup_it, max_it)

    if warmup_it > 0:
        for i in range(warmup_it):
            lr = wp0 + (1 - wp0) * i / warmup_it
            lr *= peak_lr
            lr_scheduler.append(lr)

    main_it = max_it - warmup_it
    for i in range(main_it):
        pasd = i / (main_it - 1)
        rest = 1 - pasd
        if sche_type == "cos":
            lr = wpe + (1 - wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif sche_type == "lin":
            T = 0.15
            max_rest = 1 - T
            if pasd < T:
                lr = 1
            else:
                lr = wpe + (1 - wpe) * rest / max_rest  # 1 to wpe
        elif sche_type == "lin0":
            T = 0.05
            max_rest = 1 - T
            if pasd < T:
                lr = 1
            else:
                lr = wpe + (1 - wpe) * rest / max_rest
        elif sche_type == "lin00":
            lr = wpe + (1 - wpe) * rest
        elif sche_type.startswith("lin"):
            T = float(sche_type[3:])
            max_rest = 1 - T
            wpe_mid = wpe + (1 - wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            if pasd < T:
                lr = 1 + (wpe_mid - 1) * pasd / T
            else:
                lr = wpe + (wpe_mid - wpe) * rest / max_rest
        elif sche_type == "exp":
            T = 0.15
            max_rest = 1 - T
            if pasd < T:
                lr = 1
            else:
                expo = (pasd - T) / max_rest * math.log(wpe)
                lr = math.exp(expo)
        else:
            raise NotImplementedError(f"unknown sche_type {sche_type}")

        lr *= peak_lr
        lr_scheduler.append(lr)

    return lr_scheduler
