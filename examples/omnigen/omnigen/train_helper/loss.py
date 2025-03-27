from mindspore import ops


def sample_x0(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [ops.randn_like(img_start) for img_start in x1]
    else:
        x0 = ops.randn_like(x1)

    return x0


def sample_timestep(x1):
    u = ops.normal(mean=0.0, stddev=1.0, shape=(len(x1),))
    t = 1 / (1 + ops.exp(-u))
    t = t.to(x1[0].dtype)
    return t
