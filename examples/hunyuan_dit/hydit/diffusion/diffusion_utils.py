import math

from mindspore import ops

SCALE_FACTOR = float(math.sqrt(2.0 / math.pi))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for ms.exp().
    return 0.5 * (-1.0 + logvar2 - logvar1 + ops.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * ops.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + ops.tanh(SCALE_FACTOR * (x + 0.044715 * ops.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = ops.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = ops.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = ops.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = ops.where(
        x < -0.999,
        log_cdf_plus,
        ops.where(x > 0.999, log_one_minus_cdf_min, ops.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs
