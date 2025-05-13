from typing import Optional, Tuple

from transformers import PretrainedConfig

import mindspore as ms
from mindspore import Tensor, mint


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None, seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple[Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (mint.arange(0, dim, 2, dtype=ms.int32).float() / dim))
    return inv_freq, attention_factor


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )


def _validate_longrope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "short_factor", "long_factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        logger.warning(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if not len(short_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        logger.warning(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if not len(long_factor) == dim // 2:
        logger.warning(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_scaling.get("factor")
        if factor is None:
            logger.warning("Missing required keys in `rope_scaling`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                logger.warning(
                    f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
                )


def _validate_llama3_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        logger.warning(
            "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= config.max_position_embeddings:
        logger.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}

def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    """
    Validate the RoPE config arguments, given a `PretrainedConfig` object
    """
    rope_scaling = getattr(config, "rope_scaling", None)  # not a default parameter in `PretrainedConfig`
    if rope_scaling is None:
        return

    # BC: "rope_type" was originally "type"
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config, ignore_keys=ignore_keys)
    else:
        logger.warning(
            f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
        )
