from .utils.generic import ExplicitEnum


class DebugOption(ExplicitEnum):
    UNDERFLOW_OVERFLOW = "underflow_overflow"
    NPU_METRICS_DEBUG = "npu_metrics_debug"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
