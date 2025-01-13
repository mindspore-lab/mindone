from .backbone_utils import BackboneMixin
from .generic import (
    ContextManagers,
    ExplicitEnum,
    PaddingStrategy,
    TensorType,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_property,
    can_return_loss,
    expand_dims,
    filter_out_non_signature_kwargs,
    find_labels,
    flatten_dict,
    infer_framework,
    is_mindspore_dtype,
    is_mindspore_tensor,
    is_numpy_array,
    is_tensor,
    reshape,
    squeeze,
    strtobool,
    tensor_size,
    to_numpy,
    to_py_obj,
    torch_float,
    torch_int,
    transpose,
    working_or_temp_dir,
)
from .import_utils import (
    get_mindspore_version,
    is_mindspore_available,
    is_scipy_available,
    is_vision_available,
    requires_backends,
)

FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
CHAT_TEMPLATE_NAME = "chat_template.json"
PROCESSOR_NAME = "processor_config.json"
