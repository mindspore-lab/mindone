import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

@dataclass
class CompileConfig:
    """
    Class that holds arguments relative to `mindspore.jit` behavior, when using automatic compilation in `generate`.
    See [`mindspore.jit`](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html) for more details on the arguments.

    Args:
        fullgraph (`bool`, *optional*, defaults to `False`):
            If False (default), attempts to discover compileable regions that will be optimized. If True, then require
            that the entire function be capturable into a single graph. If this is not possible (that is, if there are
            graph breaks), then an error will be raised.
        dynamic (`int` or `None`, *optional*):
            Whether to try to use dynamic shape graphs.
        backend (`str` or `Callable`, *optional*, defaults to `"ms_backend"`):
            Backend to be used.
        capture_mode (`str`, *optional*, defaults to `"ast"`):
        jit_level (`str`, *optional*, defaults to `"O0"`):
        options (`dict`, *optional*):
            A dictionary of options to pass to the backend.

    Examples:
    ```python
    >>> import mindspore as ms
    >>> from transformers import AutoTokenizer
    >>> from mindone.transformers import AutoModelForCausalLM, CompileConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

    >>> # Automatic compile configuration, used with static cache
    >>> compile_config = CompileConfig(dynamic=1)

    >>> # Generation with static cache and compile config
    >>> input = ms.tensor(tokenizer.encode("Hello there, how", return_tensors="np"))
    >>> output = model.generate(
    ...     input, do_sample=False, max_new_tokens=300, cache_implementation="static", compile_config=compile_config
    ... )
    >>> output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    ```
    """

    fullgraph: bool = False
    dynamic: Optional[int] = None
    backend: Union[str, Callable] = "ms_backend"
    capture_mode: str = "ast"
    jit_level: str = "O0"
    options: Optional[dict] = None
    # Used to flag our `generate` call to compile on e.g. CPU. Often not optimal, but useful for testing purposes.
    _compile_all_devices = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy({key: value for key, value in self.__dict__.items() if key != "_compile_all_devices"})
