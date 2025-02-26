"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Runtime type checking decorator
from typeguard import typechecked as typechecker

# Mindspore Tensor type
from mindspore import Tensor

# pre-commit fail to detect they are cited in other scripts
if False:
    print(
        Any,
        Callable,
        Dict,
        Iterable,
        List,
        Literal,
        NamedTuple,
        NewType,
        Optional,
        Sized,
        Tuple,
        Type,
        TypeVar,
        Union,
        typechecker,
        Bool,
        Complex,
        Float,
        Inexact,
        Int,
        Integer,
        Num,
        Shaped,
        UInt,
        Tensor,
    )
