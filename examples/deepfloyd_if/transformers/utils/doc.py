# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Doc utilities: Utilities related to documentation
"""

import functools
import re
import types


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = output_docstring.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = "\n".join(lines[(i + 1) :])
        params_docstring = _convert_output_args_doc(params_docstring)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    result = intro + params_docstring

    # Apply minimum indent if necessary
    if min_indent is not None:
        lines = result.split("\n")
        # Find the indent of the first nonempty line
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # If too small, add indentation to all nonempty lines
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
