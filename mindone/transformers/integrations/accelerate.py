from mindspore import nn


def find_tied_parameters(model: "nn.Cell", **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`mindspore.nn.Cell`): The model to inspect.

    Returns:
        list[list[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import mindspore.nn as nn
    >>> from mindspore import mint

    >>> model = nn.SequentialCell(OrderedDict([("linear1", mint.nn.Linear(4, 4)), ("linear2", mint.nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """

    # get ALL model parameters and their names
    all_named_parameters = dict(model.name_cells())

    # get ONLY unique named parameters,
    # if parameter is tied and have multiple names, it will be included only once
    no_duplicate_named_parameters = dict(model.name_cells())

    # the difference of the two sets will give us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    # 'tied_param_names' contains the names of parameters that are tied in the model, but we do not know
    # which names refer to the same parameter. To identify this, we need to group them together.
    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in no_duplicate_named_parameters.items():
            # compare if parameters are the same, if so, group their names together
            if param is tied_param:
                if param_name not in tied_param_groups:
                    tied_param_groups[param_name] = []
                tied_param_groups[param_name].append(tied_param_name)

    return [sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()]
