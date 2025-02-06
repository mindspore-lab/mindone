import inspect


def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    signature = inspect.signature(model_class.construct)  # MindSpore models

    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False


def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    signature = inspect.signature(model_class.construct)  # MindSpore models

    if "QuestionAnswering" in model_name:
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        return [p for p in signature.parameters if "label" in p]
