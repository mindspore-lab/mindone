from mindspore import nn


def recompute_except_output(cell: nn.Cell, **recompute_kwargs):
    if not cell._has_config_recompute:
        cell.recompute(**recompute_kwargs)
    if isinstance(cell, nn.CellList):
        recompute_except_output(cell[-1])
    else:
        cell.add_flags(output_no_recompute=True)
