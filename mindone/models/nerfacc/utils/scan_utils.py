from mindspore import mint


def inclusive_sum(chunk_starts, chunk_cnts, inputs, normalize=False):
    assert chunk_starts.ndim == 1
    assert chunk_cnts.ndim == 1
    assert inputs.ndim == 1
    assert chunk_starts.size(0) == chunk_cnts.size(0)

    n_rays = chunk_cnts.size(0)

    current_sum = 0.0
    output_index = 0
    outputs = mint.zeros(chunk_cnts)
    for i in range(n_rays):
        start = chunk_starts[i].item()
        count = chunk_cnts[i].item()
        for j in range(start, start + count):
            outputs[output_index] = current_sum + inputs[j].item()
            current_sum = outputs[output_index]
            output_index += 1

    if normalize:
        outputs /= chunk_cnts.to(inputs.dtype).unsqueeze(1).expand_as(outputs)

    return outputs


def exclusive_sum(chunk_starts, chunk_cnts, inputs, normalize=False, backward=False):
    assert chunk_starts.ndim == 1
    assert chunk_cnts.ndim == 1
    assert inputs.ndim == 1
    assert chunk_starts.size(0) == chunk_cnts.size(0)
    if backward:
        assert not normalize  # backward no normalize

    n_rays = chunk_cnts.size(0)
    n_edges = inputs.size(0)

    current_sum = 0.0 if not backward else 0.0
    output_index = 0 if not backward else n_edges - 1
    outputs = mint.zeros(chunk_cnts)
    for i in range(n_rays):
        start = chunk_starts[i].item() if not backward else n_edges - (chunk_starts[i] + chunk_cnts[i]).item()
        count = chunk_cnts[i].item()
        if not backward:  # forward
            for j in range(start, start + count):
                if j == start:
                    outputs[output_index] = 0.0
                else:
                    outputs[output_index] = current_sum
                current_sum += inputs[j].item()
                output_index += 1
        else:  # backward
            for j in range(start + count - 1, start - 1, -1):
                if j == start + count - 1:
                    outputs[output_index] = current_sum
                else:
                    outputs[output_index] = current_sum - inputs[j].item()
                current_sum = outputs[output_index]
                output_index -= 1

    return outputs


def inclusive_prod_forward():
    pass


def inclusive_prod_backward():
    pass


def exclusive_prod_forward():
    pass


def exclusive_prod_backward():
    pass
