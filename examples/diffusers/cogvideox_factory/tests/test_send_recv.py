import mindspore as ms
from mindspore import mint
from mindspore.communication import get_group_size, get_rank, init

# from mindspore.mint.distributed import isend, irecv
from mindspore.communication.comm_func import irecv, isend

init()
rank = get_rank()
group_size = get_group_size()
x = mint.ones([2, 2], dtype=ms.bfloat16) * rank
dst = (rank + 1) % group_size
src = (rank - 1) % group_size
tag = 0
print("=== Start!")
if rank != group_size - 1:
    tag = 0
    print(f"= rank {rank} === isend to {dst} tag {tag}, value: {x}")
    handle_send = isend(x, dst=dst, tag=tag)
    tag = 1
    print(f"= rank {rank} === isend to {dst} tag {tag}, value: {x*2}")
    handle_send = isend(x * 2, dst=dst, tag=tag)

if rank != 0:
    tag = 1
    # y = mint.zeros_like(x)
    # handle_recv0 = irecv(y, src=src, tag=tag)
    y, handle_recv0 = irecv(x, src=src, tag=tag)
    print(f"= rank {rank} === irecv from {src} tag {tag}, value: {y}")
    tag = 0
    # y = mint.zeros_like(x)
    # handle_recv1 = irecv(y, src=src, tag=tag)
    y, handle_recv1 = irecv(x, src=src, tag=tag)
    print(f"= rank {rank} === irecv from {src} tag {tag}, value: {y}")
    handle_recv0.wait()
    handle_recv1.wait()
