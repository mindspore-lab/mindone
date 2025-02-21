import mindspore as ms
from mindspore import Tensor, mint

ms.set_context(mode=0)
ms.set_context(jit_config={"jit_level": "O0"})

probs = Tensor([0.6, 0.3, 0.1], ms.float32)
next_token = mint.multinomial(probs, num_samples=1, replacement=False)

print(next_token)
