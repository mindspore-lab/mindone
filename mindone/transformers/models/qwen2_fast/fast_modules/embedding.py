import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, Normal


class FastEmbedding(nn.Cell):
    def __init__(self, vocab_table_size, embedding_size, init_method_std=0.01, param_init_type=mindspore.float32,
                 param_init='normal', parallel_optimizer=False, rmsnorm_compute_2d=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        if param_init == "normal":
            param_init = Normal(sigma=init_method_std, mean=0.0)
            print(f"Embedding use init method: sigma={init_method_std}, mean=0.0")
        self.weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='weight', parallel_optimizer=parallel_optimizer)
        self.rmsnorm_compute_2d = rmsnorm_compute_2d
        self.gather = P.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        if self.rmsnorm_compute_2d:
            input_ids = input_ids.reshape(-1)
        output = self.gather(self.weight, input_ids, 0)
        return output
