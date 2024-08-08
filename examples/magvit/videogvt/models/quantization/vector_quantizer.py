import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Uniform
import numpy as np


class VQ(nn.Cell):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VQ, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(
            self.n_e, self.e_dim, embedding_table=Uniform(1.0 / self.n_e)
        )
        # self.embedding.weight.data = nn.probability.distribution.Uniform(-1.0 / self.n_e, 1.0 / self.n_e)

    def construct(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, depth, channel) and flatten
        # z = z.permute(0, 2, 3, 1) # 2d
        z = z.permute(0, 2, 3, 4, 1)

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            ops.sum(z_flattened.pow(2), dim=1, keepdim=True)
            + ops.sum(self.embedding.embedding_table.value().pow(2), dim=1)
            - 2 * ops.matmul(z_flattened, self.embedding.embedding_table.value().t())
        )

        # find closest encodings
        min_encoding_indices = ops.argmin(d, axis=1).unsqueeze(1)
        min_encodings = ops.zeros((min_encoding_indices.shape[0], self.n_e))
        min_encoding_src = ops.ones_like(
            min_encoding_indices, dtype=min_encodings.dtype
        )
        min_encodings = ops.scatter(
            input=min_encodings,
            axis=1,
            index=min_encoding_indices,
            src=min_encoding_src,
        )
        # min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = ops.matmul(min_encodings, self.embedding.embedding_table.value()).view(
            z.shape
        )

        # compute loss for embedding
        loss = ops.mean((ops.stop_gradient(z_q) - z).pow(2)) + self.beta * ops.mean(
            (z_q - ops.stop_gradient(z)).pow(2)
        )

        # preserve gradients
        z_q = z + ops.stop_gradient(z_q - z)

        # perplexity
        e_mean = ops.mean(min_encodings, axis=0)
        perplexity = ops.exp(-ops.sum(e_mean * ops.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2)
        z_q = z_q.permute(0, 4, 1, 2, 3)  # 3d

        # return loss, z_q, perplexity, min_encodings, min_encoding_indices
        return z_q, loss


class ExponentialMovingAverage(nn.Cell):
    """Maintains an exponential moving average for a value.

    This module keeps track of a hidden exponential moving average that is
    initialized as a vector of zeros which is then normalized to give the average.
    This gives us a moving average which isn't biased towards either zero or the
    initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

    Initially:
        hidden_0 = 0
    Then iteratively:
        hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
        average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.hidden = ms.Parameter(ops.zeros_like(init_value), requires_gradient=False)
        # self.register_buffer("hidden", ops.zeros_like(init_value))

    def construct(self, value):
        self.counter += 1
        self.hidden = self.hidden.sub((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay**self.counter)
        return average


class VectorQuantizerEMA(nn.Cell):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(
        self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = ops.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(
            ops.zeros((self.num_embeddings,)), decay
        )

    def construct(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, H, W, C]

        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(
                torch.sum(encodings, dim=0)
            )
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            dw = torch.matmul(
                encodings.t(), flat_x
            )  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            )
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2.0 * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
