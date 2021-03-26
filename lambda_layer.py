import torch
import torch.nn as nn

def lambda_layer(queries, keys, embeddings, values):
    """Multi-query lambda layer."""
    # b: batch, n: input length, m: context length,
    # k: query/key depth, v: value depth,
    # h: number of heads, d: output dimension.
    content_lambda = torch.einsum(torch.softmax(keys), values, 'bmk, bmv->bkv')
    position_lambdas = torch.einsum(embeddings, values, ’nmk,bmv->bnkv’)
    content_output = torch.einsum(queries, content_lambda, ’bhnk,bkv->bnhv’)
    position_output = torch.einsum(queries, position_lambdas, ’bhnk,bnkv->bnhv’)
    output = torch.reshape(content_output + position_output, [b, n, d])
    return output


class LambdaLayer(nn.Module):
    """Multi-query lambda layer."""

    def __init__(self, input_size, context_size, value_size, qk_size, output_size, heads):
        super(LambdaLayer, self).__init__()

        self.n = input_size
        self.m = context_size
        self.d = output_size
        self.k = qk_size
        self.v = value_size
        self.h = heads

        # These compute the queries, keys and values
        self.toqueries = nn.Linear(self.d, self.k * self.h, bias=False)
        self.tokeys    = nn.Linear(self.d, self.k, bias=False)
        self.tovalues  = nn.Linear(self.d, self.v, bias=False)
        self.E = nn.Parameter(torch.Tensor(self.n, self.m, self.k))  # n-m-k

        # Keys softmax function
        self.softmax = nn.Softmax(dim=1)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        """
        Initialize network parameters.
        """

        std = 1.0 / math.sqrt(self.hidden_size)
        self.weight_xh.data.uniform_(-std, std)
        self.weight_hh.data.uniform_(-std, std)
        self.bias_xh.data.uniform_(-std, std)
        self.bias_hh.data.uniform_(-std, std)

    def forward(self, x, c):
        # Obtain the batch_size
        b, _, _ = x.size()     # b-n-d

        # Compute the keys, values and queries
        keys = self.tokeys(c)       # b-m-k
        values = self.tovalues(c)   # b-m-v
        queries = torch.reshape(self.toqueries(x), [b, self.n, self.h, self.k])  # b-d-h-k

        # Obtain the right query shape
        queries = torch.transpose(queries, 1, 2)   # b-h-n-k

        # Compute lambdac
        softmax_keys = self.softmax(keys)
        content_lambda = torch.einsum('bmk, bmv->bkv', softmax_keys, values)    # b-k-v

        # Compute position lambda
        position_lambdas = torch.einsum('nmk, bmv->bnkv', self.E, values)       # b-n-k-v

        # Compute content output
        content_output = torch.einsum('bhnk, bkv->bnhv', queries, content_lambda)   # b-n-h-v

        # Compute position output
        position_output = torch.einsum('bhnk, bnkv->bnhv', queries, position_lambdas)   # b-n-h-v

        # Compute output
        output = torch.reshape(content_output + position_output, [b, self.n, self.d])   # b-n-d

        return output
