import torch
import torch.nn as nn
import numpy as np

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
        self.E = nn.Parameter(torch.Tensor(self.n, self.m, self.k), requires_grad=True)  # n-m-k

        # Create batchnormalization layers
        self.bn_values = nn.BatchNorm1d(self.m)
        self.bn_queries = nn.BatchNorm2d(self.k)

        # Keys softmax function
        self.softmax = nn.Softmax(dim=1)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        """
        Initialize network parameters.
        """
        torch.nn.init.normal_(self.E, mean=0.0, std=1.0)
        std_kv = 1/np.sqrt(self.d)
        std_q = 1/np.sqrt(self.d * self.k)
        torch.nn.init.normal_(self.toqueries.weight, mean=0.0, std=std_q)
        torch.nn.init.normal_(self.tokeys.weight, mean=0.0, std=std_kv)
        torch.nn.init.normal_(self.tovalues.weight, mean=0.0, std=std_kv)


    def forward(self, x, c):
        # Obtain the batch_size
        b, d, n1, n2 = x.size()     # b-d-n1-n2
        n = n1*n2
        x = torch.reshape(x, [b, n, d])

        # Compute the keys
        keys = self.tokeys(c)       # b-m-k
        softmax_keys = self.softmax(keys)

        # Compute the values
        values = self.bn_values(self.tovalues(c))   # b-m-v

        # Compute the queries
        # queries = torch.reshape(self.toqueries(x), [b, self.n, self.h, self.k])  # b-d-h-k
        queries = torch.reshape(self.toqueries(x), [b, n, self.k, self.h])  # b-n-k-h
        queries = torch.transpose(queries, 1, 2)  # b-k-n-h
        queries = self.bn_queries(queries)  # b-k-n-h
        queries = torch.transpose(queries, 1, 2)  # b-n-k-h
        queries = torch.transpose(queries, 2, 3)  # b-n-h-k
        queries = torch.transpose(queries, 1, 2)   # b-h-n-k

        # Compute lambdac
        content_lambda = torch.einsum('bmk, bmv->bkv', softmax_keys, values)    # b-k-v

        # Compute position lambda
        position_lambdas = torch.einsum('nmk, bmv->bnkv', self.E, values)       # b-n-k-v

        # Compute content output
        content_output = torch.einsum('bhnk, bkv->bnhv', queries, content_lambda)   # b-n-h-v

        # Compute position output
        position_output = torch.einsum('bhnk, bnkv->bnhv', queries, position_lambdas)   # b-n-h-v

        # Compute output
        output = torch.reshape(content_output + position_output, [b, n, d])   # b-n-d

        return output
