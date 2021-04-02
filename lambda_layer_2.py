import torch
import torch.nn as nn
import numpy as np

class LambdaLayer(nn.Module):
    """Multi-query lambda layer. The code has been written with the assumption that substitution
     of the 3x3 conv layers by the lambda layers does not change the image size. Which in the case
      of ResNet50 is 8x8."""

    def __init__(self, input_size, context_size, value_size, qk_size, output_size, heads, E):
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
        self.E = E    #n-m-k

        # Create batch normalization layers
        self.bn_values = nn.BatchNorm1d(self.n)
        self.bn_queries = nn.BatchNorm2d(self.k)

        # Keys softmax function
        self.softmax = nn.Softmax(dim=1)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        """
        Initialize network parameters.
        """
        # torch.nn.init.normal_(self.E, mean=0.0, std=1.0)
        std_kv = 1/np.sqrt(self.d)
        std_q = 1/np.sqrt(self.d * self.k)
        torch.nn.init.normal_(self.toqueries.weight, mean=0.0, std=std_q)
        torch.nn.init.normal_(self.tokeys.weight, mean=0.0, std=std_kv)
        torch.nn.init.normal_(self.tovalues.weight, mean=0.0, std=std_kv)

    def forward(self, x):
        # Obtain the batch_size
        b, d, n1, n2 = x.size()     # b-d-n1-n2
        n = n1*n2
        x = torch.reshape(x, [b, n, d])

        # Compute the keys
        keys = self.tokeys(x)       # b-n-k
        softmax_keys = self.softmax(keys)

        # Compute the values
        values = self.bn_values(self.tovalues(x))   # b-n-v

        # Compute the queries
        queries = torch.reshape(self.toqueries(x), [b, n, self.k, self.h])  # b-n-k-h
        queries = torch.transpose(queries, 1, 2)  # b-k-n-h
        queries = self.bn_queries(queries)  # b-k-n-h

        # Compute lambdac
        content_lambda = torch.einsum('bnk, bnv->bkv', softmax_keys, values)    # b-k-v

        # Compute position lambda
        values = torch.reshape(values, [b, 1, self.n, self.v])   # b-u-n-v
        values = torch.transpose(values, 2, 3)  # b-u-v-n
        values = torch.reshape(values, [b, 1, self.v, n1, n2])  # b-u-v-n1-n2
        position_lambdas = self.E(values)  # b-k-v-n1-n2
        position_lambdas = position_lambdas.flatten(3)  # b-k-v-n

        # Compute content output
        content_output = torch.einsum('bknh, bkv->bnhv', queries, content_lambda)   # b-n-h-v

        # Compute position output
        position_output = torch.einsum('bknh, bkvn->bnhv', queries, position_lambdas)   # b-n-h-v

        # Compute output
        output = torch.reshape(content_output + position_output, [b, n, d])   # b-n-d

        # Reshape as an image
        output = torch.transpose(output, 1, 2)   # b-d-n
        output = torch.reshape(output, [b, d, n1, n2])   # b-d-n1-n2

        return output