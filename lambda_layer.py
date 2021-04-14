import torch
import torch.nn as nn
import numpy as np


class LambdaLayer(nn.Module):
    '''
    Multi-query lambda layer. The code has been written with the assumption that global-context is used.
    In that case, the first LambdaLayer in the ResNet-50 architecture observes an 8x8 image
    '''
    def __init__(self, context_size, value_size, qk_size, output_size, heads, E):
        '''
        Constructor of the LambdaLayer. Creation of all internal layers and call parameter initialisation
        Args:
            context_size: size of the contect m
            value_size: size of the value dimension v
            qk_size: size of the query and key dimension k
            output_size: number of output dimensions d
            heads: number of heads h
            E: shared embeddings
        '''
        super(LambdaLayer, self).__init__()

        # Initialisation of the LambdaLayer dimensions
        self.m = context_size
        self.d = output_size
        self.k = qk_size
        self.v = value_size
        self.h = heads

        # Linear mapping in the form of NN for the query, key and value computation.
        self.toqueries = nn.Linear(self.d, self.k * self.h, bias=False)
        self.tokeys    = nn.Linear(self.d, self.k, bias=False)
        self.tovalues  = nn.Linear(self.d, self.v, bias=False)
        self.E = E    # n-m-k

        # Create batch normalization layers
        self.bn_values = nn.BatchNorm1d(self.m)
        self.bn_queries = nn.BatchNorm2d(self.k)

        # Keys softmax function for the keys
        self.softmax = nn.Softmax(dim=1)

        # Initialize parameters
        self.reset_params()

    def reset_params(self):
        """
        Initialize network parameters.
        """
        std_kv = 1/np.sqrt(self.d)  # standard deviation for the key and value
        std_q = 1/np.sqrt(self.d * self.k)  # standard deviation for the query
        torch.nn.init.normal_(self.toqueries.weight, mean=0.0, std=std_q)  # initialise of the query projection matrix
        torch.nn.init.normal_(self.tokeys.weight, mean=0.0, std=std_kv)  # initialise of the keys projection matrix
        torch.nn.init.normal_(self.tovalues.weight, mean=0.0, std=std_kv)  # initialise of the values projection matrix

    def forward(self, x):
        '''
        Forward propagation of the LambdaLayer
        Args:
            x: input

        Returns:

        '''

        # Obtain the batch_size
        b, d, n1, n2 = x.size()            # b-d-n1-n2
        n = n1*n2                          # compute the number of pixels in the input
        x = torch.reshape(x, [b, n, d])    # b-n-d

        # Reshape the context
        c = torch.reshape(x, [b, self.m, d])

        # Compute the keys
        keys = self.tokeys(c)                   # b-m-k
        softmax_keys = self.softmax(keys)       # b-m-k

        # Compute the values
        values = self.bn_values(self.tovalues(c))   # b-m-v

        # Compute the queries
        queries = torch.reshape(self.toqueries(x), [b, n, self.k, self.h])  # b-n-k-h
        queries = torch.transpose(queries, 1, 2)  # b-k-n-h
        queries = self.bn_queries(queries)  # b-k-n-h

        # Compute content lambda
        content_lambda = torch.einsum('bmk, bmv->bkv', softmax_keys, values)    # b-k-v

        # Compute position lambda
        position_lambdas = torch.einsum('nmk, bmv->bnkv', self.E, values)       # b-n-k-v

        # Compute content output
        content_output = torch.einsum('bknh, bkv->bhvn', queries, content_lambda)   # b-h-v-n

        # Compute position output
        position_output = torch.einsum('bknh, bnkv->bhvn', queries, position_lambdas)   # b-h-v-n

        # Compute output
        output = torch.reshape(content_output + position_output, [b, d, n])   # b-d-n

        # Reshape as an image
        output = torch.reshape(output, [b, d, n1, n2])   # b-d-n1-n2

        return output
