import torch
import torch.nn as nn

def lambda_layer(queries, keys, embeddings, values):
    """Multi-query lambda layer."""
    # b: batch, n: input length, m: context length,
    # k: query/key depth, v: value depth,
    # h: number of heads, d: output dimension.
    content_lambda = torch.einsum(torch.softmax(keys), values, ’bmk, bmv->bkv’)
    position_lambdas = torch.einsum(embeddings, values, ’nmk,bmv->bnkv’)
    content_output = torch.einsum(queries, content_lambda, ’bhnk,bkv->bnhv’)
    position_output = torch.einsum(queries, position_lambdas, ’bhnk,bnkv->bnhv’)
    output = torch.reshape(content_output + position_output, [b, n, d])
    return output


class LambdaLayer(nn.Module):
    """
    LambdaLayer implementation
    """

    def __init__(self, input_size, context_size, number_filters, hidden_size, output_size):
        super(LambdaLayer, self).__init__()

        self.n = input_size
        self.m = context_size
        self.d = number_filters
        self.k = hidden_size
        self.v = output_size

        # Matrix translates context to keys
        self.weight_ck = None

        # Matrix translates context to values
        self.weight_cv = None

        # Matrix translates input to queries
        self.weight_xq = None

        # Matrix of position
        self.E = None

        # All parameters
        self.weight_ck = nn.Parameter(torch.Tensor(self.d, self.k))
        self.weight_cv = nn.Parameter(torch.Tensor(self.d, self.v))
        self.bias_xq = nn.Parameter(torch.Tensor(self.d, self.k))
        self.E = nn.Parameter(torch.Tensor(self.m, self.k, self.n))

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

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

    def forward(self, x):
        """
        Args:
            x: input with shape (N, T, D) where N is number of samples, T is
                number of timestep and D is input size which must be equal to
                self.input_size.

        Returns:
            y: output with a shape of (N, T, H) where H is hidden size
        """

        # Transpose input for efficient vectorized calculation. After transposing
        # the input will have (T, N, D).
        x = x.transpose(0, 1)

        # Unpack dimensions
        T, N, H = x.shape[0], x.shape[1], self.hidden_size

        # Initialize hidden and cell states to zero. There will be one hidden
        # and cell state for each input, so they will have shape of (N, H)
        h0 = torch.zeros(N, H, device=x.device)
        c0 = torch.zeros(N, H, device=x.device)

        # Define a list to store outputs. We will then stack them.
        y = []

        ########################################################################
        #                 TODO: Implement forward pass of LSTM                 #
        ########################################################################

        ht_1 = h0
        ct_1 = c0
        for t in range(T):
            # LSTM update rule
            xh = torch.addmm(self.bias_xh, x[t], self.weight_xh)
            hh = torch.addmm(self.bias_hh, ht_1, self.weight_hh)
            it = torch.sigmoid(xh[:, 0:H] + hh[:, 0:H])
            ft = torch.sigmoid(xh[:, H:2 * H] + hh[:, H:2 * H])
            gt = torch.tanh(xh[:, 2 * H:3 * H] + hh[:, 2 * H:3 * H])
            ot = torch.sigmoid(xh[:, 3 * H:4 * H] + hh[:, 3 * H:4 * H])
            ct = ft * ct_1 + it * gt
            ht = ot * torch.tanh(ct)

            # Store output
            y.append(ht)

            # For the next iteration c(t-1) and h(t-1) will be current ct and ht
            ct_1 = ct
            ht_1 = ht

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        # Stack the outputs. After this operation, output will have shape of
        # (T, N, H)
        y = torch.stack(y)

        # Switch time and batch dimension, (T, N, H) -> (N, T, H)
        y = y.transpose(0, 1)
        return y
