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