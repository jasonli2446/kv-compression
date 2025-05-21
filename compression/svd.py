import torch


def svd_compress(tensor, rank):
    # tensor: (..., m, n)
    orig_shape = tensor.shape
    m, n = orig_shape[-2:]
    tensor_2d = tensor.reshape(-1, m, n)
    compressed = []
    for t in tensor_2d:
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        compressed.append((U_r, S_r, Vh_r))
    return compressed


def svd_decompress(compressed, orig_shape):
    # compressed: list of (U, S, Vh)
    tensors = []
    for U, S, Vh in compressed:
        t = U @ torch.diag(S) @ Vh
        tensors.append(t)
    return torch.stack(tensors).reshape(orig_shape)
