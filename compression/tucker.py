try:
    import tensorly as tl
    from tensorly.decomposition import tucker
except ImportError:
    tl = None


def tucker_compress(tensor, ranks):
    if tl is None:
        raise ImportError("tensorly is required for Tucker compression")
    core, factors = tucker(tensor, ranks=ranks)
    return core, factors


def tucker_decompress(core, factors):
    if tl is None:
        raise ImportError("tensorly is required for Tucker decompression")
    return tl.tucker_to_tensor((core, factors))
