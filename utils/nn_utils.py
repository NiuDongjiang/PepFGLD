#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_sum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


@torch.no_grad()
def length_to_batch_id(S, lengths):
    # generate batch id
    batch_id = torch.zeros_like(S)  # [N]
    batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
    batch_id.cumsum_(dim=0)  # [N], item idx in the batch
    return batch_id


def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
    # f_src = src.float()
    # f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    # perm = norm.argsort(dim=dim, descending=descending)

    # return src[perm], perm
    src, src_perm = torch.sort(src, dim=dim, descending=descending)
    index = index.take_along_dim(src_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    src = src.take_along_dim(index_perm, dim=dim)
    perm = src_perm.take_along_dim(index_perm, dim=0)
    return src, perm


def scatter_topk(src: torch.Tensor, index: torch.Tensor, k: int, dim=0, largest=True):
    indices = torch.arange(src.shape[dim], device=src.device)
    src, perm = scatter_sort(src, index, dim, descending=largest)
    index, indices = index[perm], indices[perm]
    mask = torch.ones_like(index).bool()
    mask[k:] = index[k:] != index[:-k]
    return src[mask], indices[mask]


@torch.no_grad()
def knn_edges(all_edges, k_neighbors, X=None, atom_mask=None, given_dist=None):
    '''
    :param all_edges: [2, E], (row, col)
    :param X: [N, n_channel, 3], coordinates
    :param atom_mask: [N, n_channel], 1 for having atom
    :param given_dist: [E], given distance of edges
    IMPORTANT: either given_dist should be given, or both X and atom_mask should be given
    '''
    assert (given_dist is not None) or (X is not None and atom_mask is not None), \
        'either given_dist should be given, or both X and atom_mask should be given'

    # get distance on each edge
    if given_dist is None:
        row, col = all_edges
        dist = torch.norm(X[row][:, :, None, :] - X[col][:, None, :, :], dim=-1)  # [E, n_channel, n_channel]
        dist_mask = atom_mask[row][:, :, None] & atom_mask[col][:, None, :]  # [E, n_channel, n_channel]
        dist = torch.where(dist_mask, dist, torch.ones_like(dist) * float('inf'))  # [E, n_channel, n_channel]
        dist, _ = dist.view(dist.shape[0], -1).min(axis=-1) # [E]
    else:
        dist = given_dist

    # get topk for each node
    _, indices = scatter_topk(dist, row, k=k_neighbors, largest=False)
    edges = torch.stack([all_edges[0][indices], all_edges[1][indices]], dim=0) # [2, k*N]
    return edges  # [2, E]


