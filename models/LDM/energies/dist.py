#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import math
from data.format import VOCAB
from utils.nn_utils import graph_to_batch


@torch.no_grad()
def continuous_bool(x, k=1000):
    return (x > 0).float()


def _consec_dist_loss(gen_X, gen_X_mask, lb, ub, eps=1e-6):
    consec_dist = torch.norm(gen_X[..., 1:, :] - gen_X[..., :-1, :], dim=-1) # [bs, max_L - 1]
    consec_lb_loss = lb - consec_dist  # [bs, max_L - 1]
    consec_ub_loss = consec_dist - ub  # [bs, max_L - 1]

    consec_lb_invalid = (consec_dist < lb) & gen_X_mask[..., 1:]
    consec_ub_invalid = (consec_dist > ub) & gen_X_mask[..., 1:]
    consec_loss = torch.where(consec_lb_invalid, consec_lb_loss, torch.zeros_like(consec_lb_loss))
    consec_loss = torch.where(consec_ub_invalid, consec_ub_loss, consec_loss)

    consec_loss = consec_loss.sum(-1) / (consec_lb_invalid + consec_ub_invalid + eps).sum(-1)
    consec_loss = torch.sum(consec_loss) # consistent loss scale across different batch size
    return consec_loss


def _inner_clash_loss(gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - gen_X[..., None, :, :], dim=-1) # [bs, max_L, max_L]
    dist_mask = gen_X_mask[..., :, None] & gen_X_mask[..., None, :] # [bs, max_L, max_L]
    pos = torch.cumsum(torch.ones_like(gen_X_mask, dtype=torch.long), dim=-1) # [bs, max_L]
    non_consec_mask = torch.abs(pos[..., :, None] - pos[..., None, :]) > 1  # [bs, max_L, max_L]

    clash_loss = mean - dist
    clash_loss_mask = (clash_loss > 0) & dist_mask & non_consec_mask # [bs, max_L, max_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - ctx_X[..., None, :, :], dim=-1)  # [bs, max_gen_L, max_ctx_L]
    dist_mask = gen_X_mask[..., :, None] & ctx_X_mask[..., None, :] # [bs, max_gen_L, max_ctx_L]
    clash_loss = mean - dist  # [bs, max_gen_L, max_ctx_L]
    clash_loss_mask = (clash_loss > 0) & dist_mask  # [bs, max_gen_L, max_ctx_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def dist_energy(X, mask_generate, batch_ids, mean, std, tolerance=3, **kwargs):
    mean, std = round(mean, 4), round(std, 4)
    lb, ub = mean - tolerance * std, mean + tolerance * std

    X = X.clone() # [N, 3]

    ctx_X, ctx_batch_ids = X[~mask_generate], batch_ids[~mask_generate]
    gen_X, gen_batch_ids = X[mask_generate], batch_ids[mask_generate]
    ctx_X = ctx_X[:, VOCAB.ca_channel_idx] # CA (alpha carbon)
    gen_X = gen_X[:, 0] # latent one

    # to batch representation
    ctx_X, ctx_X_mask = graph_to_batch(ctx_X, ctx_batch_ids, mask_is_pad=False) # [bs, max_ctx_L, 3]
    gen_X, gen_X_mask = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False) # [bs, max_gen_L, 3]

    # consecutive
    consec_loss = _consec_dist_loss(gen_X, gen_X_mask, lb, ub)

    # inner clash
    inner_clash_loss = _inner_clash_loss(gen_X, gen_X_mask, mean)

    # outer clash
    outer_clash_loss = _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean)

    return consec_loss + inner_clash_loss + outer_clash_loss

import torch
import torch.nn.functional as F

def dist_energy_(X, mask_generate, batch_ids, mean, std, tolerance=3, **kwargs):
    mean, std = round(mean, 4), round(std, 4)
    lb, ub = mean - tolerance * std, mean + tolerance * std

    X = X.clone()  # [N, 3]
    ctx_X, ctx_batch_ids = X[~mask_generate], batch_ids[~mask_generate]
    gen_X, gen_batch_ids = X[mask_generate], batch_ids[mask_generate]
    ctx_X = ctx_X[:, VOCAB.ca_channel_idx]  # CA (alpha carbon)
    gen_X = gen_X[:, 0]  # latent one

    # to batch representation
    ctx_X, ctx_X_mask = graph_to_batch(ctx_X, ctx_batch_ids, mask_is_pad=False)
    gen_X, gen_X_mask = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False)

    consec_loss = _consec_dist_loss(gen_X, gen_X_mask, lb, ub)
    inner_clash_loss = _inner_clash_loss(gen_X, gen_X_mask, mean)
    outer_clash_loss = _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean)

    angle_loss = _bond_angle_loss(gen_X, gen_X_mask)
    torsion_loss = _torsion_angle_loss(gen_X, gen_X_mask)
    radius_penalty = _radius_distribution_loss(gen_X, gen_X_mask)

    total_energy = (
        1.0 * consec_loss +
        1.0 * inner_clash_loss +
        1.0 * outer_clash_loss +
        angle_loss +
        torsion_loss +
        radius_penalty
    )

    return total_energy


def _bond_angle_loss(X, mask, eps=1e-8):

    if X.size(1) < 3:
        return torch.zeros((), device=X.device)

    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tri_mask = mask[:, 1:-1] & mask[:, :-2] & mask[:, 2:]

    v1 = X[:, 1:-1, :] - X[:, :-2, :]
    v2 = X[:, 2:, :] - X[:, 1:-1, :]

    cross = torch.norm(torch.cross(v1, v2, dim=-1), dim=-1)
    dot = (v1 * v2).sum(-1)
    theta = torch.atan2(cross, dot + eps)

    ideal = 110.0 / 180.0 * math.pi

    diff = (theta - ideal) ** 2
    diff = diff * tri_mask.float()
    denom = tri_mask.sum(dim=-1) + eps
    loss = (diff.sum(dim=-1) / denom).sum()

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

def _torsion_angle_loss(X, mask, eps=1e-8):
    if X.size(1) < 4:
        return torch.zeros((), device=X.device)

    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    b1 = X[:, 1:-2, :] - X[:, :-3, :]
    b2 = X[:, 2:-1, :] - X[:, 1:-2, :]
    b3 = X[:, 3:, :] - X[:, 2:-1, :]

    quad_mask = mask[:, :-3] & mask[:, 1:-2] & mask[:, 2:-1] & mask[:, 3:]
    b2n = F.normalize(b2, dim=-1, eps=eps)
    n1 = F.normalize(torch.cross(b1, b2, dim=-1), dim=-1, eps=eps)
    n2 = F.normalize(torch.cross(b2, b3, dim=-1), dim=-1, eps=eps)

    x = (n1 * n2).sum(-1)
    m1 = torch.cross(n1, b2n, dim=-1)
    y = (m1 * n2).sum(-1)
    phi = torch.atan2(y, x + eps)

    phi_deg = torch.abs(phi * 180.0 / math.pi)
    violations = F.relu(phi_deg - 120.0) * quad_mask.float()
    loss = (violations.sum(dim=-1) / (quad_mask.sum(dim=-1) + eps)).sum()

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

def _radius_distribution_loss(X, mask, eps=1e-8):
    if X.numel() == 0 or mask.sum() == 0:
        return torch.zeros((), device=X.device)

    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mask_f = mask.float().unsqueeze(-1)
    center = (X * mask_f).sum(1, keepdim=True) / (mask_f.sum(1, keepdim=True) + eps)
    dist = torch.norm(X - center, dim=-1)

    weights = mask.float()
    counts = weights.sum(dim=-1)
    mean_dist = (dist * weights).sum(dim=-1) / counts.clamp_min(1)
    variance = (((dist - mean_dist[:, None]) ** 2) * weights).sum(dim=-1) / counts.clamp_min(1)
    std_val = torch.sqrt(variance + eps)
    target = 8.0
    loss = (((std_val - target) ** 2) * (counts >= 2).float()).sum()
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)
