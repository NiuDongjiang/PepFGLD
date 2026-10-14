#!/usr/bin/python
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sidechain.structure import geometry


def compose_translation(
    R_a: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor
) -> torch.Tensor:

    t_composed = t_a + (R_a @ t_b.unsqueeze(-1)).squeeze(-1)
    return t_composed


class FrameBuilder(nn.Module):

    def __init__(self, distance_eps: float = 1e-3):
        super().__init__()

        # Build idealized backbone fragment
        t = torch.tensor(
            [
                [1.459, 0.0, 0.0],  # N-C via Engh & Huber is 1.459
                [0.0, 0.0, 0.0],  # CA is origin
                [-0.547, 0.0, -1.424],  # C is placed 1.525 A @ 111 degrees from N
            ],
            dtype=torch.float32,
        ).reshape([1, 1, 3, 3])
        R = torch.eye(3).reshape([1, 1, 1, 3, 3])
        self.register_buffer("_t_atom", t)
        self.register_buffer("_R_atom", R)

        # Carbonyl geometry from CHARMM all36_prot ALA definition
        self._length_C_O = 1.2297
        self._angle_CA_C_O = 122.5200
        self._dihedral_Np_CA_C_O = 180
        self.distance_eps = distance_eps

    def _build_O(self, X_chain: torch.Tensor, C: torch.LongTensor):
        """Build backbone carbonyl oxygen."""
        # Build carboxyl groups
        X_N, X_CA, X_C = X_chain.unbind(-2)

        # TODO: fix this behavior for termini
        mask_next = (C > 0).float()[:, 1:].unsqueeze(-1)
        X_N_next = F.pad(mask_next * X_N[:, 1:,], (0, 0, 0, 1),)

        num_batch, num_residues = C.shape
        ones = torch.ones(list(C.shape), dtype=torch.float32, device=C.device)
        X_O = geometry.extend_atoms(
            X_N_next,
            X_CA,
            X_C,
            self._length_C_O * ones,
            self._angle_CA_C_O * ones,
            self._dihedral_Np_CA_C_O * ones,
            degrees=True,
        )
        mask = (C > 0).float().reshape(list(C.shape) + [1, 1])
        X = mask * torch.stack([X_N, X_CA, X_C, X_O], dim=-2)
        return X

    def forward(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        C: torch.LongTensor,
        q: Optional[torch.Tensor] = None,
    ):
        assert q is None or R is None

        if R is None:
            # (B,N,1,3,3) and (B,N,1,3)
            R = geometry.rotations_from_quaternions(
                q, normalize=True, eps=self.distance_eps
            )

        R = R.unsqueeze(-3)
        t_frame = t.unsqueeze(-2)
        X_chain = compose_translation(R, t_frame, self._t_atom)
        X = self._build_O(X_chain, C)
        return X

    def inverse(
        self, X: torch.Tensor, C: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        X_bb = X[:, :, :4, :]
        R, t = geometry.frames_from_backbone(X_bb, distance_eps=self.distance_eps)
        q = geometry.quaternions_from_rotations(R, eps=self.distance_eps)
        mask = (C > 0).float().unsqueeze(-1)
        R = mask.unsqueeze(-1) * R
        t = mask * t
        q = mask * q
        return R, t, q