
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import constants
from .structure.geometry import (
    dihedrals,
    extend_atoms,
)


class SideChainBuilder(nn.Module):

    def __init__(self, distance_eps=1e-6):
        super(SideChainBuilder, self).__init__()
        self.num_atoms = 10
        self.num_chi = 4
        self.num_aa = len(constants.AA20)
        self.distance_eps = distance_eps

        self._init_maps()

    def _init_maps(self):
        """Build geometry and topology maps in tensor form."""

        shape = (3, self.num_atoms, self.num_aa)
        self.register_buffer("_Z", torch.zeros(shape, dtype=torch.float))
        self.register_buffer("_parents", torch.zeros(shape, dtype=torch.long))
        self.register_buffer(
            "_chi_ix", 10 * torch.ones((self.num_chi, self.num_aa), dtype=torch.long)
        )

        for i, aa in enumerate(constants.AA20_3):
            aa_dict = constants.AA_GEOMETRY[aa]
            atoms_parents = ["N", "CA", "C", "O"] + aa_dict["atoms"]
            for j, atom in enumerate(aa_dict["atoms"]):
                # Internal coordinates per atom
                self._Z[0, j, i] = aa_dict["z-lengths"][j]
                self._Z[1, j, i] = aa_dict["z-angles"][j]
                self._Z[2, j, i] = aa_dict["z-dihedrals"][j]

                # Parent indices per atom
                parents = [atoms_parents.index(p) for p in aa_dict["parents"][j]]
                self._parents[0, j, i] = parents[0]
                self._parents[1, j, i] = parents[1]
                self._parents[2, j, i] = parents[2]

            # Map which chi angles are flexible
            for j, parent_ix in enumerate(aa_dict["chi_indices"]):
                self._chi_ix[j, i] = parent_ix

        # Convert angles from degrees to radians
        self._Z[1:, :, :] = self._Z[1:, :, :] * np.pi / 180.0

        # Manually fix Arginine, for which CHARMM places NH1 in trans to CD
        self._Z[2, 5, constants.AA20.index("R")] = 0.0

    def forward(self, X, C, S, chi=None):
        num_batch, num_residues = list(S.shape)

        if X.shape[2] > 4:
            X = X[:, :, :4, :]

        # Expand sequence indexing tensors for gathering residue-specific info
        # (B,L) => (B,L,4)
        S_expand3 = S.unsqueeze(-1).expand(-1, -1, 4)
        # (B,L) => (B,AA,ATOM,L)
        S_expand4 = S.reshape([num_batch, 1, 1, num_residues]).expand(
            -1, 3, self.num_atoms, -1
        )

        def _gather(Z):
            Z_expand = Z.unsqueeze(0).expand([num_batch, -1, -1, -1])
            # (B,3,ATOM,AA) @ (B,3,ATOM,L) => (B,3,ATOM,L) => (B,L,3,ATOM)
            Z_i = torch.gather(Z_expand, -1, S_expand4).permute([0, 3, 1, 2])
            return Z_i

        # Build ideal geometry length, angle, and dihedral tensors 3x(B,L,10)
        B, A, D = _gather(self._Z).unbind(-2)

        if chi is not None:
            # Scatter chi angles (B,L,4) onto their corresponding dihedrals (B,L,10)
            # (4,AA) => (B,AA,4)
            chi_ix_expand = (
                self._chi_ix.unsqueeze(0).expand([num_batch, -1, -1]).transpose(-2, -1)
            )
            # (B,AA,4) @ (B,L,4) => (B,L,4)
            chi_ix_i = torch.gather(chi_ix_expand, -2, S_expand3)

            # Scatter extra chi angles into an extra pad dimension & re-slice
            # (B,L,10) <- (B,L,4),(B,L,4) => (B,L,10)
            D_pad = F.pad(D, (0, 1))
            D_pad = torch.scatter(D_pad, -1, chi_ix_i, chi)
            D = D_pad[:, :, : self.num_atoms]

        # Build indices of parent atoms (B,L,3,10)
        X_full = F.pad(X, (0, 0, 0, self.num_atoms))
        parents = _gather(self._parents)

        # Build atom i given current frame
        for i in range(self.num_atoms):
            # Gather parents (B,L,A,3) => (B,L,3,3)
            parents_expand = parents[:, :, :, i].unsqueeze(-1).expand(-1, -1, -1, 3)
            # (B,L,A,3) @ (B,L,3,3) => (B,L,3,3)
            X1, X2, X3 = torch.gather(X_full, -2, parents_expand).unbind(-2)

            # Extend atom i
            X4 = extend_atoms(
                X1,
                X2,
                X3,
                B[:, :, i],
                A[:, :, i],
                D[:, :, i],
                degrees=False,
                distance_eps=self.distance_eps,
            )

            # Scatter
            # X[:,:,i+4,:] = X4
            # scatter_ix = (i+4) * torch.ones(
            #     (num_batch,num_residues,1,3), dtype=torch.long
            # )
            # print(X_full.shape, X4.shape, scatter_ix.shape, i+4)
            # print(scatter_ix)
            # X_full.scatter_(-2, scatter_ix, X4.unsqueeze(-2))
            # X_full = torch.scatter(X_full, -2, scatter_ix, X4)
            # X_full = X_full + 0.1*X4.mean()

            # For some reason direct scatter causes autograd issues
            X4_expand = F.pad(X4.unsqueeze(-2), (0, 0, 4 + i, 9 - i))
            X_full = X_full + X4_expand

            # DEBUG: TEST
            if False:
                D_reconstruct = dihedrals(X1, X2, X3, X4)
                D_error = (
                    (torch.cos(D[:, :, i]) - torch.cos(D_reconstruct)) ** 2
                    + (torch.sin(D[:, :, i]) - torch.sin(D_reconstruct)) ** 2
                ).mean()
                print(D_error)

        mask_X = atom_mask(C, S).unsqueeze(-1)
        X_full = mask_X * X_full
        return X_full, mask_X


class ChiAngles(nn.Module):

    def __init__(self, distance_eps=1e-6):
        super(ChiAngles, self).__init__()
        self.num_atoms = 10
        self.num_chi = 4
        self.num_aa = len(constants.AA20)

        self.distance_eps = distance_eps

        self._init_maps()

    def _init_maps(self):

        self.register_buffer(
            "_chi_atom_sets",
            torch.zeros((self.num_aa, self.num_chi, 4), dtype=torch.long),
        )

        for i, aa in enumerate(constants.AA20_3):
            aa_dict = constants.AA_GEOMETRY[aa]
            atoms_names = ["N", "CA", "C", "O"] + aa_dict["atoms"]

            # Map which chi angles are flexible
            for j, parent_ix in enumerate(aa_dict["chi_indices"]):
                atom_quartet = aa_dict["parents"][parent_ix] + [
                    aa_dict["atoms"][parent_ix]
                ]
                for k, atom in enumerate(atom_quartet):
                    self._chi_atom_sets[i, j, k] = atoms_names.index(atom)

    def forward(self, X, C, S):
        num_batch, num_residues = list(S.shape)
        # (B,L) => (B,L,16)
        S_expand = S.unsqueeze(-1).expand([-1, -1, 16])

        # (AA,CHI,ATOM) => (AA,16) => (B,AA,16)
        chi_indices_per_aa = self._chi_atom_sets.reshape([1, self.num_aa, 16])
        chi_indices_per_aa = chi_indices_per_aa.expand([num_batch, -1, -1])

        # (B,AA,16) @ (B,L,16) => (B,L,16) => (B,L,16)
        chi_indices = torch.gather(chi_indices_per_aa, -2, S_expand)
        chi_indices = chi_indices.unsqueeze(-1).expand([-1, -1, -1, 3])

        # (B,L,14,3) @ (B,L,16,3)  => (B,L,16,3) => (B,L,4,4,3) => (B,L,4)
        X_chi = torch.gather(X, -2, chi_indices)
        X_1, X_2, X_3, X_4 = X_chi.reshape([num_batch, num_residues, 4, 4, 3]).unbind(
            -2
        )

        chi = dihedrals(X_1, X_2, X_3, X_4, distance_eps=self.distance_eps)

        mask_chi = chi_mask(C, S)
        chi = chi * mask_chi
        return chi, mask_chi


def _gather_atom_mask(C, S, atoms_per_aa, num_atoms):
    device = S.device
    atoms_per_aa = torch.tensor(atoms_per_aa, dtype=torch.long)
    atoms_per_aa = atoms_per_aa.to(device).unsqueeze(0).expand(S.shape[0], -1)

    # (B,A) @ (B,L)  => (B,L)
    atoms_per_residue = torch.gather(atoms_per_aa, -1, S)
    atoms_per_residue = (C > 0).float() * atoms_per_residue

    ix_expand = torch.arange(num_atoms, device=device).reshape([1, 1, -1])
    mask_atoms = ix_expand < atoms_per_residue.unsqueeze(-1)
    mask_atoms = mask_atoms.float()
    return mask_atoms


def atom_mask(C, S):

    return _gather_atom_mask(C, S, constants.AA20_NUM_ATOMS, 14)


def chi_mask(C, S):

    return _gather_atom_mask(C, S, constants.AA20_NUM_CHI, 4)
