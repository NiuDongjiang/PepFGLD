#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy

import torch
from utils.checkpoint import load_model_checkpoint
import torch.nn as nn
from torch_scatter import scatter_sum

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import length_to_batch_id
from data.format import VOCAB

from .diffusion.dpm_full import GeometricLatentDiffusion
from .energies.dist import dist_energy
from ..autoencoder.model import FlexVAE
from .energies.tdeg import TDEG


@R.register('PepFGLD')
@R.register('LDMPepDesign')
class PepFGLD(nn.Module):
    """Receptor-conditioned geometric latent diffusion for peptide design."""

    def __init__(
            self,
            autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=0,
            n_rbf=0,
            cutoff=1.0,
            max_gen_position=30,
            mode='codesign',
            diffusion_opt={}):
        super().__init__()
        self.autoencoder_no_randomness = autoencoder_no_randomness
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)
        self.max_gen_position=max_gen_position
        self.autoencoder: FlexVAE = load_model_checkpoint(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False

        latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size

        self.abs_position_encoding = nn.Embedding(max_gen_position, latent_size)
        self.diffusion = GeometricLatentDiffusion(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            **diffusion_opt
        )
        if self.train_sequence:
            self.hidden2latent = nn.Linear(hidden_size, self.autoencoder.latent_size)
            self.h_loss_weight = self.autoencoder.latent_n_channel * 3 / self.autoencoder.latent_size  # make loss_X and loss_H about the same size
        if self.train_structure:
            # for better constrained sampling
            self.consec_dist_mean, self.consec_dist_std = None, None

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, L=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask,
                                                 no_randomness=self.autoencoder_no_randomness)

        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        # position_embedding = self.autoencoder.aa_feature.aa_embedding.res_pos_embedding(position_ids)
        position_embedding = self.abs_position_encoding(
            torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))
        #print("max_gen_position dis:",self.max_gen_position)
        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H

        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            # X[mask] = Z.unsqueeze(1).expand_as(X[mask])
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        loss_dict = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )

        # loss
        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        return loss, loss_dict

    def set_consec_dist(self, mean: float, std: float):
        self.consec_dist_mean = mean
        self.consec_dist_std = std

    def latent_geometry_guidance(self, H, X, mask_generate, batch_ids, t, tolerance=3, **kwargs):
        assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
        if hasattr(self, 'external_energy'):
            base_energy = self.external_energy(t, H, X, mask_generate, batch_ids, **kwargs)
        else:
            base_energy = 0
        return dist_energy(
            X, mask_generate, batch_ids,
            self.consec_dist_mean, self.consec_dist_std,
            tolerance=tolerance, **kwargs
        ) + base_energy
    def physics_consistent_guidance(self, H, X, mask_generate, batch_ids, t, tolerance=3, **kwargs):
        return TDEG.energy(
            H, X, mask_generate, batch_ids, t,
            num_steps=getattr(self.diffusion, 'num_steps', None),
            consec_dist_mean=self.consec_dist_mean,
            consec_dist_std=self.consec_dist_std,
            tolerance=tolerance,
            model_external_energy=getattr(self, 'external_energy', None),
            **kwargs)

    @torch.no_grad()
    def sample(
            self,
            X, S, mask, position_ids, lengths, atom_mask, L=None,
            sample_opt=None,
            return_tensor=False,
            optimize_sidechain=True,
            idealize=False,
            guide_mask=None  # to fix some part of the ligand
    ):

        if sample_opt is None:
            sample_opt = {'pbar': False, 'energy_func': 'tdeg', 'confidence': True}
        else:
            sample_opt = deepcopy(sample_opt)
        sample_opt.setdefault('energy_func', 'tdeg')
        use_confidence = sample_opt.pop('confidence', False)
        self.autoencoder.eval()

        if guide_mask is not None:
            with torch.no_grad():
                ref_mask = (~mask) | guide_mask
                ref_batch_ids = length_to_batch_id(S, lengths)[ref_mask]
                ref_lengths = scatter_sum(torch.ones_like(ref_batch_ids), ref_batch_ids, dim=0)
                guide_H, guide_Z, _, _ = self.autoencoder.encode(
                    X[ref_mask], S[ref_mask], mask[ref_mask], position_ids[ref_mask],
                    ref_lengths, atom_mask[ref_mask], no_randomness=self.autoencoder_no_randomness
                )

        if self.train_sequence:
            S = S.clone()
            if guide_mask is None:
                S[mask] = self.latent_idx
            else:
                S[mask & (~guide_mask)] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(
            torch.where(mask, position_ids + 1, torch.zeros_like(position_ids))
        )

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0).clone()
            if guide_mask is None:
                H_0[mask] = 0
            else:
                H_0[guide_mask] = guide_H
                H_0[mask & (~guide_mask)] = 0

        if self.train_structure:
            X = X.clone()
            if guide_mask is None:
                X[mask] = 0
            else:
                X[guide_mask] = self.autoencoder._fill_latent_channels(guide_Z)
                X[mask & (~guide_mask)] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure

        if 'energy_func' in sample_opt:
            ef = sample_opt['energy_func']
            if ef is None:
                pass
            elif ef in ('distance', 'legacy_distance'):
                sample_opt['energy_func'] = self.latent_geometry_guidance
                sample_opt.pop('energy_schedule', None)
            elif ef in ('default', 'tdeg', 'enhanced', 'physics', 'pc'):
                guidance_options = dict(sample_opt)
                guidance_options['energy_lambda'] = sample_opt.get('energy_lambda', 1.0)
                guidance_options['energy_schedule'] = sample_opt.pop('energy_schedule', None)
                # TDEG scales the energy before differentiation. Do not multiply
                # its gradient by the schedule a second time in the transition.
                sample_opt['energy_lambda'] = 1.0

                def energy_callback(H, X, mask_generate, batch_ids, t, **kw):
                    kwargs = dict(guidance_options)
                    kwargs.update(kw)
                    return self.physics_consistent_guidance(
                        H=H, X=X,
                        mask_generate=mask_generate,
                        batch_ids=batch_ids,
                        t=t,
                        **kwargs
                    )

                sample_opt['energy_func'] = energy_callback
            elif not callable(ef):
                raise ValueError(f'Unknown energy function: {ef}')

        traj = self.diffusion.sample(
            H_0, X, position_embedding, mask, lengths,
            atom_embeddings, atom_mask, L,
            guide_mask=guide_mask, **sample_opt
        )
        X_0, H_0 = traj[0]

        if use_confidence:
            batch_confidence = self.diffusion.cal_confidence(
                H_0, X_0, position_embedding, mask, lengths, atom_embeddings,
                atom_mask, L, self.train_structure, self.train_sequence
            )

        X_0, H_0 = X_0[mask][:, :self.autoencoder.latent_n_channel], H_0[mask]

        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_laten_H=H_0, given_latent_X=X_0,
            return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain, idealize=idealize
        )

        if use_confidence:
            return batch_X, batch_S, batch_confidence.tolist()
        else:
            return batch_X, batch_S, batch_ppls

    @torch.no_grad()
    def cal_confidence(self, X, S, mask, position_ids, lengths, atom_mask, L=None):

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask,
                                                 no_randomness=self.autoencoder_no_randomness)

        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        # position_embedding = self.autoencoder.aa_feature.aa_embedding.res_pos_embedding(position_ids)
        position_embedding = self.abs_position_encoding(
            torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H

        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        return self.diffusion.cal_confidence(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )

# Compatibility alias for existing full-model checkpoints.
LDMPepDesign = PepFGLD
