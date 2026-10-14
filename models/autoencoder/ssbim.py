"""Bidirectional Sequence–Structure Interaction Module (paper Section 4.1)."""

import torch
import torch.nn as nn


class SSBIM(nn.Module):
    """Trainable bidirectional coupling over peptide residues within each complex.

    Feed the updated sequence into the return attention so both directions reach
    the geometric state used by the latent posterior, without adding a fusion net.
    """

    def __init__(self, hidden_size, num_heads=8, alpha=0.5, beta=0.5):
        super().__init__()
        self.geo_to_seq_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.seq_to_geo_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def _upgrade_gates(self):
        reference = self.seq_to_geo_attn.in_proj_weight
        for name in ('alpha', 'beta'):
            if name not in self._parameters:
                value = self.__dict__.pop(name, 0.5)
                self.register_parameter(name, nn.Parameter(
                    torch.as_tensor(value, device=reference.device,
                                    dtype=reference.dtype).detach().clone(),
                    requires_grad=reference.requires_grad))

    def __setstate__(self, state):
        super().__setstate__(state)
        self._upgrade_gates()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old checkpoints contain attention weights but no trainable gate keys.
        if any(key.startswith(prefix + 'geo_to_seq_attn.') for key in state_dict):
            for name in ('alpha', 'beta'):
                state_dict.setdefault(prefix + name, getattr(self, name).detach().clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def from_legacy(cls, geo_to_seq_attn, seq_to_geo_attn, alpha, beta):
        """Reuse checkpoint modules without initializing or changing any weights."""
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.geo_to_seq_attn = geo_to_seq_attn
        module.seq_to_geo_attn = seq_to_geo_attn
        module.alpha = alpha
        module.beta = beta
        module._upgrade_gates()
        module.training = geo_to_seq_attn.training
        return module

    def forward(self, H_seq, H_geo, mask_generate=None, batch_ids=None):
        if mask_generate is None:
            mask_generate = torch.ones(H_seq.shape[0], dtype=torch.bool,
                                       device=H_seq.device)
        indices = mask_generate.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return H_seq, H_geo
        if batch_ids is None:
            batch_ids = torch.zeros(H_seq.shape[0], dtype=torch.long,
                                    device=H_seq.device)
        peptide_batch_ids = batch_ids[indices]
        attention_mask = peptide_batch_ids[:, None] != peptide_batch_ids[None, :]
        H_seq_t = H_seq[indices].unsqueeze(1)
        H_geo_t = H_geo[indices].unsqueeze(1)
        H_seq_new, _ = self.geo_to_seq_attn(
            query=H_seq_t, key=H_geo_t, value=H_geo_t,
            attn_mask=attention_mask, need_weights=False)
        H_seq_updated = H_seq_t + self.alpha * H_seq_new
        H_geo_new, _ = self.seq_to_geo_attn(
            query=H_geo_t, key=H_seq_updated, value=H_seq_updated,
            attn_mask=attention_mask, need_weights=False)
        H_geo_updated = H_geo_t + self.beta * H_geo_new
        sequence, geometry = H_seq.clone(), H_geo.clone()
        sequence[indices] = H_seq_updated.squeeze(1)
        geometry[indices] = H_geo_updated.squeeze(1)
        return sequence, geometry
