"""Time-Dependent Energy-Guided diffusion (paper Section 4.2).

The schedule is indexed by reverse-sampling progress, resolving the conflict
between the paper's t/T convention and its stated weak-to-strong guidance.
"""

import torch

from .dist import dist_energy_


class TDEG:
    """Stateless physical guidance for a geometric latent diffusion trajectory."""

    @staticmethod
    def default_schedule(progress, lambda_min=0.1, lambda_max=0.7, tau0=0.7):
        if not (0 <= lambda_min <= lambda_max and 0 <= tau0 < 1):
            raise ValueError('Invalid TDEG schedule bounds or transition point')
        progress = min(1.0, max(0.0, float(progress)))
        if progress < tau0:
            return lambda_min
        return lambda_min + (lambda_max - lambda_min) * (progress - tau0) / (1 - tau0)

    @staticmethod
    def sampling_progress(t, num_steps):
        if num_steps is None or num_steps < 1:
            raise ValueError('TDEG requires a positive number of diffusion steps')
        t_idx = int(t.reshape(-1)[0].item()) if torch.is_tensor(t) else int(t)
        return 1.0 if num_steps == 1 else (num_steps - t_idx) / float(num_steps - 1)

    @staticmethod
    def energy(H, X, mask_generate, batch_ids, t, num_steps,
               consec_dist_mean, consec_dist_std, tolerance=3,
               model_external_energy=None, energy_lambda=1.0,
               energy_schedule=None, **kwargs):
        assert consec_dist_mean is not None and consec_dist_std is not None, \
            "Please run set_consec_dist(mean, std) before sampling."
        progress = TDEG.sampling_progress(t, num_steps)

        base_energy = 0.0
        external_energy = kwargs.get("external_energy", None)
        if external_energy is None:
            external_energy = model_external_energy
        if callable(external_energy):
            base_energy = external_energy(t, H, X, mask_generate, batch_ids, **kwargs)

        geom_energy = dist_energy_(
            X, mask_generate, batch_ids, consec_dist_mean, consec_dist_std,
            tolerance=tolerance,
            **{k: v for k, v in kwargs.items()
               if k not in ("energy_lambda", "external_energy")})
        if callable(energy_lambda):
            lam = float(energy_lambda(progress))
        else:
            lam = float(energy_lambda) * TDEG.default_schedule(
                progress, **(energy_schedule or {}))
        return lam * (geom_energy + base_energy)
