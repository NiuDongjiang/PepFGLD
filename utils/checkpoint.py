"""Load the full-model checkpoints used by this project across Torch versions."""

import inspect

import torch


def load_model_checkpoint(path, map_location='cpu'):
    options = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)
