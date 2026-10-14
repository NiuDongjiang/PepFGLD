#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List
from pathlib import Path
from copy import deepcopy
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = frozenset({
    'mmap_dir', 'specify_index', 'specify_data', 'cluster',
    'save_dir', 'autoencoder_ckpt', 'load_ckpt',
})


def _map_config_paths(config, transform):
    if isinstance(config, dict):
        return {
            key: transform(value) if key in PATH_KEYS and isinstance(value, str) and value
            else _map_config_paths(value, transform)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_map_config_paths(value, transform) for value in config]
    return config


def resolve_config_paths(config):
    """Interpret configured relative paths from the project root on every entry point."""
    def resolve(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    return _map_config_paths(config, resolve)


def project_relative_config(config):
    """Keep saved training configurations portable as well as source configurations."""
    def relative(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            return value
        try:
            return './' + os.path.relpath(str(path), str(PROJECT_ROOT)).replace('\\', '/')
        except ValueError:
            # Windows cannot express a relative path between different drives.
            return value
    return _map_config_paths(config, relative)


def load_config(path, overrides=()):
    """Load YAML, apply CLI overrides, then resolve paths consistently."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    config = overwrite_values(deepcopy(config), overrides)
    return resolve_config_paths(config)


def format_args(args: List[str]):
    clean_args = []
    for arg in args:
        if not (arg.startswith('-') or arg.startswith('--')): # value
            clean_args.append(arg)
        else:
            arg = arg.lstrip('-').lstrip('-')
            clean_args.extend(arg.split('='))
    return clean_args


def get_parent_dict(config: dict, key: str):
    key_each_depth = key.split('.')
    for k in key_each_depth[:-1]:
        if k not in config:
            raise KeyError(f'Path key {key} not in the dict')
        config = config[k]
    return config, key_each_depth[-1] # last key


def overwrite_values(config, args):
    args = format_args(args)
    keys, values = args[0::2], args[1::2]
    for key, value in zip(keys, values):
        parent, last_key = get_parent_dict(config, key)
        ori_value = parent[last_key]
        parent[last_key] = type(ori_value)(value)
    return config
