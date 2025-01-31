import warnings
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


def load_checkpoint(fpath: Path) -> dict:
    if not fpath.exists():
        raise FileNotFoundError(f'File is not found at "{fpath.resolve()}"')
    try:
        checkpoint = torch.load(
            fpath, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        raise RuntimeError(f'Unable to load checkpoint from "{fpath.resolve()}"')
    return checkpoint


def load_pretrained_weights(model: nn.Module, weight_path: Path):
    checkpoint = load_checkpoint(weight_path)
    state_dict: dict[str] = checkpoint.get("state_dict", checkpoint)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers: list[str] = []
    discard_layers: list[str] = []

    for k, v in state_dict.items():
        k = k.removeprefix("module.")
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discard_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if not matched_layers:
        warnings.warn(
            f'The pretrained weights "{weight_path.resolve()}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)"
        )
    else:
        print(f'Successfully loaded pretrained weights from "{weight_path.resolve()}"')
        if discard_layers:
            print(
                "** The following layers are discarded "
                f"due to unmatched keys or layer size: {discard_layers}"
            )
