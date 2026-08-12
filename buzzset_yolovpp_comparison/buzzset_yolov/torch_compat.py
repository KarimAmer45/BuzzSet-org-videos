"""PyTorch >= 2.6 compatibility shim for loading YOLOV checkpoints.

PyTorch 2.6 changed ``torch.load`` to default ``weights_only=True``, which
refuses checkpoints that pickle numpy scalars/dtypes. The YOLOV/YOLOX trainer
saves a numpy ``best_ap`` in each checkpoint, so stage 2 fails the moment it
tries to load the stage-1 detector checkpoint (and any resume would too).

Importing this module restores the pre-2.6 full-unpickle behavior for these
local, trusted checkpoints, without editing the upstream YOLOV code. Calls that
pass ``weights_only=`` explicitly are left untouched.
"""
import torch

if not getattr(torch.load, "_buzzset_weights_only_patched", False):
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._buzzset_weights_only_patched = True
    torch.load = _load
