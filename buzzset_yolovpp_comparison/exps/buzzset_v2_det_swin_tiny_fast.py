import importlib.util
import os

# Reuse the full BuzzSet-v2 detector exp and only strip the slow, spiky
# augmentation so the run can finish under a deadline. This produces a REAL
# single-frame (Stage-1) number on the new 4-class data.
_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "buzzset_v2_det_swin_tiny", os.path.join(_here, "buzzset_v2_det_swin_tiny.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
V2Exp = _mod.Exp


class Exp(V2Exp):
    """Fast, stable variant: no mosaic, fixed 512 px, 8 epochs.

    Mosaic + multi-scale (up to 640) is what makes iter_time spike to 10-36 s
    on the local GPU. Turning both off gives a steady ~2-4 s/iter, so 8 epochs
    finish in roughly 2.5-3 h instead of 12 h. AP will be a touch lower than a
    full 576 px mosaic run, but it is a real measured number.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_v2_det_swin_tiny_fast"

        # kill the augmentation that causes the iter_time spikes
        self.mosaic_prob = 0.0
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.multiscale_range = 0          # fixed size -> stable iter_time

        # smaller, fixed input for speed
        self.input_size = (512, 512)
        self.test_size = (512, 512)

        self.max_epoch = 8
        self.warmup_epochs = 1
        self.no_aug_epochs = 0             # no mosaic to turn off anyway
        self.eval_interval = 1
        self.data_num_workers = 4

        # keep the gentle class rebalancing
        self.class_balance = True
        self.class_balance_alpha = 0.3
