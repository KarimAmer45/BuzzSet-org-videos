import importlib.util
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
YOLOV_ROOT = os.environ.get("YOLOV_ROOT", os.path.join(WORKSPACE_ROOT, "YOLOV-master"))

for path in (PROJECT_ROOT, YOLOV_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Reuse the full stage-1 detector exp (dataset wiring, optimizer, class-balanced
# sampling) and override only the heavy knobs for a quick sanity-check run.
_base_path = os.path.join(HERE, "buzzset_stage1_swin_tiny_det.py")
_spec = importlib.util.spec_from_file_location("buzzset_stage1_swin_tiny_det", _base_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Stage1Exp = _mod.Exp


class Exp(Stage1Exp):
    """Fast sanity-check variant of the stage-1 detector: 576 px, no mosaic/mixup,
    short schedule -- runs in hours, not weeks. Class-balanced oversampling stays
    ON so you can see whether it lifts the rare classes before spending the big
    768 GPU time. Set self.class_balance = False for the no-rebalance A/B baseline.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_stage1_swin_tiny_det_fast"

        # Fast, Windows-friendly settings (the config that was validated to run
        # quickly): half the pixels of 768, no mosaic/mixup, short schedule.
        self.input_size = (576, 576)
        self.test_size = (576, 576)
        self.multiscale_range = 0
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.max_epoch = 10
        self.warmup_epochs = 1
        self.no_aug_epochs = 2

        # Rebalancing on (inherits alpha=0.5). Flip to False to A/B the baseline.
        self.class_balance = True
