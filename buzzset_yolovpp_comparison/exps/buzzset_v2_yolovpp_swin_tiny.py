"""Stage-2 YOLOV++ (Swin-Tiny) on the NEW 4-class BuzzSet Challenge video data.

Backbone + Stage-1 detection head are FROZEN by the YOLOV++ base get_model();
only the temporal aggregation head trains ("train only the aggregation part").
Number of context frames = global reference frames, set via the CONTEXT_FRAMES
env var (see scripts/run_context_frame_ablation.ps1). CLI batch == lframe+gframe.

Load the trained Stage-1 detector as the checkpoint (-c) so the frozen backbone
carries the single-frame weights.
"""
import importlib.util
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
YOLOV_ROOT = os.environ.get("YOLOV_ROOT", os.path.join(WORKSPACE_ROOT, "YOLOV-master"))
for path in (PROJECT_ROOT, YOLOV_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

base_exp_path = os.path.join(YOLOV_ROOT, "exps", "yolov++", "v++_SwinTiny_decoupleReg.py")
spec = importlib.util.spec_from_file_location("v_plus_swin_tiny_dr", base_exp_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseExp = module.Exp

from buzzset_yolov import torch_compat  # noqa: F401  (PyTorch >=2.6 torch.load shim)
from buzzset_yolov.vid_v2 import BuzzSetV2VidMixin


class Exp(BuzzSetV2VidMixin, BaseExp):
    pass
