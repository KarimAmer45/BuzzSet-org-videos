"""Stage-2 YOLOV++ (Swin-Base) on the NEW 4-class BuzzSet Challenge video data.

Same data + frozen-backbone behaviour as the Swin-Tiny variant, but with the
stronger Swin-Base backbone. Used for the backbone experiment: measure how much
temporal aggregation helps for a weak (Swin-Tiny) vs strong (Swin-Base) backbone.

Needs the Swin-Base pretrained weights; discuss the backbone weights/setup in the
YOLOV meeting (per Lennart). CLI batch == lframe+gframe.
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

base_exp_path = os.path.join(YOLOV_ROOT, "exps", "yolov++", "v++_SwinBaseX_decoupleReg.py")
spec = importlib.util.spec_from_file_location("v_plus_swin_base_dr", base_exp_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseExp = module.Exp

from buzzset_yolov import torch_compat  # noqa: F401
from buzzset_yolov.vid_v2 import BuzzSetV2VidMixin


class Exp(BuzzSetV2VidMixin, BaseExp):
    pass
