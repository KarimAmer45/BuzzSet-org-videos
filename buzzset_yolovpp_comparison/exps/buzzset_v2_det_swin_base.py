"""Stage-1 single-frame detector on the NEW 4-class BuzzSet data — SWIN-BASE.

There is no ready Swin-Base *detector* base in the YOLOV checkout (`swin_base/`
only ships `swin_tiny_base.py`), so we build one: inherit all the BuzzSet data /
class-balanced-sampling logic from the Swin-Tiny detector exp and override only
`get_model()` to construct a Swin-Base backbone. The Swin-Base geometry mirrors
the (proven) Swin-Base branch of YOLOV's `v++_SwinBaseX_decoupleReg.get_model()`,
paired with the plain YOLOX detector head from `swin_tiny_base.py`.

This is the stronger-backbone half of Lennart's backbone experiment (Swin-Tiny vs
Swin-Base, Stage-1 and Stage-2). NOTE: it needs **Swin-Base** pretrained weights
for `-c` (the Swin-Tiny `.pth` will not load into a Swin-Base backbone).
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

# reuse the Swin-Tiny detector exp (data, sampling, eval) as the parent
_tiny_path = os.path.join(os.path.dirname(__file__), "buzzset_v2_det_swin_tiny.py")
_spec = importlib.util.spec_from_file_location("buzzset_v2_det_swin_tiny_forbase", _tiny_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TinyDetExp = _mod.Exp


class Exp(TinyDetExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_v2_det_swin_base"
        # Swin-Base geometry (matches YOLOV v++_SwinBaseX_decoupleReg)
        self.depth = 1.33
        self.width = 1.25
        self.pretrain_img_size = 384
        self.window_size = 12

    def get_model(self):
        import torch.nn as nn
        from yolox.models import YOLOX, YOLOPAFPN_Swin, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        out_channels = [256, 512, 1024]
        backbone = YOLOPAFPN_Swin(
            in_channels=in_channels,
            out_channels=out_channels,
            act=self.act,
            in_features=(1, 2, 3),
            swin_depth=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            base_dim=int(in_channels[0] / 2),   # 128 for Swin-Base
            pretrain_img_size=self.pretrain_img_size,
            window_size=self.window_size,
            width=self.width,
            depth=self.depth,
        )
        head = YOLOXHead(self.num_classes, self.width, in_channels=out_channels, act=self.act)
        self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
