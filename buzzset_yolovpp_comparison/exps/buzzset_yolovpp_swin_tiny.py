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
spec = importlib.util.spec_from_file_location("yolovpp_swin_tiny_decouple_reg", base_exp_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseExp = module.Exp

from buzzset_yolov import torch_compat  # noqa: F401  (PyTorch >=2.6 torch.load shim)
from buzzset_yolov import BUZZSET_CLASSES
from buzzset_yolov.dataset import BuzzSetVideoDataset
from buzzset_yolov.evaluator import build_vid_evaluator


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_yolovpp_swin_tiny"
        self.num_classes = len(BUZZSET_CLASSES)
        self.class_names = BUZZSET_CLASSES

        self.data_root = os.environ.get(
            "BUZZSET_ROOT",
            os.path.join(WORKSPACE_ROOT, "BuzzSet-org-videos", "BuzzSetV2_split"),
        )
        self.ann_dir = os.environ.get(
            "BUZZSET_YOLOV_ANN_DIR",
            os.path.join(PROJECT_ROOT, "generated", "annotations"),
        )
        self.train_name = "train"
        self.val_name = "valid"
        self.test_name = "test"
        self.eval_name = "valid"
        self.train_ann = "train_yolov_coco.json"
        self.val_ann = "valid_yolov_coco.json"
        self.test_ann = "test_yolov_coco.json"

        # Train/eval resolution. 768 recovers most of the small insects that 576
        # downscales below the detector's usable size; kept equal to stage 1.
        # Memory at 768 scales with gframe (batch size must equal lframe +
        # gframe); if it OOMs, drop gframe and gframe_val to 8 and pass -BatchSize 8.
        self.input_size = (768, 768)
        self.test_size = (768, 768)

        self.gframe = 16
        self.gframe_val = 16
        self.lframe = 0
        self.lframe_val = 0
        # 20 epochs plateaued with the old frozen ImageNet-VID detector; the
        # stage-1 BuzzSet detector has more headroom, so allow more epochs.
        self.max_epoch = 30
        self.warmup_epochs = 1
        self.no_aug_epochs = 2
        self.eval_interval = 1
        self.print_interval = 20
        self.data_num_workers = 4
        self.output_dir = os.path.join(PROJECT_ROOT, "runs")

    def _ann_file(self, split_name):
        if split_name == self.train_name:
            ann_name = self.train_ann
        elif split_name == self.val_name:
            ann_name = self.val_ann
        elif split_name == self.test_name:
            ann_name = self.test_ann
        else:
            ann_name = f"{split_name}_yolov_coco.json"
        return os.path.join(self.ann_dir, ann_name)

    def _image_root(self, split_name):
        return os.path.join(self.data_root, split_name)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform
        from yolox.data.datasets import vid

        assert batch_size == self.lframe + self.gframe
        dataset = BuzzSetVideoDataset(
            image_root=self._image_root(self.train_name),
            ann_file=self._ann_file(self.train_name),
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=100, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            lframe=self.lframe,
            gframe=self.gframe,
            val=False,
            mode="random",
        )
        return vid.get_trans_loader(
            batch_size=batch_size,
            data_num_workers=self.data_num_workers,
            dataset=dataset,
        )

    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=None, formal=False):
        from yolox.data.data_augment import Vid_Val_Transform
        from yolox.data.datasets import vid

        if tnum is None:
            tnum = self.tnum
        if data_num_workers is None:
            data_num_workers = self.data_num_workers

        assert batch_size == self.lframe_val + self.gframe_val
        dataset_val = BuzzSetVideoDataset(
            image_root=self._image_root(self.eval_name),
            ann_file=self._ann_file(self.eval_name),
            img_size=self.test_size,
            preproc=Vid_Val_Transform(),
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            val=True,
            mode="random",
            tnum=tnum,
            formal=formal,
        )
        return vid.vid_val_loader(
            batch_size=batch_size,
            data_num_workers=data_num_workers,
            dataset=dataset_val,
        )

    def get_evaluator(self, val_loader):
        return build_vid_evaluator(
            val_loader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            class_names=self.class_names,
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            first_only=False,
        )

