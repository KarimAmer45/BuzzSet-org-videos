"""Shared BuzzSet (new 4-class) Stage-2 YOLOV++ config + data loaders.

Mixed into the swin-tiny / swin-base Stage-2 exps so both share identical data
handling and differ only in the YOLOV++ backbone base they inherit.

Re: Lennart's Stage-2 asks:
  * "Frozen Stage-1 backbone, train only the aggregation part": already enforced
    by the YOLOV++ base `get_model()` (backbone + detection-head params get
    requires_grad=False; only the temporal aggregation head trains).
  * "Impact of different numbers of context frames": set CONTEXT_FRAMES env ->
    self.gframe (global reference frames). The CLI batch size must equal
    lframe + gframe.
"""
import os

BUZZSET_V2_CLASSES = ("bee", "bumblebee", "hoverfly", "moth")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

# number of temporal context frames (global reference frames) for the ablation
CONTEXT_FRAMES = int(os.environ.get("CONTEXT_FRAMES", "16"))


class BuzzSetV2VidMixin:
    def __init__(self):
        super().__init__()
        self.num_classes = len(BUZZSET_V2_CLASSES)
        self.class_names = BUZZSET_V2_CLASSES
        _bk = getattr(self, "backbone_name", "swin").lower()
        self.exp_name = f"buzzset_v2_yolovpp_{_bk}_ctx{CONTEXT_FRAMES}"

        # --- new 4-class BuzzSet Challenge (video) data at <root>/data ---
        self.data_root = os.environ.get(
            "BUZZSET_V2_ROOT", os.path.join(WORKSPACE_ROOT, "data"))
        self.ann_dir = os.environ.get(
            "BUZZSET_V2_VID_ANN_DIR", os.path.join(WORKSPACE_ROOT, "data", "annotations"))
        self.train_name, self.val_name = "train", "valid"
        self.eval_name, self.test_name = "valid", "test_devphase"
        self.train_ann, self.val_ann, self.test_ann = "train.json", "valid.json", "test_devphase.json"

        # Fixed resolution across the context-frame sweep (matches Stage-1 new-data;
        # raise to 768 only with GPU headroom -- memory scales with gframe).
        self.input_size = (576, 576)
        self.test_size = (576, 576)

        # --- context-frame knob (global reference frames) ---
        # CLI batch size MUST equal lframe + gframe.
        self.lframe = 0
        self.lframe_val = 0
        self.gframe = CONTEXT_FRAMES
        self.gframe_val = CONTEXT_FRAMES

        self.max_epoch = 30
        self.warmup_epochs = 1
        self.no_aug_epochs = 2
        self.eval_interval = 1
        self.print_interval = 20
        self.data_num_workers = 4
        self.output_dir = os.path.join(PROJECT_ROOT, "runs")

    # ---- data plumbing (mirrors buzzset_yolovpp_swin_tiny.py, new-data paths) ----
    def _ann_file(self, split_name):
        name = {self.train_name: self.train_ann, self.val_name: self.val_ann,
                self.test_name: self.test_ann}.get(split_name, f"{split_name}.json")
        return os.path.join(self.ann_dir, name)

    def _image_root(self, split_name):
        return os.path.join(self.data_root, split_name)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform
        from yolox.data.datasets import vid
        from buzzset_yolov.dataset import BuzzSetVideoDataset

        assert batch_size == self.lframe + self.gframe, (
            f"batch_size ({batch_size}) must equal lframe+gframe "
            f"({self.lframe}+{self.gframe}={self.lframe + self.gframe})")
        dataset = BuzzSetVideoDataset(
            image_root=self._image_root(self.train_name),
            ann_file=self._ann_file(self.train_name),
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=100, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            lframe=self.lframe, gframe=self.gframe, val=False, mode="random",
        )
        return vid.get_trans_loader(
            batch_size=batch_size, data_num_workers=self.data_num_workers, dataset=dataset)

    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=None, formal=False):
        from yolox.data.data_augment import Vid_Val_Transform
        from yolox.data.datasets import vid
        from buzzset_yolov.dataset import BuzzSetVideoDataset

        if tnum is None:
            tnum = self.tnum
        if data_num_workers is None:
            data_num_workers = self.data_num_workers
        assert batch_size == self.lframe_val + self.gframe_val
        dataset_val = BuzzSetVideoDataset(
            image_root=self._image_root(self.eval_name),
            ann_file=self._ann_file(self.eval_name),
            img_size=self.test_size, preproc=Vid_Val_Transform(),
            lframe=self.lframe_val, gframe=self.gframe_val, val=True,
            mode="random", tnum=tnum, formal=formal,
        )
        return vid.vid_val_loader(
            batch_size=batch_size, data_num_workers=data_num_workers, dataset=dataset_val)

    def get_evaluator(self, val_loader):
        from buzzset_yolov.evaluator import build_vid_evaluator
        return build_vid_evaluator(
            val_loader=val_loader, img_size=self.test_size, confthre=self.test_conf,
            nmsthre=self.nmsthre, class_names=self.class_names,
            lframe=self.lframe_val, gframe=self.gframe_val, first_only=False,
        )
