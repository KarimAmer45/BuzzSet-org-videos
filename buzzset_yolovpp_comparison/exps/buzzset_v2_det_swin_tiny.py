import importlib.util
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
YOLOV_ROOT = os.environ.get("YOLOV_ROOT", os.path.join(WORKSPACE_ROOT, "YOLOV-master"))

for path in (PROJECT_ROOT, YOLOV_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


base_exp_path = os.path.join(YOLOV_ROOT, "exps", "swin_base", "swin_tiny_base.py")
spec = importlib.util.spec_from_file_location("swin_tiny_base_det_v2", base_exp_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseExp = module.Exp

from buzzset_yolov import torch_compat  # noqa: F401  (PyTorch >=2.6 torch.load shim)
from buzzset_yolov.coco_dataset import BuzzSetCocoDataset

# The BuzzSet Challenge (video) dataset has four classes -- other_insect is gone.
BUZZSET_V2_CLASSES = ("bee", "bumblebee", "hoverfly", "moth")


class Exp(BaseExp):
    """Single-frame YOLOX Swin-Tiny detector on the BuzzSet Challenge keyframes
    (the official 4-class video dataset). This is the single-frame reference for
    the milestone; the temporal YOLOV++ run reuses the same data later."""

    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_v2_det_swin_tiny"
        self.num_classes = len(BUZZSET_V2_CLASSES)
        self.class_names = BUZZSET_V2_CLASSES

        # New video dataset (keyframes + 5 previous frames). For the single-frame
        # detector we train on keyframes only, via the filtered COCO in generated_v2.
        self.data_root = os.environ.get(
            "BUZZSET_V2_ROOT",
            os.path.join(WORKSPACE_ROOT, "data"),
        )
        self.ann_dir = os.environ.get(
            "BUZZSET_V2_ANN_DIR",
            os.path.join(PROJECT_ROOT, "generated_v2", "annotations"),
        )
        self.train_name = "train"
        self.val_name = "valid"
        self.eval_name = "valid"

        # 576 px is stable on the local GPU and quick for an initial milestone
        # number; raise to 768 once a stable/cloud GPU is available.
        self.input_size = (576, 576)
        self.test_size = (576, 576)
        self.multiscale_range = 2
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0

        self.max_epoch = 20
        self.warmup_epochs = 1
        self.no_aug_epochs = 4
        self.eval_interval = 1
        self.print_interval = 20
        self.data_num_workers = 2
        self.test_conf = 0.001
        self.nmsthre = 0.65
        self.output_dir = os.path.join(PROJECT_ROOT, "runs")

        # Class-balanced oversampling; alpha=0.3 is gentle enough to avoid
        # overfitting the tiny classes (bumblebee ~232, moth ~202).
        self.class_balance = True
        self.class_balance_alpha = 0.3

    def _ann_file(self, split_name):
        key = {self.train_name: "train", self.val_name: "valid"}.get(split_name, split_name)
        return os.path.join(self.ann_dir, f"{key}_keyframes.json")

    def _image_root(self, split_name):
        return os.path.join(self.data_root, split_name)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        import torch.distributed as dist
        from yolox.data import (
            TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler,
            MosaicDetection, worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        from loguru import logger
        from buzzset_yolov.sampling import WeightedInfiniteSampler, class_balanced_image_weights

        with wait_for_the_master():
            dataset = BuzzSetCocoDataset(
                image_root=self._image_root(self.train_name),
                ann_file=self._ann_file(self.train_name),
                img_size=self.input_size,
                preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        image_weights = None
        if getattr(self, "class_balance", False):
            image_weights, info = class_balanced_image_weights(dataset, getattr(self, "class_balance_alpha", 0.3))
            logger.info("BuzzSet-v2 class-balanced sampling (alpha={}): counts={}, per-class weights={}".format(
                info["alpha"], info["counts"], info["class_weights"]))

        dataset = MosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate, mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale, shear=self.shear, enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob,
        )
        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        seed = self.seed if self.seed else 0
        if image_weights is not None:
            sampler = WeightedInfiniteSampler(image_weights, seed=seed)
        else:
            sampler = InfiniteSampler(len(self.dataset), seed=seed)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        kw = {"num_workers": self.data_num_workers, "pin_memory": False,
              "batch_sampler": batch_sampler, "worker_init_fn": worker_init_reset_seed}
        return DataLoader(self.dataset, **kw)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        import torch
        from yolox.data import ValTransform
        split = self.eval_name
        valdataset = BuzzSetCocoDataset(
            image_root=self._image_root(split), ann_file=self._ann_file(split),
            img_size=self.test_size, preproc=ValTransform(legacy=legacy),
        )
        if is_distributed:
            import torch.distributed as dist
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        kw = {"num_workers": self.data_num_workers, "pin_memory": False, "sampler": sampler, "batch_size": batch_size}
        return torch.utils.data.DataLoader(valdataset, **kw)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return COCOEvaluator(
            dataloader=val_loader, img_size=self.test_size, confthre=self.test_conf,
            nmsthre=self.nmsthre, num_classes=self.num_classes, testdev=testdev,
            per_class_AP=True, per_class_AR=True,
        )
