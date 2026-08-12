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
spec = importlib.util.spec_from_file_location("swin_tiny_base_detector", base_exp_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseExp = module.Exp

from buzzset_yolov import torch_compat  # noqa: F401  (PyTorch >=2.6 torch.load shim)
from buzzset_yolov import BUZZSET_CLASSES
from buzzset_yolov.coco_dataset import BuzzSetCocoDataset


class Exp(BaseExp):
    """Stage 1 of the two-stage recipe: a single-frame YOLOX Swin-Tiny detector
    fine-tuned on the BuzzSet frames. Its best checkpoint is the initialization
    for the stage-2 YOLOV++ temporal-aggregation run."""

    def __init__(self):
        super().__init__()
        self.exp_name = "buzzset_stage1_swin_tiny_det"
        self.num_classes = len(BUZZSET_CLASSES)
        self.class_names = BUZZSET_CLASSES

        self.data_root = os.environ.get(
            "BUZZSET_ROOT",
            os.path.join(WORKSPACE_ROOT, "BuzzSet-org-videos", "BuzzSetV2_split"),
        )
        self.train_name = "train"
        self.val_name = "valid"
        self.test_name = "test"
        self.eval_name = "valid"

        # Resolution is the dominant small-object lever: at 576 about a quarter
        # of the insect boxes fall under ~12 px after downscaling; 768 cuts that
        # to ~11%. Heavier than the fast 576 config, so expect longer training.
        # Kept equal to the YOLOV++ stage size for a clean comparison.
        self.input_size = (768, 768)
        self.test_size = (768, 768)
        self.multiscale_range = 2
        # Re-enable mosaic for the sparse frames (mostly 1-2 insects each), but
        # raise the scale floor so its tiles do not shrink the tiny insects to
        # nothing. Mixup stays off; it tends to blur small-object localization.
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0

        # Fine-tune from the pretrained detector weights. With mosaic on, the
        # last no_aug_epochs run without it so the model settles on real frames.
        self.max_epoch = 25
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.eval_interval = 1
        self.print_interval = 20
        self.data_num_workers = 2
        self.test_conf = 0.001
        self.nmsthre = 0.65

        # Class-balanced oversampling. BuzzSet is ~80% bee, so without this
        # the rare classes (hoverfly, other_insect, moth) are barely seen.
        # Frames are weighted by the rarest class they contain; alpha tunes
        # how hard: 0 = off, 0.5 = sqrt inverse-frequency, 1.0 = full.
        self.class_balance = True
        self.class_balance_alpha = 0.3  # gentler: avoid overfitting moth (~200 imgs)

        self.output_dir = os.path.join(PROJECT_ROOT, "runs")

    def _ann_file(self, split_name):
        return os.path.join(self.data_root, split_name, "_annotations.coco.json")

    def _image_root(self, split_name):
        return os.path.join(self.data_root, split_name)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        import torch.distributed as dist
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        from loguru import logger
        from buzzset_yolov.sampling import (
            WeightedInfiniteSampler,
            class_balanced_image_weights,
        )

        with wait_for_the_master():
            dataset = BuzzSetCocoDataset(
                image_root=self._image_root(self.train_name),
                ann_file=self._ann_file(self.train_name),
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
            )

        image_weights = None
        if getattr(self, "class_balance", False):
            image_weights, info = class_balanced_image_weights(
                dataset, getattr(self, "class_balance_alpha", 0.5)
            )
            logger.info(
                "Stage-1 class-balanced sampling (alpha={}): counts={}, "
                "per-class weights={}".format(
                    info["alpha"], info["counts"], info["class_weights"]
                )
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        seed = self.seed if self.seed else 0
        if image_weights is not None:
            sampler = WeightedInfiniteSampler(image_weights, seed=seed)
        else:
            sampler = InfiniteSampler(len(self.dataset), seed=seed)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug
        )
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        import torch
        import torch.distributed as dist
        from yolox.data import ValTransform

        split = self.test_name if testdev else self.eval_name
        valdataset = BuzzSetCocoDataset(
            image_root=self._image_root(split),
            ann_file=self._ann_file(split),
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            per_class_AP=True,
            per_class_AR=True,
        )
