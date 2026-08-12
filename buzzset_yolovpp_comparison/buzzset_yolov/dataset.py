import json
import os
import random
import re
from collections import defaultdict
from pathlib import PurePosixPath

import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset as TorchDataset


VIDEO_RE = re.compile(
    r"^(?P<key>\d{8}_plot[^_]+_\d{2}\.\d{2})_(?P<frame>\d+)\.[^.]+$"
)


def _image_name(image_record):
    return image_record.get("name") or image_record.get("file_name")


def _infer_video_key_and_frame(file_name, fallback_frame=0):
    base = PurePosixPath(str(file_name).replace("\\", "/")).name
    match = VIDEO_RE.match(base)
    if match:
        return match.group("key"), int(match.group("frame"))

    stem = os.path.splitext(base)[0]
    if "_" in stem:
        maybe_key, maybe_frame = stem.rsplit("_", 1)
        if maybe_frame.isdigit():
            return maybe_key, int(maybe_frame)
    return stem, int(fallback_frame)


def _remove_useless_info(coco):
    dataset = coco.dataset
    dataset.pop("info", None)
    dataset.pop("licenses", None)
    for image in dataset.get("images", []):
        image.pop("license", None)
        image.pop("coco_url", None)
        image.pop("date_captured", None)
        image.pop("flickr_url", None)
    for anno in dataset.get("annotations", []):
        anno.pop("segmentation", None)


class BuzzSetVideoDataset(TorchDataset):
    """YOLOV-compatible video sampler over BuzzSet COCO frame annotations."""

    def __init__(
        self,
        image_root,
        ann_file,
        img_size=(576, 576),
        preproc=None,
        lframe=0,
        gframe=16,
        val=False,
        mode="random",
        tnum=-1,
        formal=False,
        seed=42,
    ):
        super().__init__()
        self.image_root = image_root
        self.ann_file = ann_file
        self.input_dim = img_size
        self.img_size = img_size
        self.preproc = preproc
        self.lframe = int(lframe)
        self.gframe = int(gframe)
        self.val = bool(val)
        self.mode = mode
        self.tnum = int(tnum) if tnum is not None else -1
        self.formal = bool(formal)
        self.seed = int(seed)

        self.coco = COCO(ann_file)
        _remove_useless_info(self.coco)
        self.ids = sorted(self.coco.getImgIds())
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])

        self.image_records = {img["id"]: img for img in self.coco.dataset["images"]}
        self.name_to_id = {}
        for img in self.coco.dataset["images"]:
            name = _image_name(img)
            if name in self.name_to_id:
                raise ValueError(f"Duplicate BuzzSet image name in annotations: {name}")
            self.name_to_id[name] = img["id"]

        self.annotations_by_id = {
            image_id: self._load_anno_from_id(image_id) for image_id in self.ids
        }
        self.video_sequences = self._load_video_sequences()
        self.res = self._photo_to_sequence(self.lframe, self.gframe)

    def __len__(self):
        return len(self.res)

    def _load_video_sequences(self):
        grouped = defaultdict(list)
        has_sid = all("sid" in img for img in self.coco.dataset["images"])

        for img in self.coco.dataset["images"]:
            name = _image_name(img)
            if has_sid:
                key = int(img["sid"])
                frame = int(img.get("fid", 0))
            else:
                key, frame = _infer_video_key_and_frame(name, img["id"])
            grouped[key].append((frame, name))

        sequences = []
        for key in sorted(grouped):
            ordered = [name for _, name in sorted(grouped[key], key=lambda item: item[0])]
            sequences.append(ordered)
        return sequences

    def _load_anno_from_id(self, image_id):
        image = self.image_records[image_id]
        width = image["width"]
        height = image["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objects = []

        for obj in annotations:
            x1 = max(0.0, float(obj["bbox"][0]))
            y1 = max(0.0, float(obj["bbox"][1]))
            x2 = min(float(width), x1 + max(0.0, float(obj["bbox"][2])))
            y2 = min(float(height), y1 + max(0.0, float(obj["bbox"][3])))
            area = float(obj.get("area", (x2 - x1) * (y2 - y1)))
            if area > 0 and x2 >= x1 and y2 >= y1:
                objects.append((x1, y1, x2, y2, obj["category_id"]))

        res = np.zeros((len(objects), 5), dtype=np.float32)
        for idx, (x1, y1, x2, y2, category_id) in enumerate(objects):
            res[idx, 0:4] = [x1, y1, x2, y2]
            res[idx, 4] = self.class_ids.index(category_id)

        scale = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= scale

        img_info = (height, width)
        resized_info = (int(height * scale), int(width * scale))
        return res, img_info, resized_info, _image_name(image)

    def _photo_to_sequence(self, lframe, gframe):
        res = []
        rng = random.Random(self.seed if self.val else None)

        for sequence in self.video_sequences:
            element = list(sequence)
            element_len = len(element)
            required = lframe + gframe
            if element_len < required:
                if self.formal:
                    res.append(element)
                continue

            if self.mode == "random":
                if lframe == 0:
                    rng.shuffle(element)
                    split_num = int(element_len / gframe)
                    for idx in range(split_num):
                        res.append(element[idx * gframe : (idx + 1) * gframe])
                else:
                    split_num = int(element_len / lframe)
                    all_local_frame = element[: split_num * lframe]
                    for idx in range(split_num):
                        local_frames = all_local_frame[idx * lframe : (idx + 1) * lframe]
                        reference_pool = element[: idx * lframe] + element[(idx + 1) * lframe :]
                        global_frames = rng.sample(reference_pool, gframe)
                        res.append(local_frames + global_frames)
            elif self.mode == "uniform":
                split_num = int(element_len / gframe)
                all_uniform_frame = element[: split_num * gframe]
                for idx in range(split_num):
                    res.append(all_uniform_frame[idx::split_num])
            elif self.mode == "gl":
                split_num = int(element_len / lframe)
                all_local_frame = element[: split_num * lframe]
                for idx in range(split_num):
                    reference_pool = element[: idx * lframe] + element[(idx + 1) * lframe :]
                    global_frames = rng.sample(reference_pool, gframe)
                    res.append(all_local_frame[idx * lframe : (idx + 1) * lframe] + global_frames)
            else:
                raise ValueError(f"Unsupported BuzzSet video sample mode: {self.mode}")

        if self.val:
            rng.shuffle(res)
            return res if self.tnum == -1 else res[: self.tnum]

        rng.shuffle(res)
        return res

    def pull_item(self, name):
        image_id = self.name_to_id[name]
        annos, img_info, _resized_info, img_name = self.annotations_by_id[image_id]
        img_path = os.path.join(self.image_root, img_name)
        img = cv2.imread(img_path)
        assert img is not None, f"BuzzSet image not found: {img_path}"

        height, width = img.shape[:2]
        img_info = (height, width)
        scale = min(self.img_size[0] / height, self.img_size[1] / width)
        img = cv2.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_name

    def __getitem__(self, name):
        img, target, img_info, img_name = self.pull_item(name)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_name

