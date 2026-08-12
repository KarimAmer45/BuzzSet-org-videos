from pycocotools.coco import COCO

from yolox.data.datasets.coco import COCODataset, remove_useless_info
from yolox.data.datasets.datasets_wrapper import Dataset


class BuzzSetCocoDataset(COCODataset):
    """Single-frame COCO dataset for the BuzzSet splits.

    The stock COCODataset expects ``<data_dir>/annotations/<json>`` for labels
    and ``<data_dir>/<name>/<image>`` for images. BuzzSet keeps each split's
    annotation file and its frames together in one folder, so this subclass
    takes the image folder and annotation file directly and reuses everything
    else from COCODataset (annotation parsing, resizing, caching, __getitem__).
    """

    def __init__(self, image_root, ann_file, img_size=(640, 640), preproc=None, cache=False):
        Dataset.__init__(self, img_size)
        self.data_dir = image_root
        self.name = ""  # images live directly under image_root
        self.json_file = ann_file
        self.coco = COCO(ann_file)
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()
