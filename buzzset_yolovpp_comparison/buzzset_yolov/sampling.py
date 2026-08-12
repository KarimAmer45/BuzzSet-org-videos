"""Class-balanced sampling for the stage-1 BuzzSet detector.

BuzzSet is ~80% bee, so uniform sampling barely shows the model the rare classes
(hoverfly, other_insect, moth). These helpers weight each frame by the rarest
class it contains and draw frames with replacement accordingly, so rare insects
are seen far more often -- without touching the upstream YOLOV code.
"""
import itertools

import numpy as np

try:
    import torch
    from torch.utils.data.sampler import Sampler
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover
        dist = None
except Exception:  # torch absent (e.g. offline weight analysis); class unused then
    torch = None
    dist = None
    Sampler = object


def class_balanced_image_weights(dataset, alpha=0.5):
    """Per-image sampling weights for class-balanced oversampling.

    For class ``c`` with instance fraction ``f_c`` the class weight is
    ``(1 / f_c) ** alpha``; each image takes the max class weight among the
    objects it holds, so a frame with a rare insect is oversampled even if it
    also holds common ones. ``alpha`` tunes aggressiveness: 0 = uniform,
    0.5 = sqrt inverse-frequency (gentle), 1.0 = full inverse-frequency.

    Returns ``(weights, info)``; ``info`` carries counts/weights for logging.
    """
    n = len(dataset.class_ids)
    counts = np.zeros(n, dtype=np.float64)
    per_image = []
    for ann in dataset.annotations:
        res = ann[0]
        cls = res[:, 4].astype(int) if len(res) else np.empty(0, dtype=int)
        per_image.append(cls)
        for c in cls:
            if 0 <= c < n:
                counts[c] += 1.0

    freq = np.maximum(counts, 1.0)
    freq = freq / freq.sum()
    class_w = (1.0 / freq) ** float(alpha)
    background = float(class_w.min())

    weights = np.full(len(per_image), background, dtype=np.float64)
    for i, cls in enumerate(per_image):
        if len(cls):
            weights[i] = float(class_w[cls].max())

    info = {
        "alpha": float(alpha),
        "counts": counts.astype(int).tolist(),
        "class_weights": [round(float(w), 3) for w in class_w],
    }
    return weights, info


class WeightedInfiniteSampler(Sampler):
    """Infinite sampler that draws indices with replacement by per-image weight.

    Drop-in replacement for YOLOX's ``InfiniteSampler``: same infinite stream and
    rank/world-size sharding, but indices come from a weighted multinomial rather
    than a uniform permutation.
    """

    def __init__(self, weights, seed=0, rank=0, world_size=1):
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._size = int(self._weights.numel())
        assert self._size > 0
        self._seed = int(seed)
        if dist is not None and dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        yield from itertools.islice(
            self._infinite_indices(), self._rank, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            idx = torch.multinomial(
                self._weights, self._size, replacement=True, generator=g
            )
            yield from idx.tolist()

    def __len__(self):
        return self._size // self._world_size
