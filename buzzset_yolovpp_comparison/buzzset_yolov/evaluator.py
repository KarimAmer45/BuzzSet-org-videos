from . import BUZZSET_CLASSES


def buzzset_categories(class_names=BUZZSET_CLASSES):
    return [
        {"supercategory": "insect", "id": idx, "name": name}
        for idx, name in enumerate(class_names)
    ]


def build_vid_evaluator(
    val_loader,
    img_size,
    confthre,
    nmsthre,
    class_names=BUZZSET_CLASSES,
    lframe=0,
    gframe=16,
    first_only=False,
):
    from yolox.evaluators.vid_evaluator_v2 import VIDEvaluator

    evaluator = VIDEvaluator(
        dataloader=val_loader,
        img_size=img_size,
        confthre=confthre,
        nmsthre=nmsthre,
        num_classes=len(class_names),
        lframe=lframe,
        gframe=gframe,
        first_only=first_only,
    )
    categories = buzzset_categories(class_names)
    evaluator.vid_to_coco["categories"] = categories
    evaluator.vid_to_coco_ori["categories"] = categories
    return evaluator

