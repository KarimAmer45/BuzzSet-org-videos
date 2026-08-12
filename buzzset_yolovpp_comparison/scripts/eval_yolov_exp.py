import argparse
import os
import random
import sys
import warnings


def make_parser():
    parser = argparse.ArgumentParser("BuzzSet YOLOV++ evaluation")
    parser.add_argument("-f", "--exp-file", required=True)
    parser.add_argument("-c", "--ckpt", required=True)
    parser.add_argument("--yolov-root", default=os.environ.get("YOLOV_ROOT", "YOLOV-master"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-split", default=None, choices=["valid", "test"])
    parser.add_argument("--tsize", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--nms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--formal",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Evaluate every frame, including clips shorter than gframe "
        "(default on; pass --no-formal to skip short clips, matching the "
        "faster per-epoch eval used during training).",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


def main():
    args = make_parser().parse_args()
    yolov_root = os.path.abspath(args.yolov_root)
    if yolov_root not in sys.path:
        sys.path.insert(0, yolov_root)

    import torch
    import torch.backends.cudnn as cudnn
    from loguru import logger
    from yolox.exp import get_exp
    from yolox.utils import get_model_info

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "A fixed seed was set, so cudnn runs in deterministic mode "
            "(cudnn.benchmark disabled)."
        )
    else:
        cudnn.benchmark = True

    exp = get_exp(args.exp_file, None)
    exp.merge(args.opts)
    if args.eval_split is not None:
        exp.eval_name = args.eval_split
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms

    model = exp.get_model()
    logger.info("Model Summary: {}", get_model_info(model, exp.test_size))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.cuda().eval()

    val_loader = exp.get_eval_loader(
        batch_size=args.batch_size,
        data_num_workers=args.workers,
        formal=args.formal,
    )
    evaluator = exp.get_evaluator(val_loader)
    *_, summary = evaluator.evaluate(model, distributed=False, half=args.fp16)
    logger.info("\n{}", summary)


if __name__ == "__main__":
    main()
