"""Single-frame reference model: RF-DETR on the BuzzSet Challenge keyframes.

Dataset must be in COCO format with train/ valid/ (and optionally test/) subdirs,
each holding _annotations.coco.json + images. The keyframe annotations were
already placed in the challenge train/ and valid/ folders.

Install (once):  pip install rfdetr
Run:             python train_rfdetr.py --epochs 30 --batch-size 4
"""
import argparse


def main():
    ap = argparse.ArgumentParser("RF-DETR on BuzzSet keyframes")
    ap.add_argument("--dataset-dir",
                    default=r"F:\BuzzSet-org-videos\BuzzSet-org-videos\BuzzSet_Challenge\BuzzSet_challenge")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=560, help="multiple of 56 for RF-DETR")
    ap.add_argument("--model", default="base", choices=["nano", "small", "base", "large"])
    ap.add_argument("--output-dir",
                    default=r"F:\BuzzSet-org-videos\buzzset_yolovpp_comparison\runs\rfdetr_v2")
    args = ap.parse_args()

    import rfdetr
    cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase", "large": "RFDETRLarge"}[args.model]
    model = getattr(rfdetr, cls)()
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
