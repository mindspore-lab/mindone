import argparse
import os

from fid import FrechetInceptionDistance
from fid.utils import get_image_paths

from mindspore import context


def eval_fid(args):
    # get image file paths
    real_imgs = get_image_paths(args.real_dir)
    gen_imgs = get_image_paths(args.gen_dir)
    print("Num real images: ", len(real_imgs))
    print("Num generated images: ", len(gen_imgs))

    # compute fid
    print(f"Backend: {args.backend}")
    if args.backend == "ms":
        device_id = int(os.getenv("DEVICE_ID", 0))
        context.set_context(
            mode=0,
            device_id=device_id,
        )

        fid_scorer = FrechetInceptionDistance(batch_size=args.batch_size)
        fid_score = fid_scorer.compute(gen_imgs, real_imgs)
    elif args.backend == "pt":
        from fid.utils import compute_torchmetric_fid

        fid_score = compute_torchmetric_fid(gen_imgs, real_imgs)
    else:
        raise ValueError(f"Unknown backend: {args.backend}. Valid backend: [ms, pt]")

    print("FID: ", fid_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, help="Directory to real images")
    parser.add_argument("--gen_dir", type=str, help="Directory to generated images")
    parser.add_argument(
        "--backend",
        type=str,
        default="ms",
        help="Backend to do inception model inference for FID compute. Option: ms, pt.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    eval_fid(args)
