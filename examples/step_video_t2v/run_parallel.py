import time

from stepvideo.config import parse_args
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.parallel import initialize_parall_group
from stepvideo.utils import setup_seed

import mindspore as ms

if __name__ == "__main__":
    args = parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        jit_config={"jit_level": "O0"},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        # max_device_memory="59GB",
        # jit_syntax_level=ms.STRICT,
    )

    initialize_parall_group(args, ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)

    setup_seed(args.seed)

    s_time = time.time()
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(ms.bfloat16)
    pipeline.setup_api(
        vae_url=args.vae_url,
        caption_url=args.caption_url,
    )

    prompt = args.prompt
    videos = pipeline(
        prompt=prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50],
    )
