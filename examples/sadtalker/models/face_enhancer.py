from models.gfpgan.gfpganer import GFPGANer


model_path = "checkpoints/ms/ms_gfpgan.ckpt"
arch = "clean"
channel_multiplier = 2
bg_upsampler = None

restorer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler,
)
