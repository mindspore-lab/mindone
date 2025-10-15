from dataclasses import dataclass
from math import ceil
from typing import Literal, Optional, Union

from ..datasets.aspect_v2 import get_image_size


@dataclass
class SamplingOption:
    """
    Handle the configuration for the sampling process, including
    resolution settings, guidance parameters, and other sampling-specific options.

    Attributes:
        cond_type: The type of conditioning for the sampling process. Can be "t2i", "i2v", or "v2v".
        use_t2i2v: Whether to use a T2I (Flux) -> I2V (OpenSora) pipeline for text-to-video generation. Default is True.
        width: The width of the image/video in pixels.
        height: The height of the image/video in pixels.
        resolution: The resolution of the image/video (e.g., "256px", "768px").
                    If provided, it will override height and width.
        aspect_ratio: The aspect ratio of the image/video (e.g., "16:9", "1:1").
                      If provided, it will override height and width.
        num_frames: The number of frames for video generation. Default is 1 (image).
        num_steps: The number of sampling steps. Default is 50.
        guidance: The classifier-free guidance scale for text. Default is 4.0.
        text_osci: Whether to use oscillation for text guidance. Default is False.
        guidance_img: The classifier-free guidance scale for image, or for the guidance on condition for i2v and v2v.
        image_osci: Whether to use oscillation for image guidance. Default is False.
        scale_temporal_osci: Whether to use temporal scaling for image guidance. Default is False.
        shift: Whether to shift the schedule. Default is True.
        method: The sampling method. Default is "i2v".
        temporal_reduction: Temporal reduction factor. Default is 1.
        is_causal_vae: Whether using causal VAE. Default is False.
        flow_shift: Flow shift parameter. Default is None.
        num_samples: The number of samples to generate. Default is 1.
        motion_score: The motion score for video generation. Default is 4.0.
        batch_size: The number of samples to generate at once. Default is 1.

    Note:
        Either both `height` and `width` or both `resolution` and `aspect_ratio` must be provided.
        If the latter, the height and width will be calculated based on the resolution and aspect ratio.
        The final height and width will be adjusted to be multiples of 16.
    """

    cond_type: Literal["t2i", "t2v", "i2v", "v2v"] = "t2i"
    use_t2i2v: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    num_frames: int = 1
    num_steps: int = 50
    guidance: float = 4.0
    text_osci: bool = False
    guidance_img: Optional[float] = None
    image_osci: bool = False
    scale_temporal_osci: bool = False
    shift: bool = True
    method: Literal["i2v", "distill"] = "i2v"
    temporal_reduction: int = 1
    is_causal_vae: bool = False
    flow_shift: Optional[float] = None
    num_samples: int = 1
    motion_score: Union[Literal["dynamic"], int] = 4
    batch_size: int = 1
    resized_resolution: Optional[str] = None

    def __post_init__(self):
        if self.resolution is not None or self.aspect_ratio is not None:
            assert (
                self.resolution is not None and self.aspect_ratio is not None
            ), "Both `resolution` and `aspect_ratio` must be provided."
            self.height, self.width = get_image_size(self.resolution, self.aspect_ratio, training=False)
        else:
            assert self.height is not None and self.width is not None, "Both `height` and `width` must be provided."

        self.height, self.width = ceil(self.height / 16) * 16, ceil(self.width / 16) * 16
