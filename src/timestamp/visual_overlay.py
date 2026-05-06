"""Timestamp overlay utilities for videos."""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.timestamp.get_timestamp import get_uniform_sample_timestamps

FONT_SIZE = 40
TEXT_COLOR = (255, 0, 0)
TEXT_MARGIN = 16
FONT = ImageFont.load_default()


def overlay_timestamp(
    video_inputs: list[torch.Tensor],  # each tensor: [num_frames, C, H, W]
    video_paths: list[str],
    max_frames: int,
    fps: float,
    mode: str = "time",

) -> list[torch.Tensor]:
    """Overlay timestamp."""
    overlaid_videos: list[torch.Tensor] = []

    for video_idx, (video, video_path) in enumerate(zip(video_inputs, video_paths)):
        overlaid_frames = []
        for frame_idx, frame in enumerate(video):
            # from frame to numpy
            image_dtype = frame.dtype
            image_device = frame.device
            frame = frame.detach().cpu()
            is_unit_range = bool(frame.numel() > 0 and frame.max().item() <= 1.0)
            if is_unit_range:
                frame = frame.mul(255.0)

            # draw frame idx or time onto the image
            image = Image.fromarray(frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())
            draw = ImageDraw.Draw(image)

            if mode == "frame":
                label =  f"Frame {frame_idx + 1}"
            elif mode == "time":
                _, _, _, timestamps = get_uniform_sample_timestamps(
                    video_path = video_path,
                    fps = fps,
                    max_frames = max_frames,
                )
                label = f"{timestamps[frame_idx]:.2f}"
            else:
                raise NotImplementedError(f"Unsupported overlay mode: {mode}")

            left, top, right, bottom = draw.textbbox((0, 0), label, font=FONT)
            x = max(TEXT_MARGIN, image.width - (right - left) - TEXT_MARGIN)
            y = max(TEXT_MARGIN, image.height - (bottom - top) - TEXT_MARGIN)
            draw.text((x, y), label, fill=TEXT_COLOR, font=FONT)
            # from numpy to frame
            frame = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32)
            if is_unit_range:
                frame = frame / 255.0
            overlaid_frames.append(frame.to(dtype=image_dtype, device=image_device))
        overlaid_videos.append(torch.stack(overlaid_frames, dim=0))
    return overlaid_videos
