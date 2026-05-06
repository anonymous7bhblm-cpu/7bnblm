"""Generate a temporary mp4 with a bounded long edge using ffmpeg."""
import os
import subprocess
import tempfile


def encode_video_max_long_edge(src_path: str, max_side: int) -> str:
    """
    Write an H.264 mp4 whose long edge does not exceed max_side px and return the tempfile path.
    The caller should remove it with os.remove. Requires ffmpeg on the system.
    """
    max_side = max(64, int(max_side))
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    vf = (
        f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
        ":flags=lanczos"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-an",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if os.path.isfile(out_path):
            os.remove(out_path)
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}) resizing {src_path!r}: {err}"
        )
    return out_path
