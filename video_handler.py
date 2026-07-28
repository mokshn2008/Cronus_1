"""
Video support for Cronus.

Honest scope: Ollama's API has no dedicated "video" content type, so this
works by extracting several evenly-spaced frames from the video and sending
them to Gemma as a sequence of images. Gemma reasons over the frame
sequence, which gives it a real (if sampled, not continuous) sense of what
happens in the video. Audio/spoken content is NOT captured by this --
that would need a separate transcription step (e.g. Whisper), not included
here. Requires ffmpeg installed on the system (`brew install ffmpeg` on Mac).
"""

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames(video_path: str, num_frames: int = 6) -> List[str]:
    """
    Extracts num_frames evenly-spaced JPEG frames from the video into a
    temp directory and returns their file paths.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError("Could not read video duration -- file may be corrupt or unsupported.")

    tmp_dir = tempfile.mkdtemp(prefix="cronus_frames_")
    frame_paths = []

    for i in range(num_frames):
        timestamp = (duration / (num_frames + 1)) * (i + 1)  # skip exact start/end
        out_path = str(Path(tmp_dir) / f"frame_{i:02d}.jpg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", out_path,
            ],
            capture_output=True,
        )
        if Path(out_path).exists():
            frame_paths.append(out_path)

    return frame_paths


def encode_frame_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_video_content_blocks(video_path: str, num_frames: int = 6) -> List[dict]:
    """
    Returns a list of multimodal content blocks (text intro + image frames)
    ready to append into an OpenAI-style message's "content" list.
    """
    if not ffmpeg_available():
        return [{
            "type": "text",
            "text": "[Video uploaded, but ffmpeg is not installed on this system -- "
                    "cannot extract frames. Install with `brew install ffmpeg` (Mac) "
                    "and try again. Do not guess at the video's contents.]",
        }]

    try:
        frames = extract_frames(video_path, num_frames=num_frames)
    except Exception as exc:
        return [{
            "type": "text",
            "text": f"[Video uploaded, but frame extraction failed: {exc}. "
                    f"Do not guess at the video's contents.]",
        }]

    if not frames:
        return [{
            "type": "text",
            "text": "[Video uploaded, but no frames could be extracted. "
                    "Do not guess at the video's contents.]",
        }]

    blocks = [{
        "type": "text",
        "text": f"[The following {len(frames)} images are evenly-spaced frames sampled "
                f"from an uploaded video, in chronological order. They do NOT capture "
                f"audio/speech. Reason about what happens across the sequence, but be "
                f"clear this is a sampled sequence, not continuous footage.]",
    }]
    for frame_path in frames:
        b64 = encode_frame_b64(frame_path)
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    return blocks
