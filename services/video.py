# services/video.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional
import math

# Optional YouTube download
from yt_dlp import YoutubeDL
import cv2  # opencv

def try_download_youtube_mp4(url: str, outdir: str = "downloads") -> Optional[str]:
    """Download a YouTube video as MP4 (no cookies). Returns local path or None."""
    os.makedirs(outdir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "format": "(bv*[ext=mp4][vcodec^=avc1]/bv*)+ba/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "noprogress": True,
        "retries": 3,
        "concurrent_fragment_downloads": 1,
        "nocheckcertificate": True,
        # DO NOT touch cookiesfrombrowser; corporate Chrome locks its DB
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # prepare_filename returns the file that would have been downloaded,
            # but due to merges the actual path is info["requested_downloads"][i]["filepath"]
            # safer to scan the outdir for the id with .mp4
            vid = info.get("id")
            cand = os.path.join(outdir, f"{vid}.mp4")
            if os.path.exists(cand):
                return cand
            # Fallback: look at requested_downloads
            for d in info.get("requested_downloads", []):
                f = d.get("filepath")
                if f and os.path.exists(f):
                    return f
    except Exception:
        return None
    return None

def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def extract_frames(video_path: str, outdir: str = "frames", every_sec: float = 2.0, max_frames: int = 180) -> List[Tuple[str, str]]:
    """
    Sample frames roughly every `every_sec` seconds (simple & robust).
    Returns [(timestamp, jpg_path), ...] and writes files to outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur_s = total / fps if fps > 0 else 0

    step = max(int(fps * every_sec), 1)
    frames = []
    idx = 0
    saved = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        ts = _hhmmss(t)
        out = os.path.join(outdir, f"frame_{ts.replace(':','-')}.jpg")
        cv2.imwrite(out, frame)
        frames.append((ts, out))
        saved += 1
        if saved >= max_frames:
            break
        idx += step

    cap.release()
    return frames
