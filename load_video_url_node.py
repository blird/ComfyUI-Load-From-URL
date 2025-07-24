import os, hashlib, shutil, subprocess, time, requests, re, tempfile
from typing import Literal

from tqdm import tqdm
import cv2
import torch
import folder_paths

# ------------------------------------------------------------
# Quick CDN‑cache check (1 HEAD request, 200 ms)
# ------------------------------------------------------------
_CACHE_HIT_RE = re.compile(r"\bHIT\b", re.I)


def _edge_is_hot(url: str, timeout: float = 2.0) -> bool:
    """Return True if Front Door says the object is in edge cache."""
    try:
        r = requests.head(url, timeout=timeout)
        # Age header present and > 0 => cached
        if int(r.headers.get("Age", "0")) > 0:
            return True
        # X-Cache or X-FD-Cache-Type may say "HIT"
        x_cache = r.headers.get("X-Cache") or r.headers.get("X-FD-Cache-Type", "")
        return bool(_CACHE_HIT_RE.search(x_cache))
    except Exception:
        # If HEAD fails, fall back to assuming it's cold
        return False


# ------------------------------------------------------------
# Helper: choose downloader
# ------------------------------------------------------------
def _detect_downloader(url: str, preference: str = "auto") -> str:
    """
    'auto' = python if edge‑hot, else aria2; otherwise honor explicit choice.
    """
    if preference != "auto":
        return preference

    # Decide based on cache state
    is_hot = _edge_is_hot(url)
    if is_hot:
        return "python"  # single stream is fastest on warm cache
    if shutil.which("aria2c"):
        return "aria2"
    return "python"


class LoadVideoFromURL:
    """
    Download a video from URL (with simple caching) and load it as a tensor stack.

    Parameters
    ----------
    force_fps : int
        * 0 → keep the source FPS.
        * >0 → cap FPS to this value, **only if** the source FPS is higher.
          The clip duration stays identical; frames are dropped deterministically
          to reach the requested frame‑rate.
    """

    CATEGORY = "video"
    RETURN_TYPES = ("IMAGE", "INT", "VHS_VIDEOINFO")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "load_video_from_url"

    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_videos")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------- ComfyUI interface -------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4"}),
                "force_fps": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "force_size": (
                    [
                        "Disabled",
                        "Custom Height",
                        "Custom Width",
                        "Custom",
                        "256x?",
                        "?x256",
                        "256x256",
                        "512x?",
                        "?x512",
                        "512x512",
                    ],
                ),
                "custom_width": (
                    "INT",
                    {"default": 512, "min": 0, "max": 8192, "step": 8},
                ),
                "custom_height": (
                    "INT",
                    {"default": 512, "min": 0, "max": 8192, "step": 8},
                ),
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "step": 1},
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1000000, "step": 1},
                ),
            },
            "optional": {
                # Match the downloader choices from the Lora node
                "downloader": (["auto", "aria2", "python"],),
            },
        }

    # ------- internal helpers -------
    def _target_path(self, url: str) -> str:
        """Return deterministic cache path for the URL."""
        # Retain original extension if present, fall back to .mp4
        ext = os.path.splitext(url)[1] or ".mp4"
        return os.path.join(
            self.cache_dir, hashlib.md5(url.encode()).hexdigest() + ext
        )

    def _download_aria2(self, url: str, dest: str):
        subprocess.run(
            [
                "aria2c",
                "-x16",
                "-s16",
                "-k8M",
                "--file-allocation=none",
                "--retry-wait=1",
                "--max-tries=5",
                "-o",
                os.path.basename(dest),
                "-d",
                os.path.dirname(dest),
                url,
            ],
            check=True,
        )

    def _download_python(self, url: str, dest: str):
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                desc=os.path.basename(dest), total=total, unit="iB", unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    bar.update(f.write(chunk))

    def _download_if_needed(self, url: str, preference: str):
        dest = self._target_path(url)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest, 0.0, "cached"

        tool = _detect_downloader(url, preference)
        start = time.time()
        try:
            if tool == "aria2":
                self._download_aria2(url, dest)
            else:
                # Treat 'python' and any unrecognised tool the same
                self._download_python(url, dest)
        except Exception:
            # If aria2 fails, fall back once
            if tool == "aria2":
                tool = "python"
                self._download_python(url, dest)
            else:
                raise
        return dest, time.time() - start, tool

    # ------- public entry point -------
    def load_video_from_url(
        self,
        url,
        force_fps,
        force_size,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        downloader="auto",
    ):
        # -----------------------------------------------------------------
        # 1) Download (or fetch from cache)
        # -----------------------------------------------------------------
        path, seconds, tool = self._download_if_needed(url, downloader)
        print(f"[LoadVideoURL] Downloader: {tool}, Time: {seconds:.2f}s")

        # -----------------------------------------------------------------
        # 2) Decode & process video
        # -----------------------------------------------------------------
        cap = cv2.VideoCapture(path)

        # Get source properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps else 0

        # ------------------------------------------------------------
        # Calculate target size
        # ------------------------------------------------------------
        if force_size != "Disabled":
            if force_size == "Custom Width":
                new_height = int(height * (custom_width / width))
                new_width = custom_width
            elif force_size == "Custom Height":
                new_width = int(width * (custom_height / height))
                new_height = custom_height
            elif force_size == "Custom":
                new_width, new_height = custom_width, custom_height
            else:
                target_width, target_height = map(
                    int, force_size.replace("?", "0").split("x")
                )
                if target_width == 0:
                    new_width = int(width * (target_height / height))
                    new_height = target_height
                else:
                    new_height = int(height * (target_width / width))
                    new_width = target_width
        else:
            new_width, new_height = width, height

        # ------------------------------------------------------------
        # Determine frame stride to satisfy force_fps + select_every_nth
        # ------------------------------------------------------------
        auto_stride = 1
        if fps and force_fps > 0 and fps > force_fps:
            auto_stride = max(1, int(round(fps / force_fps)))

        stride = max(select_every_nth, auto_stride)

        # Effective FPS after down‑sampling
        loaded_fps = fps / stride if fps else 0

        # ------------------------------------------------------------
        # Read frames
        # ------------------------------------------------------------
        frames = []
        frame_count = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i < skip_first_frames:
                continue

            if (i - skip_first_frames) % stride != 0:
                continue

            if force_size != "Disabled":
                frame = cv2.resize(frame, (new_width, new_height))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)

            frame_count += 1

            if frame_load_cap > 0 and frame_count >= frame_load_cap:
                break

        cap.release()

        # -----------------------------------------------------------------
        # 3) Wrap up
        # -----------------------------------------------------------------
        if frame_count == 0:
            raise ValueError("No frames were loaded from the video.")

        frames = torch.stack(frames)

        loaded_duration = frame_count / loaded_fps if loaded_fps else 0

        video_info = {
            "source_fps": fps,
            "source_frame_count": total_frames,
            "source_duration": duration,
            "source_width": width,
            "source_height": height,
            "loaded_fps": loaded_fps,
            "loaded_frame_count": frame_count,
            "loaded_duration": loaded_duration,
            "loaded_width": new_width,
            "loaded_height": new_height,
        }

        return (frames, frame_count, video_info)
