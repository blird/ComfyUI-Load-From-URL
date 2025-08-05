import os
import cv2
import torch
import folder_paths
from .utils import get_cache_path, ensure_cache_dir, download_if_needed


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
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "load_video_from_url"

    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_videos")
        ensure_cache_dir(self.cache_dir)

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
        dest = get_cache_path(url, self.cache_dir, ".mp4")
        seconds, tool = download_if_needed(url, dest, downloader)
        print(f"[LoadVideoURL] Downloader: {tool}, Time: {seconds:.2f}s")

        # -----------------------------------------------------------------
        # 2) Decode & process video
        # -----------------------------------------------------------------
        cap = cv2.VideoCapture(dest)

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

        # Convert video_info dict to string for output
        video_info_str = f"FPS: {loaded_fps:.2f}, Frames: {frame_count}, Size: {new_width}x{new_height}, Duration: {loaded_duration:.2f}s"
        
        return (frames, frame_count, video_info_str)
