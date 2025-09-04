import os
import cv2
import torch
import subprocess
import re
import shutil
import folder_paths
from .utils import get_cache_path, ensure_cache_dir, download_if_needed


class LoadVideoFromURL:
    """
    Download a video from URL (with simple caching) and load it as a tensor stack with audio.

    Parameters
    ----------
    force_fps : int
        * 0 → keep the source FPS.
        * >0 → cap FPS to this value, **only if** the source FPS is higher.
          The clip duration stays identical; frames are dropped deterministically
          to reach the requested frame‑rate.
    """

    CATEGORY = "video"
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "audio", "video_info")
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

    def get_ffmpeg_path(self):
        """Find FFmpeg executable."""
        # Try system FFmpeg first
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        
        # Try common locations
        common_paths = [
            "ffmpeg",
            "ffmpeg.exe",
            os.path.join(os.getcwd(), "ffmpeg"),
            os.path.join(os.getcwd(), "ffmpeg.exe")
        ]
        
        for path in common_paths:
            if os.path.isfile(path):
                return os.path.abspath(path)
        
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg or place it in your PATH.")

    def extract_audio(self, video_path, start_time=0, duration=0):
        """
        Extract audio from video file using FFmpeg.
        Returns a dictionary with waveform tensor and sample rate.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        ffmpeg_path = self.get_ffmpeg_path()
        
        # Build FFmpeg command - extract audio stream specifically
        args = [ffmpeg_path, "-i", video_path]
        
        if start_time > 0:
            args += ["-ss", str(start_time)]
        
        if duration > 0:
            args += ["-t", str(duration)]
        
        # Extract audio stream and convert to 32-bit float PCM
        args += ["-vn", "-acodec", "pcm_f32le", "-ar", "44100", "-ac", "2", "-f", "f32le", "-"]
        
        try:
            print(f"[LoadVideoURL] FFmpeg command: {' '.join(args)}")
            result = subprocess.run(args, capture_output=True, check=True)
            
            # Convert bytes to torch tensor
            audio_data = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)
            
            # Parse stderr to get sample rate and channels
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            match = re.search(r', (\d+) Hz, (\w+)', stderr_text)
            
            if match:
                sample_rate = int(match.group(1))
                channel_layout = match.group(2)
                channels = {"mono": 1, "stereo": 2}.get(channel_layout, 2)
            else:
                sample_rate = 44100
                channels = 2
            
            # Reshape audio data to match VHS expectations
            if len(audio_data) > 0:
                # VHS expects: (batch, channels, samples)
                audio_data = audio_data.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
                print(f"[LoadVideoURL] Audio tensor shape: {audio_data.shape}")
            else:
                # Create empty audio tensor
                audio_data = torch.zeros((1, channels, 0))
                print(f"[LoadVideoURL] Empty audio tensor created: {audio_data.shape}")
            
            return {'waveform': audio_data, 'sample_rate': sample_rate}
            
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown error"
            stdout_text = e.stdout.decode('utf-8', errors='replace') if e.stdout else "No output"
            print(f"[LoadVideoURL] FFmpeg stderr: {stderr_text}")
            print(f"[LoadVideoURL] FFmpeg stdout: {stdout_text}")
            raise Exception(f"Failed to extract audio from {video_path}: {stderr_text}")

    class LazyAudio:
        """Lazy loader for audio data."""
        def __init__(self, extract_func, video_path, start_time, duration):
            self.extract_func = extract_func
            self.video_path = video_path
            self.start_time = start_time
            self.duration = duration
            self._data = None
        
        def __getitem__(self, key):
            if self._data is None:
                self._data = self.extract_func(self.video_path, self.start_time, self.duration)
            return self._data[key]
        
        def __iter__(self):
            if self._data is None:
                self._data = self.extract_func(self.video_path, self.start_time, self.duration)
            return iter(self._data)
        
        def __len__(self):
            if self._data is None:
                self._data = self.extract_func(self.video_path, self.start_time, self.duration)
            return len(self._data)

    def get_audio_from_video(self, video_path, start_time=0, duration=0):
        """
        Create a lazy audio loader for the video file.
        """
        return self.LazyAudio(self.extract_audio, video_path, start_time, duration)

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
        # Calculate timing for audio extraction
        # ------------------------------------------------------------
        target_frame_time = 1.0 / loaded_fps if loaded_fps else 0
        start_time = skip_first_frames / fps if fps else 0
        
        # Calculate audio duration based on loaded frames
        if frame_load_cap > 0:
            audio_duration = frame_load_cap * target_frame_time
        else:
            # Calculate how many frames we'll actually load
            available_frames = (total_frames - skip_first_frames) // stride
            audio_duration = available_frames * target_frame_time

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
        # 3) Extract audio (eager loading for VHS compatibility)
        # -----------------------------------------------------------------
        try:
            audio = self.extract_audio(dest, start_time, audio_duration)
        except Exception as e:
            print(f"[LoadVideoURL] Warning: Could not extract audio: {e}")
            # Return empty audio structure that VHS can handle
            audio = {'waveform': torch.zeros((1, 2, 0)), 'sample_rate': 44100}

        # -----------------------------------------------------------------
        # 4) Wrap up
        # -----------------------------------------------------------------
        if frame_count == 0:
            raise ValueError("No frames were loaded from the video.")

        frames = torch.stack(frames)

        loaded_duration = frame_count / loaded_fps if loaded_fps else 0

        # Convert video_info dict to string for output
        video_info_str = f"FPS: {loaded_fps:.2f}, Frames: {frame_count}, Size: {new_width}x{new_height}, Duration: {loaded_duration:.2f}s"
        
        # Debug logging for audio
        print(f"[LoadVideoURL] Audio extracted: {type(audio)}, keys: {list(audio.keys()) if isinstance(audio, dict) else 'Not a dict'}")
        if isinstance(audio, dict) and 'waveform' in audio:
            print(f"[LoadVideoURL] Audio waveform shape: {audio['waveform'].shape}, sample_rate: {audio['sample_rate']}")
        
        return (frames, frame_count, audio, video_info_str)