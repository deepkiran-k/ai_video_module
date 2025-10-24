'''
File: core/video_editor.py
DESCRIPTION:
- Handles the final video assembly using MoviePy.
- Prepends an optional intro clip (e.g., from veo3).
- Stitches the selected screen recording segments.
- Converts the final video to 9:16 with a blurred background.
- Synchronizes the final video with the generated voiceover.
'''

import moviepy.editor as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from moviepy.video.fx.all import crop
from PIL import Image, ImageFilter
import json
import sys

class VideoEditor:
    """
    Handles final video assembly:
    - Prepends an optional intro clip.
    - Stitches screen recording segments.
    - Converts to 9:16 with a blurred background.
    - Synchronizes audio.
    """

    def __init__(self, output_size: tuple = (720, 1280)):
        self.output_size = output_size

    def _blur_frame(self, image: np.ndarray, sigma: int) -> np.ndarray:
        """Applies a Gaussian blur to a single video frame."""
        from PIL import Image, ImageFilter
        pil_image = Image.fromarray(image)
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(sigma))
        return np.array(blurred_image)

    def create_guide_video(
        self,
        original_video_path: str,
        selected_steps: List[Dict[str, Any]],
        audio_path: str,
        script: str,  # Kept for potential future use (e.g., captions)
        output_filename: str = "final_guide.mp4",
        intro_clip_path: Optional[str] = None  # <-- NEW ARGUMENT
    ) -> str:
        print("🎬 Starting final video editing process...")

        # --- 1. Cut and Collect Clips ---
        clips = []

        # --- NEW: Check for and add the intro clip first ---
        if intro_clip_path and Path(intro_clip_path).exists():
            print(f"  + Attaching intro clip: {intro_clip_path}")
            try:
                intro_clip = mp.VideoFileClip(intro_clip_path)
                clips.append(intro_clip)
            except Exception as e:
                print(f"⚠️ Warning: Could not load intro clip. Error: {e}")
        # --- End of new logic ---

        for step in selected_steps:
            start = step.get("start_time")
            end = step.get("end_time")
            if start is not None and end is not None and end > start:
                try:
                    clip = mp.VideoFileClip(original_video_path).subclip(start, end)
                    clips.append(clip)
                except Exception as e:
                    print(f"⚠️ Warning: Could not cut segment from {start}s to {end}s. Error: {e}")

        if not clips:
            raise ValueError("❌ No valid video clips could be created.")

        # --- 2. Combine Clips ---
        stitched_video = mp.concatenate_videoclips(clips, method="compose").fadein(0.5).fadeout(0.5)

        # --- 3. Create Blurred Background and Foreground for 9:16 ---
        # Background: Scale to fill height, crop to 9:16, and blur.
        background_clip = stitched_video.resize(height=self.output_size[1])
        background_clip = crop(
            background_clip, 
            width=self.output_size[0], height=self.output_size[1], 
            x_center=background_clip.w / 2, y_center=background_clip.h / 2
        )
        background_clip = background_clip.fl_image(lambda frame: self._blur_frame(frame, 15))

        # Foreground: Scale to fit width and place in the center.
        foreground_clip = stitched_video.resize(width=self.output_size[0])
        foreground_clip = foreground_clip.set_position(('center', 'center'))
        
        final_video = mp.CompositeVideoClip([background_clip, foreground_clip], size=self.output_size)

        # --- 4. Add Audio ---
        if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
            voiceover_audio = mp.AudioFileClip(audio_path)
            # Synchronize audio duration with video duration
            final_duration = min(final_video.duration, voiceover_audio.duration)
            final_video = final_video.subclip(0, final_duration)
            final_video = final_video.set_audio(voiceover_audio.subclip(0, final_duration))
        else:
            print("⚠️ Audio file is missing or empty. Video will have no sound.")

        # --- 5. Export ---
        print(f"💾 Writing final video to {output_filename}...")
        final_video.write_videofile(
            str(output_filename), codec='libx264', audio_codec='aac',
            temp_audiofile='temp-audio.m4a', remove_temp=True,
            threads=8, logger='bar'
        )

        # --- Cleanup ---
        for clip in clips: clip.close()
        stitched_video.close()
        background_clip.close()
        foreground_clip.close()
        if 'voiceover_audio' in locals(): voiceover_audio.close()
        final_video.close()

        print(f"✅ Video editing complete! Final video saved to '{output_filename}'.")
        return output_filename

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python core/video_editor.py <video_path> <steps_json_path> <audio_path> [intro_clip_path]")
        sys.exit(1)
    
    video_path, steps_json_path, audio_path = sys.argv[1], sys.argv[2], sys.argv[3]
    intro_path = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        with open(steps_json_path, 'r') as f: selected_steps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading or parsing steps JSON: {e}"); sys.exit(1)
        
    editor = VideoEditor()
    editor.create_guide_video(
        original_video_path=video_path, 
        selected_steps=selected_steps,
        audio_path=audio_path, 
        script="Test script for CLI execution.",
        output_filename="cli_edited_video.mp4",
        intro_clip_path=intro_path
    )