
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
    Handles video formatting and final assembly.
    1. format_clip_9x16: Formats a single clip to 9:16.
    2. stitch_and_save: Concatenates all finished clips.
    """

    def __init__(self, output_size: tuple = (720, 1280), zoom_factor: float = 1.1):
        self.output_size = output_size
        self.zoom_factor = zoom_factor

    def _blur_frame(self, image: np.ndarray, sigma: int) -> np.ndarray:
        """Applies a Gaussian blur to a single video frame."""
        from PIL import Image, ImageFilter
        pil_image = Image.fromarray(image)
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(sigma))
        return np.array(blurred_image)

    def format_clip_9x16(
        self,
        raw_clip: mp.VideoClip,
        is_intro: bool = False
    ) -> mp.VideoClip:
        """
        Formats a single raw video clip into the 9:16 aspect ratio.
        - Intros are resized to fill the screen.
        - Screen recordings get the blurred-background and zoom effect.
        """
        
        if is_intro:
            # --- Format Intro Clip (Full Screen) ---
            print("  Formatting intro clip (full-screen)...")
            formatted_clip = raw_clip.resize(height=self.output_size[1])
            formatted_clip = crop(
                formatted_clip, 
                width=self.output_size[0], height=self.output_size[1], 
                x_center=formatted_clip.w / 2, y_center=formatted_clip.h / 2
            )
            return formatted_clip
        
        else:
            # --- Format Screen Recording (Blur BG + Zoom FG) ---
            print("  Formatting screen clip (blur/zoom)...")
            
            # Create Blurred Background
            background_clip = raw_clip.resize(height=self.output_size[1])
            background_clip = crop(
                background_clip, 
                width=self.output_size[0], height=self.output_size[1], 
                x_center=background_clip.w / 2, y_center=background_clip.h / 2
            )
            background_clip = background_clip.fl_image(lambda frame: self._blur_frame(frame, 15))

            # Create Zooming Foreground
            foreground_clip = raw_clip.resize(width=self.output_size[0])
            
            # --- NEW SAFETY CHECK ---
            duration = foreground_clip.duration
            if duration > 0.01: # Check for a valid duration
                zoomed_foreground = foreground_clip.resize(
                    lambda t: 1 + ((self.zoom_factor - 1) * (t / duration))
                )
                zoomed_foreground = zoomed_foreground.set_position(('center', 'center'))
            else:
                # Duration is zero, skip zoom
                zoomed_foreground = foreground_clip.set_position(('center', 'center'))
            # --- END SAFETY CHECK ---
            
            # Composite them
            formatted_clip = mp.CompositeVideoClip(
                [background_clip, zoomed_foreground], 
                size=self.output_size
            ).fadein(0.2).fadeout(0.2)
            
            return formatted_clip

    def stitch_and_save(
        self,
        final_clips: List[mp.VideoClip],
        output_filename: str
    ):
        """
        Concatenates a list of final (audio-synced) clips and saves.
        """
        if not final_clips:
            print("❌ No clips to stitch. Aborting.")
            return

        print(f"🧵 Stitching {len(final_clips)} final clips together...")
        
        # Concatenate all the finished clips
        final_video = mp.concatenate_videoclips(final_clips, method="compose")

        print(f"💾 Writing final video to {output_filename}...")
        final_video.write_videofile(
            str(output_filename), codec='libx264', audio_codec='aac',
            temp_audiofile='temp-audio.m4a', remove_temp=True,
            threads=8, logger='bar'
        )
        
        # --- Cleanup ---
        for clip in final_clips:
            clip.close()
        final_video.close()
        
        print(f"✅ Video editing complete!")


if __name__ == "__main__":
    print("This script is not intended to be run directly.")
    print("Please run 'run_pipeline_direct.py' to generate a video.")