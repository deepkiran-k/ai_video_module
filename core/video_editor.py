'''
import os
import sys
import json 
import moviepy.editor as mp
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add this import for the new blur method
from moviepy.video.fx.all import crop

class VideoEditor:
    """
    Handles the final video assembly using a robust blur method
    for professional 9:16 conversion.
    """

    def __init__(self, output_size: tuple = (720, 1280)):
        self.output_size = output_size

    def create_guide_video(
        self,
        original_video_path: str,
        selected_steps: List[Dict[str, Any]],
        audio_path: str,
        script: str, # Script is kept for future captioning use
        output_filename: str = "final_guide.mp4"
    ) -> str:
        print("🎬 Starting final video editing process...")

        # --- 1. Cut and Collect Clips ---
        clips = []
        for step in selected_steps:
            start = step.get("start_time")
            end = step.get("end_time")
            if start is not None and end is not None:
                try:
                    clip = mp.VideoFileClip(original_video_path).subclip(start, end)
                    clips.append(clip)
                except Exception as e:
                    print(f"⚠️ Warning: Could not cut segment from {start}s to {end}s. Error: {e}")

        if not clips:
            raise ValueError("❌ No valid video clips could be created.")

        # --- 2. Combine Clips ---
        stitched_video = mp.concatenate_videoclips(clips, method="compose").fadein(0.5).fadeout(0.5)

        # --- 3. Create Blurred Background and Foreground ---
        # Background: Scale to fill height, then crop to 9:16
        background_clip = stitched_video.resize(height=self.output_size[1])
        background_clip = crop(background_clip, 
                               width=self.output_size[0], 
                               height=self.output_size[1], 
                               x_center=background_clip.w/2, 
                               y_center=background_clip.h/2)
        # Apply blur using a more stable method
        background_clip = background_clip.fl_image(lambda frame: self._blur_frame(frame, 15))


        # Foreground: Scale to fit width, place in center
        foreground_clip = stitched_video.resize(width=self.output_size[0])
        foreground_clip = foreground_clip.set_position(('center', 'center'))
        
        # Layer the foreground on top of the background
        final_video = mp.CompositeVideoClip([background_clip, foreground_clip], size=self.output_size)

        # --- 4. Add Audio ---
        voiceover_audio = mp.AudioFileClip(audio_path)
        if voiceover_audio.duration > final_video.duration:
            voiceover_audio = voiceover_audio.subclip(0, final_video.duration)
        final_video = final_video.set_audio(voiceover_audio)
        
        # --- 5. Export ---
        print(f"💾 Writing final video to {output_filename}...")
        final_video.write_videofile(
            output_filename,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger='bar'
        )

        # --- Cleanup ---
        for clip in clips:
            clip.close()
        stitched_video.close()
        background_clip.close()
        foreground_clip.close()
        voiceover_audio.close()
        final_video.close()

        print(f"✅ Video editing complete! Final video saved to '{output_filename}'.")
        return output_filename

    def _blur_frame(self, image, sigma):
        """A more stable way to apply blur to each frame using Pillow."""
        from PIL import Image, ImageFilter
        pil_image = Image.fromarray(image)
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(sigma))
        # Convert back to a numpy array, which moviepy expects
        return np.array(blurred_image)


if __name__ == "__main__":
   
    # We simplified the command for testing, so only 3 arguments are needed now.
    if len(sys.argv) < 4:
        print("Usage: python core/video_editor.py <video_path> <steps_json_path> <audio_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    steps_json_path = sys.argv[2]
    audio_path = sys.argv[3]

    try:
        with open(steps_json_path, 'r') as f:
            selected_steps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading or parsing steps JSON: {e}")
        sys.exit(1)
        
    editor = VideoEditor()
    # The script argument is kept for future use but not needed for the CLI test
    script_placeholder = "This is a test script." 
    editor.create_guide_video(
        original_video_path=video_path,
        selected_steps=selected_steps,
        audio_path=audio_path,
        script=script_placeholder,
        output_filename="final_edited_video.mp4"
    )



#################

class VideoEditor:
    """
    Handles the final video assembly using MoviePy for robust editing
    and audio synchronization. (Captions removed for simplicity).
    """

    def __init__(self, output_size: tuple = (720, 1280)):
        """
        Initializes the video editor.

        Args:
            output_size: The final resolution of the video (width, height).
                         Defaults to 9:16 aspect ratio.
        """
        self.output_size = output_size

    def create_guide_video(
        self,
        original_video_path: str,
        selected_steps: List[Dict[str, Any]],
        audio_path: str,
        output_filename: str = "final_guide.mp4"
    ) -> str:
        """
        Creates the final video from clips and audio.
        """
        print("🎬 Starting video editing process (no captions)...")

        # --- 1. Cut and Collect Relevant Clips ---
        clips = []
        for step in selected_steps:
            start = step.get("start_time")
            end = step.get("end_time")
            if start is not None and end is not None:
                try:
                    # Added fadein/fadeout to each clip for smooth transitions
                    clip = mp.VideoFileClip(original_video_path).subclip(start, end).fadein(0.5).fadeout(0.5)
                    clips.append(clip)
                except Exception as e:
                    print(f"⚠️ Warning: Could not cut segment from {start}s to {end}s. Error: {e}")
        
        if not clips:
            raise ValueError("❌ No valid video clips could be created from the selected segments.")

        # --- 2. Combine Clips ---
        final_video = mp.concatenate_videoclips(clips, method="compose")
        
        # --- 3. Prepare and Add Audio ---
        voiceover_audio = mp.AudioFileClip(audio_path)
        if voiceover_audio.duration > final_video.duration:
            voiceover_audio = voiceover_audio.subclip(0, final_video.duration)
        
        final_video = final_video.set_audio(voiceover_audio)

        # --- 4. Resize to 9:16 and Export ---
        final_video_resized = final_video.fx(mp.vfx.crop, \
            width=self.output_size[0], height=self.output_size[1], \
            x_center=final_video.w/2, y_center=final_video.h/2)

        print(f"💾 Writing final video to {output_filename}...")
        final_video_resized.write_videofile(
            output_filename,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger='bar'
        )

        # --- Cleanup ---
        for clip in clips:
            clip.close()
        final_video.close()
        voiceover_audio.close()
        final_video_resized.close()

        print(f"✅ Video editing complete! Final video saved to '{output_filename}'.")
        return output_filename


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 4:
        print("Usage: python core/video_editor.py <video_path> <steps_json_path> <audio_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    steps_json_path = sys.argv[2]
    audio_path = sys.argv[3]

    try:
        with open(steps_json_path, 'r') as f:
            selected_steps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading or parsing steps JSON: {e}")
        sys.exit(1)
        
    editor = VideoEditor()
    editor.create_guide_video(
        original_video_path=video_path,
        selected_steps=selected_steps,
        audio_path=audio_path,
        output_filename="test_edited_video.mp4"
    )
########################

import moviepy.editor as mp
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from moviepy.video.fx.all import crop
import json
import sys

class VideoEditor:
    """
    Handles the final video assembly using a robust blur method
    for professional 9:16 conversion.
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
        output_filename: str = "final_guide.mp4"
    ) -> str:
        print("🎬 Starting final video editing process...")

        # --- 1. Cut and Collect Clips ---
        clips = []
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
            output_filename, codec='libx264', audio_codec='aac',
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
        print("Usage: python core/video_editor.py <video_path> <steps_json_path> <audio_path>")
        sys.exit(1)
    
    video_path, steps_json_path, audio_path = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        with open(steps_json_path, 'r') as f: selected_steps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading or parsing steps JSON: {e}"); sys.exit(1)
        
    editor = VideoEditor()
    editor.create_guide_video(
        original_video_path=video_path, selected_steps=selected_steps,
        audio_path=audio_path, script="Test script for CLI execution.",
        output_filename="cli_edited_video.mp4"
    )
'''


import moviepy.editor as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from moviepy.video.fx.all import crop
from PIL import Image, ImageFilter

class VideoEditor:
    """
    Handles final video assembly using a robust blur method for professional 9:16 conversion.
    """
    def __init__(self, output_size: tuple = (720, 1280)):
        self.output_size = output_size

    def _blur_frame(self, image: np.ndarray, sigma: int) -> np.ndarray:
        """Applies a Gaussian blur to a single video frame using Pillow."""
        pil_image = Image.fromarray(image)
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(sigma))
        return np.array(blurred_image)

    def create_guide_video(self, original_video_path: str, selected_steps: List[Dict[str, Any]], audio_path: str, script: str, output_filename: str = "final_guide.mp4") -> str:
        print("🎬 Starting final video editing process...")
        clips = []
        for step in selected_steps:
            start, end = step.get("start_time"), step.get("end_time")
            if start is not None and end is not None and end > start:
                try:
                    clip = mp.VideoFileClip(original_video_path).subclip(start, end)
                    clips.append(clip)
                except Exception as e:
                    print(f"⚠️ Warning: Could not cut segment from {start}s to {end}s. Error: {e}")

        if not clips: raise ValueError("❌ No valid video clips could be created.")

        stitched_video = mp.concatenate_videoclips(clips, method="compose").fadein(0.5).fadeout(0.5)

    # In core/video_editor.py

    # def create_guide_video(
    #     self,
    #     original_video_path: str,
    #     selected_steps: List[Dict[str, Any]],
    #     audio_path: str,
    #     script: str,
    #     output_filename: str = "final_guide.mp4",
    #     intro_clip_path: Optional[str] = None  # <-- NEW optional argument
    # ) -> str:
    #     print("🎬 Starting final video editing process...")

    #     # --- 1. Cut and Collect Clips ---
    #     clips = []

    #     # --- NEW: Check for and add the intro clip first ---
    #     if intro_clip_path and Path(intro_clip_path).exists():
    #         print(f"  + Attaching intro clip: {intro_clip_path}")
    #         intro_clip = mp.VideoFileClip(intro_clip_path)
    #         clips.append(intro_clip)
    #     # --- End of new logic ---

    #     for step in selected_steps:
    #         start = step.get("start_time")
    #         end = step.get("end_time")
    #         if start is not None and end is not None and end > start:
    #             try:
    #                 clip = mp.VideoFileClip(original_video_path).subclip(start, end)
    #                 clips.append(clip)
    #             except Exception as e:
    #                 print(f"⚠️ Warning: Could not cut segment from {start}s to {end}s. Error: {e}")

    #     if not clips:
    #         raise ValueError("❌ No valid video clips could be created.")

    #     # --- The rest of the function remains the same ---
    #     stitched_video = mp.concatenate_videoclips(clips, method="compose").fadein(0.5).fadeout(0.5)

        # (Code for 9:16 formatting, adding audio, and exporting is unchanged)
        # ...
        background_clip = stitched_video.resize(height=self.output_size[1])
        background_clip = crop(background_clip, width=self.output_size[0], height=self.output_size[1], x_center=background_clip.w / 2, y_center=background_clip.h / 2)
        background_clip = background_clip.fl_image(lambda frame: self._blur_frame(frame, 15))

        foreground_clip = stitched_video.resize(width=self.output_size[0])
        foreground_clip = foreground_clip.set_position(('center', 'center'))
        
        final_video = mp.CompositeVideoClip([background_clip, foreground_clip], size=self.output_size)

        if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
            voiceover_audio = mp.AudioFileClip(audio_path)
            final_duration = min(final_video.duration, voiceover_audio.duration)
            final_video = final_video.subclip(0, final_duration).set_audio(voiceover_audio.subclip(0, final_duration))
        else:
            print("⚠️ Audio file is missing or empty. Video will have no sound.")

        print(f"💾 Writing final video to {output_filename}...")
        # final_video.write_videofile(output_filename, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, threads=8, logger='bar')
        # Convert the output_filename Path object to a string
        final_video.write_videofile(str(output_filename), codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, threads=8, logger='bar')

        # Cleanup
        for clip in clips: clip.close()
        stitched_video.close()
        background_clip.close()
        foreground_clip.close()
        if 'voiceover_audio' in locals(): voiceover_audio.close()
        final_video.close()

        print(f"✅ Video editing complete! Final video saved to '{output_filename}'.")
        return output_filename