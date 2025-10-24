# 

import sys
import json
from pathlib import Path
import moviepy.editor as mp
# --- NEW IMPORTS for 9:16 conversion ---
import numpy as np
from moviepy.video.fx.all import crop
from PIL import Image, ImageFilter

import sys
sys.stdout.reconfigure(encoding='utf-8')

# --- NEW HELPER FUNCTION for 9:16 format ---
def _blur_frame(image: np.ndarray, sigma: int) -> np.ndarray:
    """Applies a Gaussian blur to a single video frame using Pillow."""
    pil_image = Image.fromarray(image)
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(sigma))
    return np.array(blurred_image)

def assemble_final_video(project_name: str):
    """
    FINAL VERSION: Assembles clips, syncs with audio, and converts the final
    video to a 9:16 aspect ratio with a blurred background.
    """
    print(f" Assembling final 9:16 video for project: {project_name}")
    
    # Define paths
    clips_dir = Path("generated_clips") / project_name
    narrations_dir = Path("narrations") / project_name
    plan_path = Path("outputs") / f"{project_name}_production_plan.json"
    final_video_path = Path("outputs") / f"{project_name}_creative_final_video.mp4"

    # Validate paths
    if not plan_path.exists(): print(f"❌ Plan not found: {plan_path}"); return
    if not narrations_dir.is_dir(): print(f"❌ Narrations folder not found: {narrations_dir}"); return
    if not clips_dir.is_dir(): print(f"❌ Generated clips folder not found: {clips_dir}"); return

    with open(plan_path, "r") as f:
        production_plan = json.load(f)
    
    video_clips_list = []
    audio_clips_list = []

    print("  Loading and preparing clips...")
    for shot in sorted(production_plan, key=lambda x: x['shot_number']):
        video_clip_path = clips_dir / shot['output_filename']
        audio_clip_path = narrations_dir / f"{video_clip_path.stem}.mp3"

        if video_clip_path.exists() and audio_clip_path.exists():
            print(f"  + Preparing {video_clip_path.name}")
            video_clip = mp.VideoFileClip(str(video_clip_path))
            audio_clip = mp.AudioFileClip(str(audio_clip_path))
            min_duration = min(video_clip.duration, audio_clip.duration)
            video_clips_list.append(video_clip.subclip(0, min_duration))
            audio_clips_list.append(audio_clip.subclip(0, min_duration))
        elif video_clip_path.exists():
            print(f"  + Preparing video-only clip {video_clip_path.name}")
            video_clips_list.append(mp.VideoFileClip(str(video_clip_path)))
        else:
            print(f"⚠️ Warning: Missing video clip for {shot['output_filename']}, skipping.")

    if not video_clips_list: print("❌ No valid clips found to assemble."); return

    # Concatenate the landscape video and audio tracks separately
    stitched_landscape_video = mp.concatenate_videoclips(video_clips_list, method="compose")
    final_audio_track = mp.concatenate_audioclips(audio_clips_list)

    # --- NEW: Apply 9:16 formatting to the entire stitched video ---
    print("  Converting to 9:16 format...")
    output_size = (720, 1280)
    
    # Create the blurred background
    background = stitched_landscape_video.resize(height=output_size[1])
    background = crop(background, width=output_size[0], height=output_size[1], x_center=background.w/2, y_center=background.h/2)
    background = background.fl_image(lambda frame: _blur_frame(frame, 20))
    
    # Create the centered foreground
    foreground = stitched_landscape_video.resize(width=output_size[0]).set_position(('center', 'center'))
    
    # Composite the foreground on the background
    final_video_9x16 = mp.CompositeVideoClip([background, foreground], size=output_size)
    # --- End of 9:16 formatting ---

    # Set the final audio track and sync duration
    final_video = final_video_9x16.set_audio(final_audio_track)
    if final_video.duration > final_audio_track.duration:
        final_video = final_video.subclip(0, final_audio_track.duration)

    print(f" Writing final video to {final_video_path}...")
    final_video.write_videofile(
        str(final_video_path), codec='libx264', audio_codec='aac',
        temp_audiofile='temp-audio.m4a', remove_temp=True, logger='bar'
    )
    
    # Cleanup
    for clip in video_clips_list: clip.close()
    for clip in audio_clips_list: clip.close()
    final_video.close()
    
    print(f"\n Success! Final 9:16 video saved as '{final_video_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assemble_video.py <project_name>"); sys.exit(1)
    
    assemble_final_video(sys.argv[1])