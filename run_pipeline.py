import sys
from pathlib import Path
import json
import os
import moviepy.editor as mp
from moviepy.video.fx.all import speedx # Import speedx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes from our core modules
from core.ocr_extractor import OCRExtractor, extract_frames
from core.action_analyzer import ActionAnalyzer, save_action_log
from core.ai_manager import AIManager
from core.video_editor import VideoEditor
from typing import List, Dict, Any
import shutil

import sys
sys.stdout.reconfigure(encoding='utf-8')

def run_full_pipeline(video_path_str: str, user_prompt: str, voice: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path_str}")
        return

    video_name = video_path.stem
    
    # --- Create dedicated temp folders ---
    output_dir = Path("outputs") / video_name
    temp_audio_dir = output_dir / "temp_audio"
    temp_audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting AI video guide generation for '{video_name}'...")
    print(f"💬 User Prompt: \"{user_prompt}\"")
    print(f"🎤 Selected Voice: \"{voice}\"")

    # --- 1. OCR Extraction (Unchanged) ---
    ocr_results_path = output_dir / f"{video_name}_ocr_results.json"
    print("\n--- Step 1: Extracting Text (OCR) ---")
    ocr_extractor = OCRExtractor()
    frame_data = extract_frames(str(video_path), interval=2)
    ocr_results = {ts: ocr_extractor.extract_text_from_frame(fp) for fp, ts in frame_data}
    ocr_extractor.save_ocr_results(ocr_results, str(ocr_results_path))
    print("✅ OCR extraction complete.")

    # --- 2. Action Analysis (Unchanged) ---
    action_log_path = output_dir / f"{video_name}_action_log.json"
    print("\n--- Step 2: Analyzing Actions ---")
    analyzer = ActionAnalyzer(ocr_results)
    action_log = analyzer.analyze(str(video_path))
    save_action_log(action_log, str(action_log_path))
    print("✅ Action analysis complete.")

    # --- 3. AI Processing (NEW: Single Pass) ---
    print("\n--- Step 3: AI Processing (Generating Clip Plan) ---")
    ai_manager = AIManager()
    clip_plan = ai_manager.create_clip_plan(action_log, user_prompt)
    if not clip_plan:
        print("❌ AI failed to generate a clip plan. Exiting."); return
    print("✅ AI clip plan generation complete.")

    # --- 4. Initialize Editor and Clip List (Unchanged) ---
    editor = VideoEditor()
    final_clips_with_audio = []

    # --- 5. Process INTRO Clip (Unchanged) ---
    print("\n--- Step 5: Processing Intro Clip ---")
    intro_clip_path = Path("generated_clips") / video_name / "shot_01_intro.mp4"
    intro_prompt = ai_manager.create_intro_plan(user_prompt)
    ai_manager.generate_intro_video(intro_prompt, intro_clip_path) # Prints manual instructions

    if intro_clip_path.exists():
        try:
            raw_intro_clip = mp.VideoFileClip(str(intro_clip_path))
            silent_intro_clip = editor.format_clip_9x16(raw_intro_clip, is_intro=True)
            
            intro_script = ai_manager.generate_narration(
                user_prompt, "", silent_intro_clip.duration, is_intro=True
            )
            intro_audio_path = temp_audio_dir / "audio_intro.mp3"
            ai_manager.create_voiceover(intro_script, str(intro_audio_path), voice)

            if intro_audio_path.exists():
                audio_clip = mp.AudioFileClip(str(intro_audio_path))
                video_duration = silent_intro_clip.duration
                final_audio = mp.CompositeAudioClip([
                    audio_clip.set_start(0)
                ]).set_duration(video_duration)
                
                final_intro = silent_intro_clip.set_audio(final_audio)
                final_clips_with_audio.append(final_intro)
            else:
                final_clips_with_audio.append(silent_intro_clip)
        except Exception as e:
            print(f"⚠️ Failed to process intro clip: {e}")
    else:
        print("⚠️ Intro clip not found, skipping...")

    # --- 6. Process BODY Clips (NEW: Simplified Multi-Pass) ---
    print(f"\n--- Step 6: Processing {len(clip_plan)} Body Clips ---")
    
    # --- NO MORE GROUPING OR TIMING LOGIC NEEDED ---
    
    clip_processing_data = [] # This will store all our data for each clip

    # --- PASS 1: Cut Video & Generate DRAFT Scripts ---
    print("  Pass 1: Cutting video clips and generating draft scripts...")
    for i, clip in enumerate(clip_plan):
        
        # Get start/end/description directly from the AI's plan
        start_time = clip['start_time']
        end_time = clip['end_time']
        description = clip['description']
        
        print(f"    Processing clip {i+1}: {description}...")

        # Ensure start_time is not negative
        start_time = max(0.0, start_time)

        if end_time <= start_time:
            print(f"    ⚠️ SKIPPING: Clip has zero or negative duration (Start: {start_time}, End: {end_time}).")
            continue
        
        try:
            # 1. Cut and Format Video
            raw_clip = mp.VideoFileClip(str(video_path)).subclip(start_time, end_time)
            silent_clip = editor.format_clip_9x16(raw_clip, is_intro=False)
            
            # 2. Generate DRAFT Narration
            is_outro = (i == len(clip_plan) - 1)
            draft_script = ai_manager.generate_narration(
                user_prompt, description, silent_clip.duration, is_outro=is_outro
            )
            
            # 3. Store data for the next passes
            clip_processing_data.append({
                "clip_number": i + 1,
                "silent_clip": silent_clip,
                "draft_script": draft_script,
                "description": description, # We need this for the speed-up check
                "duration": silent_clip.duration,
                "is_outro": is_outro
            })

        except Exception as e:
            print(f"⚠️ Failed to process clip {i+1} during Pass 1: {e}")

    # --- PASS 2: AI Script Polishing (Unchanged) ---
    scripts_to_polish = [
        {
            "script": data["draft_script"],
            "description": data["description"],
            "target_duration": data["duration"]
        }
        for data in clip_processing_data if not data["is_outro"]
    ]
    
    if scripts_to_polish:
        polished_script_data = ai_manager.polish_scripts(scripts_to_polish)
        
        script_index = 0
        for data in clip_processing_data:
            if not data["is_outro"]:
                if script_index < len(polished_script_data):
                    data["polished_script"] = polished_script_data[script_index]["script"]
                    script_index += 1
                else:
                    data["polished_script"] = data["draft_script"] 
            else:
                data["polished_script"] = data["draft_script"] 
    else:
        for data in clip_processing_data:
             data["polished_script"] = data["draft_script"]

    # --- PASS 3: Generate Audio & Combine Clips (Unchanged) ---
    print("  Pass 3: Generating audio from polished scripts and combining...")
    for data in clip_processing_data:
        
        clip_num = data["clip_number"]
        silent_clip = data["silent_clip"]
        polished_script = data["polished_script"]
        description = data["description"].lower() # Get description for checking
        
        print(f"    Generating audio for clip {clip_num}...")
        
        try:
            clip_audio_path = temp_audio_dir / f"audio_clip_{clip_num}.mp3"
            ai_manager.create_voiceover(polished_script, str(clip_audio_path), voice)
            
            if clip_audio_path.exists():
                audio_clip = mp.AudioFileClip(str(clip_audio_path))
                video_duration = silent_clip.duration
                audio_duration = audio_clip.duration

                is_boring_action = "type" in description or "typing" in description or "scroll" in description
                is_long_silence = (video_duration > audio_duration + 2.0)

                if is_boring_action and is_long_silence:
                    print(f"    -> Action is 'typing/scrolling' and audio is short. Speeding up clip {clip_num} to match audio.")
                    speed_factor = video_duration / audio_duration
                    
                    final_clip = silent_clip.fx(speedx, speed_factor)
                    final_clip = final_clip.set_duration(audio_duration).set_audio(audio_clip)
                    
                    final_clips_with_audio.append(final_clip)
                
                else:
                    final_audio = mp.CompositeAudioClip([
                        audio_clip.set_start(0)
                    ]).set_duration(video_duration)

                    final_clip = silent_clip.set_audio(final_audio)
                    final_clips_with_audio.append(final_clip)
            else:
                final_clips_with_audio.append(silent_clip)
        
        except Exception as e:
            print(f"⚠️ Failed to generate audio/combine clip {clip_num}: {e}")
            final_clips_with_audio.append(silent_clip) # Add silent clip as fallback

    # --- 7. Final Assembly (Unchanged) ---
    final_video_path = output_dir / f"{video_name}_final_guide.mp4"
    print(f"\n--- Step 7: Assembling Final Video ---")
    editor.stitch_and_save(final_clips_with_audio, str(final_video_path))

    # --- 8. Cleanup (Unchanged) ---
    try:
        shutil.rmtree(temp_audio_dir)
        print(f"🧹 Cleaned up temp audio files.")
    except Exception as e:
        print(f"⚠️ Warning: Could not remove temp audio directory: {e}")

    print(f"\n🎉 Success! Your scene-synced video guide is saved as '{final_video_path}'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py \"<path_to_video>\" \"<your_prompt>\"")
        sys.exit(1)
    
    video_file = sys.argv[1]
    prompt = sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else "nova"
    
    try:
        run_full_pipeline(video_file, prompt, voice)
    except Exception as e:
        print(f"\nAn error occurred during the pipeline: {e}")
        import traceback
        traceback.print_exc()

