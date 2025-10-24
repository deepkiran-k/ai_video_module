import sys
from pathlib import Path
import json
import os
from typing import List, Dict, Any
import moviepy.editor as mp
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ocr_extractor import OCRExtractor, extract_frames
from core.action_analyzer import ActionAnalyzer
from core.ai_manager import AIManager

def group_events_into_steps(action_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not action_log: return []
    steps = []
    start_time = 0.0
    for i, event in enumerate(action_log):
        end_time = event['timestamp']
        if end_time > start_time:
            steps.append({
                "step_number": i + 1, "start_time": start_time, "end_time": end_time,
                "text_added": event.get('text_added', []), "text_removed": event.get('text_removed', [])
            })
        start_time = end_time
    return steps

# In run_pipeline.py, REPLACE the entire run_full_pipeline function
# In run_pipeline.py, REPLACE the entire run_full_pipeline function with this:

def run_full_pipeline(video_path_str: str, user_prompt: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path_str}"); return

    video_name = video_path.stem
    
    # Setup directories
    Path("outputs").mkdir(exist_ok=True)
    source_clips_dir = Path("source_clips_for_runway") / video_name
    source_clips_dir.mkdir(parents=True, exist_ok=True)
    narrations_dir = Path("narrations") / video_name
    narrations_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting production plan for '{video_name}'...")
    print(f"💬 User Prompt: \"{user_prompt}\"")

    print("\n--- Step 1 & 2: Analyzing Video Content ---")
    ai_manager = AIManager()
    ocr_extractor = OCRExtractor()
    frame_data = extract_frames(str(video_path), interval=2)
    ocr_results = {ts: ocr_extractor.extract_text_from_frame(fp) for fp, ts in frame_data}
    analyzer = ActionAnalyzer(ocr_results)
    action_log = analyzer.analyze(str(video_path))
    steps_with_data = group_events_into_steps(action_log)
    print("✅ Video analysis complete.")

    print("\n--- Step 3: AI Processing & Plan Generation ---")
    structured_summary = ai_manager.summarize_actions(steps_with_data)
    selected_steps = ai_manager.select_relevant_segments(structured_summary, user_prompt)
    production_plan = ai_manager.create_production_plan(selected_steps, user_prompt)
    if not production_plan: print("❌ AI Director failed to create a plan. Exiting."); return
    
    narrations_list = ai_manager.generate_narrations(production_plan)
    if not narrations_list: print("❌ AI failed to create narrations. Exiting."); return

    # --- MODIFIED: Convert the list of narrations into a dictionary for easy lookup ---
    narrations_dict = {item['output_filename']: item['narration'] for item in narrations_list if 'output_filename' in item and 'narration' in item}

    print("\n--- Step 4: Saving Artifacts & Cutting Source Clips ---")
    plan_path = Path("outputs") / f"{video_name}_production_plan.json"
    with open(plan_path, "w") as f: json.dump(production_plan, f, indent=4)
    print(f"💾 Production plan saved to '{plan_path}'")
    
    with mp.VideoFileClip(video_path_str) as video:
        for shot in production_plan:
            output_filename = shot['output_filename']
            # Use the new dictionary for a reliable lookup
            narration_text = narrations_dict.get(output_filename)

            if narration_text:
                audio_path = narrations_dir / f"{Path(output_filename).stem}.mp3"
                ai_manager.create_voiceover_for_shot(narration_text, str(audio_path))

            step_num_to_find = shot.get('original_step_number')
            if step_num_to_find:
                original_step = next((s for s in selected_steps if s.get('step_number') == step_num_to_find), None)
                if original_step:
                    start = original_step.get('start_time')
                    end = original_step.get('end_time')
                    if start is not None and end is not None:
                        clip_filename = source_clips_dir / f"source_{output_filename}"
                        print(f"  -> Cutting clip '{clip_filename.name}' from {start:.2f}s to {end:.2f}s")
                        subclip = video.subclip(start, end)
                        subclip.write_videofile(str(clip_filename), codec="libx264", audio=False, logger=None)
                    else:
                        print(f"⚠️ Warning: Timestamp data missing for step #{step_num_to_find}. Skipping clip.")
                else:
                    print(f"⚠️ Warning: Could not find original step for step number {step_num_to_find}. Skipping clip.")

    print("✅ All source clips and narrations have been generated.")
    
    # (Final printout section is unchanged)
    print("\n\n" + "="*60)
    print("🎬 YOUR MANUAL PRODUCTION PLAN (VIDEO-TO-VIDEO) 🎬")
    # ...
    # (The rest of the script is the same)
    print(f"Source clips have been saved to the '{source_clips_dir}' folder.")
    print(f"Please create the final clips using RunwayML and save them in 'generated_clips/{video_name}/'")
    for shot in production_plan:
        print("\n" + "-"*20)
        print(f"  Shot {shot['shot_number']}: {shot['output_filename']}")
        print(f"  Model: {shot['model_to_use']}")
        if shot.get('model_to_use').lower() == 'gen4_aleph':
             print(f"  Source: Upload the pre-cut clip 'source_{shot['output_filename']}' from the '{source_clips_dir}' folder.")
        print(f"  RunwayML Prompt: \"{shot['prompt']}\"")
    print("\n" + "="*60)
    print(f"✅ After you have generated all clips, run the final assembly script:")
    print(f"python assemble_video.py {video_name}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py \"<path_to_video>\" \"<your_prompt>\""); sys.exit(1)
    run_full_pipeline(sys.argv[1], sys.argv[2])