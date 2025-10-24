import sys
from pathlib import Path
import json
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the classes from our core modules
from core.ocr_extractor import OCRExtractor, extract_frames
from core.action_analyzer import ActionAnalyzer, save_action_log
from core.ai_manager2 import AIManager
from core.video_editor import VideoEditor
from typing import List, Dict, Any

import sys
sys.stdout.reconfigure(encoding='utf-8')


def group_events_into_steps(action_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    CORRECTED: Groups a flat list of timestamped events into logical steps,
    each with a defined start and end time. This is crucial for video editing.
    """
    if not action_log:
        return []

    steps = []
    # The first step starts at the beginning of the video.
    start_time = 0.0
    
    for i, event in enumerate(action_log):
        end_time = event['timestamp']
        
        # A step is the time between the last event and the current one.
        # We ensure the duration is meaningful.
        if end_time > start_time:
            step_data = {
                "step_number": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "text_added": event.get('text_added', []),
                "text_removed": event.get('text_removed', [])
            }
            steps.append(step_data)
        
        # The next step starts where the current one ended.
        start_time = end_time
        
    print(f" Grouped {len(action_log)} events into {len(steps)} logical steps.")
    return steps



def run_full_pipeline(video_path_str: str, user_prompt: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"❌ Error: Video file not found at {video_path_str}")
        return

    Path("outputs").mkdir(exist_ok=True)
    video_name = video_path.stem
    print(f"🚀 Starting AI video guide generation for '{video_path.name}'...")
    print(f"💬 User Prompt: \"{user_prompt}\"")

    # --- 1. OCR Extraction ---
    ocr_results_path = Path("outputs") / f"{video_name}_ocr_results.json"
    print("\n--- Step 1: Extracting Text (OCR) ---")
    ocr_extractor = OCRExtractor()
    frame_data = extract_frames(str(video_path), interval=2)
    ocr_results = {}
    for frame_path, timestamp in frame_data:
        ocr_results[timestamp] = ocr_extractor.extract_text_from_frame(frame_path)
    ocr_extractor.save_ocr_results(ocr_results, str(ocr_results_path))
    print("✅ OCR extraction complete.")

    # --- 2. Action Analysis ---
    action_log_path = Path("outputs") / f"{video_name}_action_log.json"
    print("\n--- Step 2: Analyzing Actions ---")
    analyzer = ActionAnalyzer(ocr_results)
    action_log = analyzer.analyze(str(video_path))
    save_action_log(action_log, str(action_log_path))
    print("✅ Action analysis complete.")

    # --- 3. AI Management ---
    print("\n--- Step 3: AI Processing ---")
    ai_manager = AIManager()
    
    structured_summary = ai_manager.summarize_actions(action_log)
    if not structured_summary:
        print("❌ AI failed to generate a summary. Exiting."); return
        
    selected_steps = ai_manager.select_relevant_segments(structured_summary, user_prompt)
    if not selected_steps:
        print("❌ AI could not select any relevant steps. Exiting."); return

    script = ai_manager.generate_script(selected_steps, user_prompt)
    print("✅ AI processing complete.")
    
    # --- 4. Voiceover Generation ---
    audio_path = Path("outputs") / f"{video_name}_voiceover.mp3"
    print("\n--- Step 4: Generating Voiceover ---")
    ai_manager.create_voiceover(script, str(audio_path))
    print("✅ Voiceover generated.")
    
    # --- 5. Video Editing ---
    final_video_path = Path("outputs") / f"{video_name}_final_guide.mp4"
    print("\n--- Step 5: Editing Final Video ---")
    editor = VideoEditor()
    editor.create_guide_video(
        original_video_path=str(video_path),
        selected_steps=selected_steps,
        audio_path=str(audio_path),
        script=script,
        output_filename=final_video_path
    )

    print(f"\n🎉 Success! Your AI-generated video guide is saved as '{final_video_path}'")


# def run_full_pipeline(video_path_str: str, user_prompt: str):
#     video_path = Path(video_path_str)
#     if not video_path.exists():
#         print(f"❌ Error: Video file not found at {video_path_str}")
#         return

#     video_name = video_path.stem
#     Path("outputs").mkdir(exist_ok=True)
#     Ensure the generated_clips folder for this project exists
#     (Path("generated_clips") / video_name).mkdir(parents=True, exist_ok=True)

#     print(f" Starting AI video guide generation for '{video_path.name}'...")
#     print(f" User Prompt: \"{user_prompt}\"")

#     --- Steps 1-4 for analysis and voiceover remain the same ---
#     (Assuming the logic you have for these steps is correct)
#     print("\n--- Step 1 & 2: Analysis ---")
#     ocr_extractor = OCRExtractor()
#     frame_data = extract_frames(str(video_path), interval=2)
#     ocr_results = {ts: ocr_extractor.extract_text_from_frame(fp) for fp, ts in frame_data}
#     analyzer = ActionAnalyzer(ocr_results)
#     action_log = analyzer.analyze(str(video_path))
#     steps_with_data = group_events_into_steps(action_log)

#     print("\n--- Step 3: AI Processing ---")
#     ai_manager = AIManager()
#     structured_summary = ai_manager.summarize_actions(steps_with_data)
#     selected_steps = ai_manager.select_relevant_segments(structured_summary, user_prompt)
#     if not selected_steps:
#         print(" AI could not select any relevant steps. Exiting."); return
#     script = ai_manager.generate_script(selected_steps, user_prompt)
    
#     print("\n--- Step 4: Generating Voiceover ---")
#     audio_path = Path("outputs") / f"{video_name}_voiceover.mp3"
#     ai_manager.create_voiceover(script, str(audio_path))
    
#     --- Step 5. Video Editing ---
#     final_video_path = Path("outputs") / f"{video_name}_direct_final_video.mp4"
#     print("\n--- Step 5: Editing Final Video ---")
    
#     --- ADD THIS LOGIC to look for the intro clip ---
#     intro_path = Path("generated_clips") / video_name / "shot_01_intro.mp4"
    
#     editor = VideoEditor()
#     editor.create_guide_video(
#         original_video_path=str(video_path),
#         selected_steps=selected_steps,
#         audio_path=str(audio_path),
#         script=script,
#         output_filename=str(final_video_path),
#         --- ADD THIS ARGUMENT to pass the intro path ---
#         intro_clip_path=str(intro_path) if intro_path.exists() else None
#     )

#     print(f"\n Success! Your AI-generated video guide is saved as '{final_video_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py \"<path_to_video>\" \"<your_prompt>\"")
        sys.exit(1)
    
    video_file = sys.argv[1]
    prompt = sys.argv[2]
    
    try:
        run_full_pipeline(video_file, prompt)
    except Exception as e:
        print(f"\nAn error occurred during the pipeline: {e}")
        import traceback
        traceback.print_exc()

