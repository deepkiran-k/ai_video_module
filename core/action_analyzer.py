'''import json
from pathlib import Path
from typing import Dict, List, Any

# PySceneDetect is a specialized library for this task.
# Install it with: pip install pyscenedetect[opencv]
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class ActionAnalyzer:
    """
    Analyzes a video to identify key user actions by combining visual scene
    changes with changes in OCR text.
    """

    def __init__(self, ocr_data: Dict[str, List[Dict[str, Any]]]):
        """
        Initializes the analyzer with pre-extracted OCR data.

        Args:
            ocr_data: A dictionary where keys are timestamps (as strings) and
                      values are the list of OCR results for that frame.
        """
        # Convert string timestamps from JSON keys to float for easier comparison
        self.ocr_data = {float(k): v for k, v in ocr_data.items()}
        self.ocr_timestamps = sorted(self.ocr_data.keys())

    def _find_visual_scenes(self, video_path: str, threshold: float = 27.0) -> List[float]:
        """
        Uses PySceneDetect to find major visual changes (scene cuts).

        Args:
            video_path: Path to the video file.
            threshold: Sensitivity for the ContentDetector. Lower is more sensitive.

        Returns:
            A list of timestamps (in seconds) where scene changes occur.
        """
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            
            # We only need the start time of each new scene
            change_timestamps = [scene[0].get_seconds() for scene in scene_list if scene[0].get_seconds() > 0]
            print(f"✅ Found {len(change_timestamps)} major visual changes.")
            return change_timestamps
        except Exception as e:
            print(f"⚠️ Could not process video with PySceneDetect: {e}")
            return []

    def _get_text_at_timestamp(self, timestamp: float) -> List[str]:
        """
        Finds the set of OCR'd text for the frame closest to a given timestamp.
        """
        if not self.ocr_timestamps:
            return []
        
        # Find the OCR timestamp that is closest to the requested timestamp
        closest_ts = min(self.ocr_timestamps, key=lambda t: abs(t - timestamp))
        
        return [item['text'] for item in self.ocr_data.get(closest_ts, [])]

    def _diff_text_states(self, before_text: List[str], after_text: List[str]) -> Dict[str, List[str]]:
        """
        Compares two sets of text to find what was added or removed.
        """
        before_set = set(before_text)
        after_set = set(after_text)
        
        added = sorted(list(after_set - before_set))
        removed = sorted(list(before_set - after_set))
        
        return {"added": added, "removed": removed}

    def analyze(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes the video and produces a structured log of events.

        Returns:
            A list of event dictionaries, where each event has a timestamp,
            type, and data about what changed.
        """
        visual_changes = self._find_visual_scenes(video_path)
        
        # Combine OCR timestamps and visual change timestamps to create a master timeline of "moments"
        all_event_timestamps = sorted(list(set(self.ocr_timestamps + visual_changes)))
        
        event_log = []
        last_text_state = []

        for i, ts in enumerate(all_event_timestamps):
            current_text_state = self._get_text_at_timestamp(ts)
            text_diff = self._diff_text_states(last_text_state, current_text_state)
            
            # An event is significant if there was a visual cut OR if the text changed.
            is_visual_change = any(abs(ts - v_ts) < 0.1 for v_ts in visual_changes) # Check if ts is close to a visual change
            has_text_change = text_diff['added'] or text_diff['removed']

            if i > 0 and (is_visual_change or has_text_change):
                event = {
                    "timestamp": ts,
                    "type": "Visual & Text Change" if is_visual_change and has_text_change else "Visual Change" if is_visual_change else "Text Change",
                    "text_added": text_diff['added'],
                    "text_removed": text_diff['removed']
                }
                event_log.append(event)
            
            last_text_state = current_text_state
            
        print(f"✅ Generated an action log with {len(event_log)} events.")
        return event_log

def save_action_log(action_log: List[Dict], output_path: str):
    """Saves the action log to a JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(action_log, f, indent=4)
    print(f"💾 Saved action log to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a video for actions using OCR data.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("ocr_json_path", help="Path to the pre-generated ocr_results.json file.")
    args = parser.parse_args()

    # 1. Load the OCR data from the file generated by ocr_extractor.py
    try:
        with open(args.ocr_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: OCR JSON file not found at {args.ocr_json_path}")
        exit()

    # 2. Initialize and run the analyzer
    analyzer = ActionAnalyzer(ocr_data)
    action_log = analyzer.analyze(args.video_path)

    # 3. Save the results
    video_name = Path(args.video_path).stem
    output_path = Path("outputs") / f"{video_name}_action_log.json"
    save_action_log(action_log, str(output_path))

    # Optional: Print a readable version to the console
    print("\n--- Action Log Summary ---")
    for event in action_log:
        print(f"Timestamp: {event['timestamp']:.2f}s ({event['type']})")
        if event['text_added']:
            print(f"  ➡️ Added: {event['text_added']}")
        if event['text_removed']:
            print(f"  ⬅️ Removed: {event['text_removed']}")
'''

import json
from pathlib import Path
from typing import Dict, List, Any
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class ActionAnalyzer:
    """
    Analyzes a video to identify key user actions by combining visual scene
    changes with changes in OCR text.
    """
    def __init__(self, ocr_data: Dict[str, List[Dict[str, Any]]]):
        self.ocr_data = {float(k): v for k, v in ocr_data.items()}
        self.ocr_timestamps = sorted(self.ocr_data.keys())

    def _find_visual_scenes(self, video_path: str, threshold: float = 27.0) -> List[float]:
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            change_timestamps = [scene[0].get_seconds() for scene in scene_list if scene[0].get_seconds() > 0]
            print(f"✅ Found {len(change_timestamps)} major visual changes.")
            return change_timestamps
        except Exception as e:
            print(f"⚠️ Could not process video with PySceneDetect: {e}")
            return []

    def _get_text_at_timestamp(self, timestamp: float) -> List[str]:
        if not self.ocr_timestamps:
            return []
        closest_ts = min(self.ocr_timestamps, key=lambda t: abs(t - timestamp))
        return [item['text'] for item in self.ocr_data.get(closest_ts, [])]

    def _diff_text_states(self, before_text: List[str], after_text: List[str]) -> Dict[str, List[str]]:
        before_set, after_set = set(before_text), set(after_text)
        return {"added": sorted(list(after_set - before_set)), "removed": sorted(list(before_set - after_set))}

    def analyze(self, video_path: str) -> List[Dict[str, Any]]:
        visual_changes = self._find_visual_scenes(video_path)
        all_event_timestamps = sorted(list(set(self.ocr_timestamps + visual_changes)))
        
        event_log = []
        last_text_state = []

        for i, ts in enumerate(all_event_timestamps):
            current_text_state = self._get_text_at_timestamp(ts)
            text_diff = self._diff_text_states(last_text_state, current_text_state)
            
            is_visual_change = any(abs(ts - v_ts) < 0.1 for v_ts in visual_changes)
            has_text_change = text_diff['added'] or text_diff['removed']

            if i > 0 and (is_visual_change or has_text_change):
                event = {
                    "timestamp": ts,
                    "type": "Visual & Text Change" if is_visual_change and has_text_change else "Visual Change" if is_visual_change else "Text Change",
                    "text_added": text_diff['added'],
                    "text_removed": text_diff['removed']
                }
                event_log.append(event)
            
            last_text_state = current_text_state
            
        print(f"✅ Generated an action log with {len(event_log)} events.")
        return event_log

def save_action_log(action_log: List[Dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(action_log, f, indent=4)
    print(f"💾 Saved action log to {output_path}")