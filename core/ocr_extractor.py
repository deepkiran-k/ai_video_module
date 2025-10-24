import cv2
import easyocr
from pathlib import Path
from typing import List, Dict
import json
import os
import sys
import numpy as np

class OCRExtractor:
    """
    Extracts and structures text from image frames using EasyOCR.
    EasyOCR is generally more accurate for UI elements and automatically groups text into lines.
    """

    def __init__(self, languages: List[str] = ['en']):
        """
        Initializes the EasyOCR reader. This can take a moment on first run
        as it downloads the language models.
        """
        try:
            self.reader = easyocr.Reader(languages, gpu=True) # Set gpu=False if you don't have a compatible GPU
            print("✅ EasyOCR Reader initialized successfully (using GPU).")
        except Exception:
            print("⚠️ GPU not available or PyTorch not installed with CUDA support. Falling back to CPU.")
            self.reader = easyocr.Reader(languages, gpu=False)
            print("✅ EasyOCR Reader initialized successfully (using CPU).")


    def extract_text_from_frame(self, frame_path: str) -> List[Dict]:
        """
        Extracts, filters, and structures text from a single frame image.

        Args:
            frame_path: The file path to the image frame.

        Returns:
            A list of dictionaries, where each dictionary represents a line of text
            with its content, confidence, and bounding box.
        """
        results = []
        try:
            # EasyOCR's readtext method returns a list of tuples: (bbox, text, confidence)
            ocr_results = self.reader.readtext(frame_path)

            for (bbox, text, confidence) in ocr_results:
                # 1. --- FILTERING ---
                # We apply stricter filtering to remove noise.
                # Ignore results with low confidence or very short, non-meaningful text.
                if confidence < 0.6 or len(text.strip()) <= 1:
                    continue

                # 2. --- STRUCTURING ---
                # EasyOCR provides the bounding box as a list of 4 points.
                # We convert it to a serializable format.
                top_left = [int(p) for p in bbox[0]]
                bottom_right = [int(p) for p in bbox[2]]

                results.append({
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": [top_left, bottom_right] # Simplified bbox for easier use
                })

        except Exception as e:
            print(f"⚠️ Could not process frame {frame_path} with EasyOCR: {e}")
            return []

        return results

    def save_ocr_results(self, ocr_data: Dict[float, List[Dict]], output_path: str):
        """Saves the structured OCR data to a JSON file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, indent=4, ensure_ascii=False)
        print(f"💾 Saved structured OCR results to {output_path}")


def extract_frames(video_path: str, interval: int = 5) -> list[tuple[str, float]]:
    """
    Extracts frames from a video at a specified interval.
    This function remains the same as it's working well.
    """
    video_path_obj = Path(video_path)
    temp_frames_dir = Path("temp_frames") / video_path_obj.stem
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    frames_to_extract = np.arange(0, duration, interval)
    extracted_frames = []

    for i, t in enumerate(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_path = str(temp_frames_dir / f"frame_{i:04d}_{t:.2f}s.png")
        cv2.imwrite(frame_path, frame)
        extracted_frames.append((frame_path, t))

    cap.release()
    print(f"✅ Extracted {len(extracted_frames)} frames from '{video_path_obj.name}'.")
    return extracted_frames


if __name__ == "__main__":
    # This main block allows you to run this file directly for testing.
    # Example: python core/ocr_extractor.py "path/to/your/video.mp4" --interval 2
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from a video or image.")
    parser.add_argument("input_path", help="Path to the video file or image.")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds to extract frames from video.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {args.input_path}")
        sys.exit(1)

    # Initialize the extractor
    ocr_extractor = OCRExtractor()
    
    ocr_results = {}

    if input_path.suffix.lower() in [".mp4", ".mov", ".avi"]:
        # Process video
        frame_data = extract_frames(str(input_path), args.interval)
        for frame_path, timestamp in frame_data:
            ocr_results[timestamp] = ocr_extractor.extract_text_from_frame(frame_path)
    else:
        # Process a single image
        ocr_results[0.0] = ocr_extractor.extract_text_from_frame(str(input_path))

    # Save the results
    output_filename = Path("outputs") / f"{input_path.stem}_ocr_results.json"
    ocr_extractor.save_ocr_results(ocr_results, str(output_filename))


    