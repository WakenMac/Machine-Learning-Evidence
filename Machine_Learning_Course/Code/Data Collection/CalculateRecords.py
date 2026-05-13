import cv2
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
BASE_VIDEO_PATH = Path(r"D:\Datasets\Publishing Research_Vid Recordings")
DETECTOR_CSV = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\frame_detector.csv')

def calculate_max_records():
    if not DETECTOR_CSV.exists():
        print("Error: frame_detector.csv not found. Please run your detector first.")
        return

    # Load your starting frames
    df = pd.read_csv(DETECTOR_CSV)
    
    results = []
    total_potential_dataset_size = 0

    print(f"{'Video Name':<30} | {'Max Records':<15}")
    print("-" * 50)

    for index, row in df.iterrows():
        video_name = row['file_name']
        start_frame = row['start_frame']
        video_path = BASE_VIDEO_PATH / video_name

        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            continue

        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 1. Calculate seconds
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # 2. Multiply by 60
        base_calc = duration_seconds * 60
        
        # 3. Subtract starting frame
        usable_frames = base_calc - start_frame
        
        # 4. Multiply by 5 (Maximum records)
        max_records = max(0, int(usable_frames * 5))
        
        results.append({
            'video_name': video_name,
            'max_records': max_records
        })
        
        total_potential_dataset_size += max_records
        print(f"{video_name:<30} | {max_records:<15,}")

        cap.release()

    print("-" * 50)
    print(f"TOTAL POTENTIAL RECORDS: {total_potential_dataset_size:,}")

    # Optional: Save this to a summary CSV
    # summary_df = pd.DataFrame(results)
    # summary_df.to_csv('extraction_estimate.csv', index=False)

if __name__ == "__main__":
    calculate_max_records()