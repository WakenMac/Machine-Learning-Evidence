"""
PreparePianoMotionDataset.py
Aligns 3D kinematics data with MIDI keypress data and extracts features.
Prepares data for SVM and Random Forest classification models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PianoMotionDataProcessor:
    """
    Processes PianoMotion10M dataset:
    - Aligns 3D kinematics with MIDI ground truth
    - Extracts motion features
    - Generates labeled dataset for ML training
    """
    
    def __init__(self, dataset_dir: str, fps: float = 30.0):
        """
        Initialize the data processor.
        
        Args:
            dataset_dir: Path to PianoMotion10M dataset directory
            fps: Frames per second of the motion capture data
        """
        self.dataset_dir = Path(dataset_dir)
        self.fps = fps
        self.frame_duration = 1.0 / fps  # Duration of each frame in seconds
        self.features_list = []
        
    def load_annotations(self, annotation_file: str) -> np.ndarray:
        """
        Load 3D hand kinematics from annotation file.
        Expected format: JSON or CSV with 3D coordinates for hand joints.
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            Array of shape (frames, joints, 3) with 3D coordinates
        """
        logger.info(f"Loading annotations from {annotation_file}")
        
        file_path = Path(annotation_file)
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Convert to numpy array - adjust based on actual structure
                if isinstance(data, dict) and 'frames' in data:
                    frames = data['frames']
                    return np.array([frame['joints'] if 'joints' in frame else frame for frame in frames])
                else:
                    return np.array(data)
            
            elif file_path.suffix in ['.csv', '.txt']:
                data = np.loadtxt(file_path, delimiter=',')
                # Reshape assuming format: (frames, features)
                # For 21 hand joints with x, y, z: (frames, 21*3)
                num_joints = 21  # Standard hand pose (MediaPipe/MANO)
                if data.shape[1] == num_joints * 3:
                    return data.reshape(-1, num_joints, 3)
                return data
            
            else:
                logger.warning(f"Unknown file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return None
    
    def load_midi_labels(self, midi_file: str) -> Dict[float, int]:
        """
        Load MIDI keypress data and convert to frame-level labels.
        Label: 1 = key press, 0 = hover/no press
        
        Args:
            midi_file: Path to MIDI file or MIDI annotation file
            
        Returns:
            Dictionary mapping frame index to label (0 or 1)
        """
        logger.info(f"Loading MIDI labels from {midi_file}")
        
        try:
            # Try using mido library if available
            try:
                import mido
                midi_data = mido.MidiFile(midi_file)
                frame_labels = self._process_midi_with_mido(midi_data)
                return frame_labels
            except ImportError:
                logger.warning("mido not available, attempting alternative parsing")
                # Fallback: parse JSON or CSV with MIDI information
                return self._parse_midi_annotation(midi_file)
        
        except Exception as e:
            logger.error(f"Error loading MIDI labels: {e}")
            return {}
    
    def _process_midi_with_mido(self, midi_data) -> Dict[int, int]:
        """Process MIDI data using mido library."""
        frame_labels = {}
        current_frame = 0
        time_accumulator = 0.0
        
        # Process MIDI events
        for track in midi_data.tracks:
            time_accumulator = 0.0
            for msg in track:
                time_accumulator += msg.time
                frame_idx = int(time_accumulator * self.fps)
                
                # Note-on events indicate key presses
                if msg.type == 'note_on' and msg.velocity > 0:
                    frame_labels[frame_idx] = 1  # Key press
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    frame_labels[frame_idx] = 0  # Release
        
        return frame_labels
    
    def _parse_midi_annotation(self, annotation_file: str) -> Dict[int, int]:
        """Fallback: parse MIDI annotation from JSON/CSV."""
        frame_labels = {}
        
        try:
            if annotation_file.endswith('.json'):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                # Assume structure like {'frames': [{'midi': 1}, ...]}
                for idx, frame_data in enumerate(data.get('frames', [])):
                    label = frame_data.get('midi', 0)
                    frame_labels[idx] = 1 if label > 0 else 0
            
            elif annotation_file.endswith('.csv'):
                df = pd.read_csv(annotation_file)
                if 'midi' in df.columns:
                    for idx, row in df.iterrows():
                        frame_labels[idx] = 1 if row['midi'] > 0 else 0
        
        except Exception as e:
            logger.error(f"Error parsing MIDI annotation: {e}")
        
        return frame_labels
    
    def extract_features(self, kinematics: np.ndarray, frame_idx: int) -> Dict[str, float]:
        """
        Extract motion features from 3D hand kinematics for a single frame.
        
        Args:
            kinematics: Array of shape (frames, joints, 3) with 3D coordinates
            frame_idx: Current frame index
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if kinematics is None or frame_idx >= len(kinematics):
            return features
        
        current_frame = kinematics[frame_idx]  # Shape: (joints, 3)
        
        # Index fingertip and DIP joint (standard MediaPipe/MANO indices)
        # Adjust these based on actual hand model structure
        fingertip_idx = 8    # Index finger tip
        dip_idx = 6          # Index finger DIP joint
        wrist_idx = 0        # Wrist/palm base
        
        try:
            if len(current_frame) > max(fingertip_idx, dip_idx):
                fingertip = current_frame[fingertip_idx]
                dip_joint = current_frame[dip_idx]
                wrist = current_frame[wrist_idx]
                
                # 1. Finger Position (x, y, z coordinates of fingertip)
                features['finger_position_x'] = float(fingertip[0])
                features['finger_position_y'] = float(fingertip[1])
                features['finger_position_z'] = float(fingertip[2])
                
                # 4. Depth Feature (Z-coordinate) - Critical for press/hover distinction
                features['depth_feature'] = float(fingertip[2])
                
                # 5. Posture Feature (distance between fingertip and DIP joint)
                posture = np.linalg.norm(fingertip - dip_joint)
                features['posture_feature'] = float(posture)
                
                # 6. Euclidean Distance (fingertip to DIP joint - captures finger curvature)
                features['euclidean_distance'] = float(posture)
                
                # Distance from wrist for context
                distance_from_wrist = np.linalg.norm(fingertip - wrist)
                features['distance_from_wrist'] = float(distance_from_wrist)
                
                # 1. Finger Velocity (if previous frame exists)
                if frame_idx > 0:
                    prev_frame = kinematics[frame_idx - 1]
                    prev_fingertip = prev_frame[fingertip_idx]
                    velocity = np.linalg.norm(fingertip - prev_fingertip) / self.frame_duration
                    features['finger_velocity'] = float(velocity)
                else:
                    features['finger_velocity'] = 0.0
                
                # 2. Finger Acceleration (if two previous frames exist)
                if frame_idx > 1:
                    prev_frame = kinematics[frame_idx - 1]
                    prev_prev_frame = kinematics[frame_idx - 2]
                    
                    prev_fingertip = prev_frame[fingertip_idx]
                    prev_prev_fingertip = prev_prev_frame[fingertip_idx]
                    
                    velocity_current = np.linalg.norm(fingertip - prev_fingertip) / self.frame_duration
                    velocity_prev = np.linalg.norm(prev_fingertip - prev_prev_fingertip) / self.frame_duration
                    
                    acceleration = (velocity_current - velocity_prev) / self.frame_duration
                    features['finger_acceleration'] = float(acceleration)
                else:
                    features['finger_acceleration'] = 0.0
                
        except (IndexError, ValueError) as e:
            logger.warning(f"Error extracting features for frame {frame_idx}: {e}")
        
        return features
    
    def align_and_extract_features(self, kinematics_file: str, midi_file: str) -> pd.DataFrame:
        """
        Align 3D kinematics with MIDI labels and extract all features.
        
        Args:
            kinematics_file: Path to 3D hand kinematics file
            midi_file: Path to MIDI annotation file
            
        Returns:
            DataFrame with extracted features and labels
        """
        logger.info("Starting alignment and feature extraction...")
        
        # Load data
        kinematics = self.load_annotations(kinematics_file)
        midi_labels = self.load_midi_labels(midi_file)
        
        if kinematics is None:
            logger.error("Failed to load kinematics data")
            return pd.DataFrame()
        
        # Extract features for each frame
        all_features = []
        
        for frame_idx in range(len(kinematics)):
            frame_features = self.extract_features(kinematics, frame_idx)
            
            # Add label
            frame_features['ground_truth_label'] = midi_labels.get(frame_idx, 0)
            frame_features['frame_index'] = frame_idx
            
            all_features.append(frame_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        logger.info(f"Extracted {len(df)} frames with {len(df.columns) - 2} features")
        logger.info(f"Label distribution: {df['ground_truth_label'].value_counts().to_dict()}")
        
        return df
    
    def process_dataset(self, output_csv: str = "features.csv") -> str:
        """
        Process entire dataset and save to CSV.
        
        Args:
            output_csv: Path to save combined features CSV
            
        Returns:
            Path to saved CSV file
        """
        logger.info(f"Processing dataset from {self.dataset_dir}")
        
        all_data = []
        
        # Find all kinematics and MIDI pairs
        data_dirs = list(self.dataset_dir.glob("*/"))
        
        if not data_dirs:
            logger.warning(f"No subdirectories found in {self.dataset_dir}")
            logger.info("Creating sample dataset structure...")
            return self._create_sample_dataset(output_csv)
        
        for data_dir in data_dirs:
            # Look for kinematics and MIDI files
            kinematics_files = list(data_dir.glob("*annotation*.json")) + list(data_dir.glob("*annotation*.csv"))
            midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*midi*.json"))
            
            if kinematics_files and midi_files:
                for kit_file, midi_file in zip(kinematics_files, midi_files):
                    logger.info(f"Processing: {kit_file.name} + {midi_file.name}")
                    df = self.align_and_extract_features(str(kit_file), str(midi_file))
                    if not df.empty:
                        all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
        else:
            logger.warning("No valid dataset pairs found")
            combined_df = self._create_sample_dataset_df()
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Dataset saved to {output_path}")
        logger.info(f"   Total samples: {len(combined_df)}")
        logger.info(f"   Features: {list(combined_df.columns)}")
        
        return str(output_path)
    
    def _create_sample_dataset(self, output_csv: str) -> str:
        """Create sample synthetic dataset for testing/demonstration."""
        logger.info("Creating synthetic sample dataset...")
        
        df = self._create_sample_dataset_df()
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Sample dataset created: {output_path}")
        return str(output_path)
    
    def _create_sample_dataset_df(self) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        np.random.seed(42)
        
        n_samples = 500
        data = {
            'finger_velocity': np.random.uniform(0, 2, n_samples),
            'finger_acceleration': np.random.uniform(-1, 1, n_samples),
            'finger_position_x': np.random.uniform(-0.1, 0.1, n_samples),
            'finger_position_y': np.random.uniform(-0.1, 0.1, n_samples),
            'finger_position_z': np.random.uniform(0.01, 0.3, n_samples),
            'depth_feature': np.random.uniform(0.01, 0.3, n_samples),
            'posture_feature': np.random.uniform(0.01, 0.1, n_samples),
            'euclidean_distance': np.random.uniform(0.01, 0.1, n_samples),
            'distance_from_wrist': np.random.uniform(0.1, 0.5, n_samples),
        }
        
        # Create biased labels based on depth and velocity
        # Press: higher velocity, lower depth
        labels = []
        for i in range(n_samples):
            score = (1 - data['depth_feature'][i] / 0.3) * 0.5 + (data['finger_velocity'][i] / 2) * 0.5
            labels.append(1 if score > 0.5 else 0)
        
        data['ground_truth_label'] = labels
        data['frame_index'] = range(n_samples)
        
        return pd.DataFrame(data)


def main():
    """Example usage of the data processor."""
    
    # Define paths
    dataset_dir = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M"
    output_csv = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M" / "features.csv"
    
    # Process dataset
    processor = PianoMotionDataProcessor(str(dataset_dir), fps=30.0)
    csv_path = processor.process_dataset(str(output_csv))
    
    # Load and display sample
    df = pd.read_csv(csv_path)
    print(f"\n📊 Dataset Summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 5 rows:")
    print(df.head())
    print(f"\n   Label distribution:")
    print(df['ground_truth_label'].value_counts())


if __name__ == "__main__":
    main()
