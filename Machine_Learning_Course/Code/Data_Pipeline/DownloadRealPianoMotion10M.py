"""
DownloadRealPianoMotion10M.py
Downloads and parses the REAL PianoMotion10M dataset from GitHub.
Extracts 3D hand kinematics and MIDI annotations for ML training.

Dataset: https://github.com/agnJason/PianoMotion10M
Paper: https://arxiv.org/abs/2104.01266
"""

import os
import json
import zipfile
import urllib.request
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PianoMotion10MDownloader:
    """
    Downloads and manages the PianoMotion10M dataset.
    Handles extraction, parsing, and validation.
    """
    
    # GitHub URLs
    GITHUB_REPO = "https://github.com/google-research-datasets/pianomotion-10m"
    GITHUB_ZIP = "https://github.com/google-research-datasets/pianomotion-10m/archive/refs/heads/main.zip"
    
    # Expected dataset structure
    EXPECTED_STRUCTURE = {
        'data': 'Motion capture and MIDI data',
        'split': 'Train/validation/test splits',
        'README.md': 'Dataset documentation'
    }
    
    def __init__(self, output_dir: str = None, use_cache: bool = True):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save dataset (defaults to Data/PianoMotion10M)
            use_cache: Use cached data if available
        """
        if output_dir is None:
            # Updated path to reflect manual data placement by user (Code/Data_Pipeline/PianoMotion10M)
            output_dir = Path(__file__).parent / "PianoMotion10M"
        
        self.output_dir = Path(output_dir)
        self.use_cache = use_cache
        self.data_dir = self.output_dir / "data"
        self.split_dir = self.output_dir / "split"
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def download(self) -> bool:
        """
        Download PianoMotion10M dataset from GitHub.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("📥 DOWNLOADING PIANOMOTION10M DATASET")
        logger.info("="*80)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if self.use_cache and self._check_dataset_exists():
            logger.info("✅ Dataset already exists locally. Skipping download.")
            return True
        
        zip_path = self.output_dir / "PianoMotion10M.zip"
        
        try:
            logger.info(f"\n📍 Source: {self.GITHUB_ZIP}")
            logger.info(f"📁 Destination: {self.output_dir}")
            logger.info("\n⏳ Downloading... (this may take a few minutes)")
            
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 // total_size, 100)
                mb_downloaded = downloaded / (1024*1024)
                mb_total = total_size / (1024*1024)
                print(f"\r   Progress: {percent}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end="")
            
            urllib.request.urlretrieve(self.GITHUB_ZIP, zip_path, download_progress)
            print("\n✅ Download complete!\n")
            
            # Extract
            logger.info("📂 Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            
            # Move files from PianoMotion10M-main to parent
            extracted_dir = self.output_dir / "PianoMotion10M-main"
            if extracted_dir.exists():
                logger.info("🔄 Reorganizing directory structure...")
                for item in extracted_dir.iterdir():
                    dest = self.output_dir / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(dest))
                shutil.rmtree(extracted_dir)
            
            # Clean up zip
            zip_path.unlink()
            
            logger.info("✅ Extraction complete!\n")
            
            # Verify structure
            if self._verify_structure():
                logger.info("✅ Dataset structure verified!")
                return True
            else:
                logger.warning("⚠️  Dataset structure incomplete")
                return False
        
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def _check_dataset_exists(self) -> bool:
        """Check if dataset already exists."""
        return (self.data_dir.exists() and 
                len(list(self.data_dir.glob("*"))) > 0)
    
    def _verify_structure(self) -> bool:
        """Verify dataset has expected structure."""
        logger.info("\n🔍 Verifying dataset structure...")
        
        # Check for main directories
        required_dirs = ['data']
        found_all = True
        
        for req_dir in required_dirs:
            dir_path = self.output_dir / req_dir
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                logger.info(f"   ✅ {req_dir}/ ({file_count} items)")
            else:
                logger.warning(f"   ❌ {req_dir}/ (missing)")
                found_all = False
        
        return found_all
    
    def get_dataset_info(self) -> Dict:
        """Get information about downloaded dataset."""
        logger.info("\n📊 DATASET INFORMATION")
        logger.info("="*80)
        
        info = {
            'dataset_path': str(self.output_dir),
            'data_directory': str(self.data_dir),
            'date_downloaded': datetime.now().isoformat(),
            'total_subjects': 0,
            'total_sequences': 0,
            'total_files': 0,
        }
        
        # Count subjects and sequences
        if self.data_dir.exists():
            subjects = [d for d in self.data_dir.iterdir() if d.is_dir()]
            info['total_subjects'] = len(subjects)
            
            for subject_dir in subjects:
                sequences = list(subject_dir.glob("*"))
                info['total_sequences'] += len(sequences)
            
            info['total_files'] = len(list(self.data_dir.glob("**/*")))
            
            logger.info(f"Total Subjects: {info['total_subjects']}")
            logger.info(f"Total Sequences: {info['total_sequences']}")
            logger.info(f"Total Files: {info['total_files']}")
        
        return info


class PianoMotion10MParser:
    """
    Parses PianoMotion10M dataset files.
    Extracts 3D hand coordinates and MIDI annotations.
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize parser.
        
        Args:
            dataset_dir: Path to PianoMotion10M dataset root
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / "data"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Parser initialized with dataset: {self.dataset_dir}")
    
    def list_subjects(self) -> List[str]:
        """Get list of all subjects in dataset."""
        subjects = sorted([d.name for d in self.data_dir.iterdir()
                          if d.is_dir() and d.name not in ['midi']])
        logger.info(f"Found {len(subjects)} subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")
        return subjects
    
    def list_sequences(self, subject: str) -> List[str]:
        """Get list of sequences for a subject."""
        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            logger.warning(f"Subject {subject} not found")
            return []
        
        # Check for nested structure common in PianoMotion10M partitions
        if subject in ['audio-002', 'annotation-001']:
            # Assuming sequences are nested under 'audio' or similar sub-directory
            # Based on file list, audio-002 has audio/ which contains subject IDs (e.g., 688183660)
            data_root = subject_dir / 'audio'
            if data_root.exists():
                # Sequences are the directories under data_root
                sequences = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
                return sequences
            
        # Default behavior for standard subject directories
        sequences = sorted([d.name for d in subject_dir.iterdir() if d.is_dir()])
        return sequences
    
    def parse_hand_pose(self, pose_file: str) -> Optional[np.ndarray]:
        """
        Parse hand pose file (JSON or NPZ format).
        
        Args:
            pose_file: Path to pose file
            
        Returns:
            Array of shape (frames, 21, 3) with 3D hand coordinates
        """
        pose_path = Path(pose_file)
        
        try:
            if pose_path.suffix == '.json':
                with open(pose_path, 'r') as f:
                    data = json.load(f)
                
                # Parse based on structure
                if isinstance(data, dict):
                    if 'poses' in data:
                        poses = data['poses']
                    elif 'frames' in data:
                        poses = [f['pose'] for f in data['frames']]
                    else:
                        poses = list(data.values())
                else:
                    poses = data
                
                # Convert to numpy array
                poses_array = np.array(poses)
                
                # Reshape if needed (might be (frames, 63) -> (frames, 21, 3))
                if len(poses_array.shape) == 2 and poses_array.shape[1] == 63:
                    poses_array = poses_array.reshape(-1, 21, 3)
                
                logger.debug(f"Loaded pose: {poses_array.shape}")
                return poses_array
            
            elif pose_path.suffix == '.npz':
                data = np.load(pose_path)
                # Get first array in file
                first_key = list(data.files)[0]
                poses_array = data[first_key]
                
                if len(poses_array.shape) == 2 and poses_array.shape[1] == 63:
                    poses_array = poses_array.reshape(-1, 21, 3)
                
                return poses_array
            
            elif pose_path.suffix == '.npy':
                poses_array = np.load(pose_path)
                
                if len(poses_array.shape) == 2 and poses_array.shape[1] == 63:
                    poses_array = poses_array.reshape(-1, 21, 3)
                
                return poses_array
            
            else:
                logger.warning(f"Unknown file format: {pose_path.suffix}")
                return None
        
        except Exception as e:
            logger.error(f"Error parsing pose file {pose_file}: {e}")
            return None
    
    def parse_midi(self, midi_file: str) -> Dict[int, int]:
        """
        Parse MIDI file to frame-level labels.
        
        Args:
            midi_file: Path to MIDI file
            
        Returns:
            Dictionary mapping frame index to label (0=hover, 1=press)
        """
        midi_path = Path(midi_file)
        
        try:
            import mido
        except ImportError:
            logger.warning("mido not installed. Install with: pip install mido")
            return {}
        
        try:
            midi_data = mido.MidiFile(midi_file)
            frame_labels = {}
            
            ticks_per_beat = midi_data.ticks_per_beat
            tempo = 500000  # Default: 120 BPM
            fps = 30  # Standard frame rate
            
            time_accumulator = 0.0
            
            for track in midi_data.tracks:
                for msg in track:
                    time_accumulator += msg.time
                    
                    # Convert MIDI ticks to seconds
                    seconds = (time_accumulator / ticks_per_beat) * (tempo / 1_000_000)
                    frame_idx = int(seconds * fps)
                    
                    # Note on = press, Note off = release
                    if msg.type == 'note_on' and msg.velocity > 0:
                        frame_labels[frame_idx] = 1
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        frame_labels[frame_idx] = 0
            
            logger.debug(f"Loaded MIDI: {len(frame_labels)} events")
            return frame_labels
        
        except Exception as e:
            logger.error(f"Error parsing MIDI file {midi_file}: {e}")
            return {}
    
    def parse_annotations(self, annotation_file: str) -> Dict:
        """
        Parse annotation file (JSON format).
        
        Args:
            annotation_file: Path to annotation JSON file
            
        Returns:
            Dictionary with parsed annotations
        """
        annotation_path = Path(annotation_file)
        
        try:
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            logger.debug(f"Loaded annotations: {len(annotations)} entries")
            return annotations
        
        except Exception as e:
            logger.error(f"Error parsing annotations {annotation_file}: {e}")
            return {}
    
    def extract_sequence_data(self, subject: str, sequence: str) -> Optional[Dict]:
        """
        Extract all data for a subject-sequence pair.
        
        Args:
            subject: Subject ID (e.g., 'subject_001')
            sequence: Sequence ID (e.g., 'seq_001')
            
        Returns:
            Dictionary with pose, MIDI, and metadata
        """
        if subject in ['audio-002', 'annotation-001']:
            # Handle nested structure: data_dir / partition / audio / sequence
            sequence_dir = self.data_dir / subject / 'audio' / sequence
        else:
            # Default structure: data_dir / subject / sequence
            sequence_dir = self.data_dir / subject / sequence
        
        if not sequence_dir.exists():
            # Check if the sequence is nested one level deeper (e.g., under the subject ID itself)
            sequence_dir_nested = self.data_dir / subject / sequence
            if sequence_dir_nested.exists():
                sequence_dir = sequence_dir_nested
            else:
                logger.warning(f"Sequence not found: {subject}/{sequence}")
                return None
        
        logger.info(f"\n📂 Extracting: {subject}/{sequence}")
        
        # Find files
        pose_files = list(sequence_dir.glob("*pose*")) + list(sequence_dir.glob("*hand*"))
        
        # Search for MIDI files in the dedicated MIDI directory structure
        midi_dir = self.data_dir / 'midi' / 'midi' / sequence
        midi_files = list(midi_dir.glob("*.mid"))
        
        # Fallback to sequence directory search if needed (original logic)
        if not midi_files:
            midi_files = list(sequence_dir.glob("*.mid")) + list(sequence_dir.glob("*midi*"))
        annotation_files = list(sequence_dir.glob("*annotation*"))
        
        result = {
            'subject': subject,
            'sequence': sequence,
            'directory': str(sequence_dir),
            'pose_data': None,
            'midi_labels': None,
            'annotations': None,
        }
        
        # Parse pose
        if pose_files:
            pose_file = pose_files[0]
            logger.info(f"   Loading pose from: {pose_file.name}")
            result['pose_data'] = self.parse_hand_pose(str(pose_file))
            
            if result['pose_data'] is not None:
                logger.info(f"   ✅ Pose shape: {result['pose_data'].shape}")
        
        # Parse MIDI
        if midi_files:
            midi_file = midi_files[0]
            logger.info(f"   Loading MIDI from: {midi_file.name}")
            result['midi_labels'] = self.parse_midi(str(midi_file))
            logger.info(f"   ✅ MIDI events: {len(result['midi_labels'])}")
        
        # Parse annotations
        if annotation_files:
            annotation_file = annotation_files[0]
            logger.info(f"   Loading annotations from: {annotation_file.name}")
            result['annotations'] = self.parse_annotations(str(annotation_file))
        
        return result
    
    def extract_all_sequences(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Extract data from all sequences in dataset.
        
        Args:
            limit: Maximum number of sequences to extract (None = all)
            
        Returns:
            List of extracted sequence data
        """
        logger.info("\n" + "="*80)
        logger.info("📊 EXTRACTING ALL SEQUENCES")
        logger.info("="*80)
        
        subjects = self.list_subjects()
        all_data = []
        count = 0
        
        for subject in subjects:
            sequences = self.list_sequences(subject)
            logger.info(f"\n{subject}: {len(sequences)} sequences")
            
            for sequence in sequences:
                if limit and count >= limit:
                    logger.info(f"Reached limit of {limit} sequences")
                    return all_data
                
                data = self.extract_sequence_data(subject, sequence)
                if data and data['pose_data'] is not None:
                    all_data.append(data)
                    count += 1
        
        logger.info(f"\n✅ Extracted {count} sequences")
        return all_data


class PianoMotion10MFeatureExtractor:
    """
    Extracts features from parsed PianoMotion10M data.
    Aligns pose and MIDI to create training dataset.
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize feature extractor.
        
        Args:
            fps: Frames per second
        """
        self.fps = fps
        self.frame_duration = 1.0 / fps
    
    def extract_features_from_sequence(self, sequence_data: Dict) -> pd.DataFrame:
        """
        Extract ML features from a single sequence.
        
        Args:
            sequence_data: Parsed sequence data
            
        Returns:
            DataFrame with extracted features
        """
        pose_data = sequence_data['pose_data']
        midi_labels = sequence_data['midi_labels']
        
        if pose_data is None:
            logger.warning(f"No pose data for {sequence_data['sequence']}")
            return pd.DataFrame()
        
        features_list = []
        n_frames = len(pose_data)
        
        logger.info(f"   Extracting features from {n_frames} frames...")
        
        for frame_idx in range(n_frames):
            frame_pose = pose_data[frame_idx]  # (21, 3)
            
            # Extract features (same as before)
            frame_features = self._extract_frame_features(frame_pose, frame_idx, pose_data)
            
            # Add label from MIDI
            if midi_labels:
                frame_features['ground_truth_label'] = midi_labels.get(frame_idx, 0)
            else:
                frame_features['ground_truth_label'] = 0
            
            frame_features['subject'] = sequence_data['subject']
            frame_features['sequence'] = sequence_data['sequence']
            frame_features['frame_index'] = frame_idx
            
            features_list.append(frame_features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"   ✅ Extracted {len(df)} frames")
        
        return df
    
    def _extract_frame_features(self, frame_pose: np.ndarray, frame_idx: int, 
                               all_poses: np.ndarray) -> Dict:
        """Extract features from single frame."""
        features = {}
        
        try:
            # Joint indices
            fingertip_idx = 8      # Index finger tip
            dip_idx = 6            # Index finger DIP
            wrist_idx = 0          # Wrist
            
            fingertip = frame_pose[fingertip_idx]
            dip_joint = frame_pose[dip_idx]
            wrist = frame_pose[wrist_idx]
            
            # Static features
            features['finger_position_x'] = float(fingertip[0])
            features['finger_position_y'] = float(fingertip[1])
            features['finger_position_z'] = float(fingertip[2])
            features['depth_feature'] = float(fingertip[2])
            features['posture_feature'] = float(np.linalg.norm(fingertip - dip_joint))
            features['euclidean_distance'] = float(np.linalg.norm(fingertip - dip_joint))
            features['distance_from_wrist'] = float(np.linalg.norm(fingertip - wrist))
            
            # Dynamic features
            if frame_idx > 0:
                prev_pose = all_poses[frame_idx - 1]
                prev_fingertip = prev_pose[fingertip_idx]
                displacement = np.linalg.norm(fingertip - prev_fingertip)
                features['finger_velocity'] = float(displacement / self.frame_duration)
            else:
                features['finger_velocity'] = 0.0
            
            if frame_idx > 1:
                prev_pose = all_poses[frame_idx - 1]
                prev_prev_pose = all_poses[frame_idx - 2]
                vel_current = np.linalg.norm(frame_pose[fingertip_idx] - prev_pose[fingertip_idx]) / self.frame_duration
                vel_prev = np.linalg.norm(prev_pose[fingertip_idx] - prev_prev_pose[fingertip_idx]) / self.frame_duration
                features['finger_acceleration'] = float((vel_current - vel_prev) / self.frame_duration)
            else:
                features['finger_acceleration'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            # Return zeros
            features = {
                'finger_position_x': 0.0, 'finger_position_y': 0.0, 'finger_position_z': 0.0,
                'depth_feature': 0.0, 'posture_feature': 0.0, 'euclidean_distance': 0.0,
                'distance_from_wrist': 0.0, 'finger_velocity': 0.0, 'finger_acceleration': 0.0,
            }
        
        return features


def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(description="Download and parse PianoMotion10M dataset.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory for the dataset.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the download step and proceed directly to parsing/feature extraction.")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎹 REAL PIANOMOTION10M DATASET - DOWNLOAD & PARSE")
    print("="*80 + "\n")
    
    # Define paths
    if args.output_dir:
        dataset_dir = Path(args.output_dir)
    else:
        dataset_dir = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M"
    
    # Step 1: Download (Skipped if --skip-download is used)
    if not args.skip_download:
        print("STEP 1: Download Dataset")
        print("-" * 80)
        downloader = PianoMotion10MDownloader(str(dataset_dir))
        
        if not downloader.download():
            print("Failed to download dataset. Please check your internet connection.")
            return
    else:
        print("STEP 1: Download Skipped (Manual download assumed)")
        print("-" * 80)
        # Ensure the directory exists for subsequent steps
        dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Get info (Only if download was attempted or skipped)
    print("\nSTEP 2: Dataset Information")
    print("-" * 80)
    downloader = PianoMotion10MDownloader(str(dataset_dir), use_cache=True)
    info = downloader.get_dataset_info()
    
    # Step 3: Parse
    print("\nSTEP 3: Parse Dataset")
    print("-" * 80)
    try:
        parser = PianoMotion10MParser(str(dataset_dir))
    except FileNotFoundError as e:
        logger.error(f"Cannot proceed with parsing: {e}")
        return

    subjects = parser.list_subjects()
    if not subjects:
        logger.error("No subjects found in the data directory. Check your manual extraction.")
        return
        
    print(f"\n✅ Found {len(subjects)} subjects")
    
    # Extract all sequences
    print("\n📂 Extracting all sequences...")
    extracted_data = parser.extract_all_sequences() # Removed limit=5 for full feature extraction
    
    if not extracted_data:
        logger.error("Failed to extract any sequence data.")
        return
        
    # Step 4: Extract features
    print("\nSTEP 4: Extract Features")
    print("-" * 80)
    feature_extractor = PianoMotion10MFeatureExtractor()
    
    all_features = []
    for seq_data in extracted_data:
        df = feature_extractor.extract_features_from_sequence(seq_data)
        if not df.empty:
            all_features.append(df)
    
    # Combine all features
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Save to CSV
        output_csv = dataset_dir / "features_real_pianomotion10m.csv"
        combined_df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Saved features to: {output_csv}")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Total features: {len(combined_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(combined_df.head())
        print(f"\nLabel distribution:")
        print(combined_df['ground_truth_label'].value_counts())
    
    print("\n" + "="*80)
    print("✅ DATASET PARSING AND FEATURE EXTRACTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
