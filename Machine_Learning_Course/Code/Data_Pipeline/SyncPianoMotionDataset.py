"""
SyncPianoMotionDataset.py
Synchronizes 3D hand kinematics from JSON files with MIDI keypress data.
Extracts features for all five fingers of the right hand and labels key presses
based on a heuristic that identifies the finger with the highest downward velocity
at the time of a MIDI note_on event.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
import mido

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyncPianoMotionDataset:
    """
    Processes the PianoMotion10M dataset to create a synchronized feature set for
    machine learning.
    """
    def __init__(self, dataset_dir: str, fps: float = 30.0):
        """
        Initializes the data processor.

        Args:
            dataset_dir: Path to the PianoMotion10M dataset directory.
            fps: Frames per second of the motion capture data.
        """
        self.dataset_dir = Path(dataset_dir)
        self.fps = fps
        self.frame_duration = 1.0 / fps

    def load_midi_labels(self, midi_file: Path) -> dict:
        """
        Loads a MIDI file and extracts note press events with their start and end frames.

        Args:
            midi_file: Path to the MIDI file.

        Returns:
            A dictionary where keys are note numbers and values are lists of
            (start_frame, end_frame) tuples.
        """
        note_events = {}
        try:
            midi = mido.MidiFile(str(midi_file))
            ticks_per_beat = midi.ticks_per_beat or 480
            tempo = 500000  # Default tempo (120 BPM)

            # Find the first tempo change event
            for msg in mido.merge_tracks(midi.tracks):
                if msg.is_meta and msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break

            time_in_seconds = 0.0
            open_notes = {}

            for msg in mido.merge_tracks(midi.tracks):
                delta_ticks = msg.time
                time_in_seconds += mido.tick2second(delta_ticks, ticks_per_beat, tempo)

                if msg.type == 'note_on' and msg.velocity > 0:
                    open_notes[msg.note] = time_in_seconds
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in open_notes:
                        start_time = open_notes.pop(msg.note)
                        end_time = time_in_seconds
                        start_frame = int(start_time * self.fps)
                        end_frame = int(end_time * self.fps)

                        if msg.note not in note_events:
                            note_events[msg.note] = []
                        note_events[msg.note].append((start_frame, end_frame))

        except Exception as e:
            logger.error(f"Could not process MIDI file {midi_file}: {e}")
        return note_events

    def load_kinematics(self, kinematics_file: Path) -> np.ndarray:
        """
        Loads 3D hand kinematics from a JSON annotation file.

        Args:
            kinematics_file: Path to the kinematics JSON file.

        Returns:
            A numpy array of shape (frames, 21, 3) representing hand joint coordinates.
        """
        try:
            with kinematics_file.open('r') as f:
                data = json.load(f)

            # Accommodate both 'right' and 'left' hand data
            hand_data = data.get('right') or data.get('left')
            if not hand_data:
                return None

            # Process frames, padding if necessary
            processed_frames = []
            for frame_data in hand_data:
                if len(frame_data) == 62:
                    frame_data.append(0)  # Pad to 63 for 21 joints

                if len(frame_data) == 63:
                    processed_frames.append(np.array(frame_data).reshape(21, 3))
                else:
                    # Handle empty or malformed frames
                    processed_frames.append(np.zeros((21, 3)))

            return np.array(processed_frames)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading or parsing kinematics file {kinematics_file}: {e}")
            return None

    def extract_and_label_features(self, kinematics: np.ndarray, note_events: dict) -> list:
        """
        Extracts features for all fingers and labels them based on MIDI events.

        Args:
            kinematics: A numpy array of hand kinematics.
            note_events: A dictionary of note events from the MIDI file.

        Returns:
            A list of dictionaries, where each dictionary represents a frame-finger's features.
        """
        all_features = []
        num_frames = kinematics.shape[0]
        labels = np.zeros((num_frames, 5))  # Labels for 5 fingers

        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

        # Assign labels based on the heuristic
        for note, events in note_events.items():
            for start_frame, end_frame in events:
                if start_frame > 0 and start_frame < num_frames:
                    velocities = []
                    for finger_idx in fingertip_indices:
                        pos_current = kinematics[start_frame, finger_idx]
                        pos_prev = kinematics[start_frame - 1, finger_idx]
                        velocity = (pos_current - pos_prev) / self.frame_duration
                        velocities.append(velocity[2])  # Z-velocity

                    pressing_finger_index = np.argmin(velocities)
                    labels[start_frame:end_frame, pressing_finger_index] = 1

        # Extract features for each frame and finger
        for frame_idx in range(1, num_frames):
            for i, finger_tip_idx in enumerate(fingertip_indices):
                features = {'frame': frame_idx, 'finger': i}

                current_pos = kinematics[frame_idx, finger_tip_idx]
                prev_pos = kinematics[frame_idx - 1, finger_tip_idx]

                velocity = (current_pos - prev_pos) / self.frame_duration

                features['pos_x'], features['pos_y'], features['pos_z'] = current_pos
                features['vel_x'], features['vel_y'], features['vel_z'] = velocity

                features['ground_truth_label'] = labels[frame_idx, i]
                all_features.append(features)

        return all_features

    def run(self, max_files=None, output_csv="features.csv"):
        """
        Main entry point to run the dataset generation.

        Args:
            max_files: The maximum number of file pairs to process.
            output_csv: The name of the output CSV file.
        """
        logger.info("Starting dataset generation...")

        annotation_dir = self.dataset_dir / "annotation" / "annotation"
        midi_dir = self.dataset_dir / "midi" / "midi"
        all_features = []

        # Create a list of all kinematics files
        kinematics_files = list(annotation_dir.glob("**/*.json"))
        if max_files:
            kinematics_files = kinematics_files[:max_files]

        for kinematics_file in tqdm(kinematics_files, desc="Processing files"):
            subject_id = kinematics_file.parent.name
            sequence_name = kinematics_file.stem.split('_seq_')[0]

            midi_file = midi_dir / subject_id / f"{sequence_name}.mid"

            if midi_file.exists():
                kinematics = self.load_kinematics(kinematics_file)
                note_events = self.load_midi_labels(midi_file)

                if kinematics is not None and note_events:
                    features = self.extract_and_label_features(kinematics, note_events)
                    all_features.extend(features)
            else:
                logger.warning(f"MIDI file not found for {kinematics_file}")

        if not all_features:
            logger.warning("No features were extracted. The dataset might be empty or paths are incorrect.")
            return

        df = pd.DataFrame(all_features)

        # Determine project root and save to the correct Data directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "Data" / "PianoMotion10M"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_csv

        df.to_csv(output_path, index=False)

        logger.info(f"Dataset generation complete. Saved to {output_path}")
        logger.info(f"Generated {len(df)} samples.")
        logger.info(f"Label distribution:\n{df['ground_truth_label'].value_counts()}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    DATASET_DIR = script_dir / "PianoMotion10M" / "data"

    processor = SyncPianoMotionDataset(dataset_dir=DATASET_DIR)
    # To process the full dataset, remove the max_files argument
    processor.run(max_files=5)
