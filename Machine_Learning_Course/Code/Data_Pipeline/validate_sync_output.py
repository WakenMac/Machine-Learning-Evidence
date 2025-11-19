"""
validate_sync_output.py
Validates the output of the SyncPianoMotionDataset.py script.
"""

import pandas as pd
from pathlib import Path

def validate_output():
    """Validates the features.csv file."""

    features_csv = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M" / "features.csv"

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)

    if df.empty:
        raise ValueError("The features.csv file is empty.")

    expected_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'ground_truth_label']
    missing_cols = [col for col in expected_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    print("✅ Validation successful: features.csv has the correct format.")

if __name__ == "__main__":
    try:
        validate_output()
        print("✅ All checks passed!")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Validation failed: {e}")
        import sys
        sys.exit(1)
