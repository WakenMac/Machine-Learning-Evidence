"""
DownloadPianoMotion10M.py
Downloads and extracts the PianoMotion10M dataset from GitHub.
Dataset: https://github.com/agnJason/PianoMotion10M
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import shutil

def download_pianomotion10m(output_dir: str = None) -> str:
    """
    Downloads the PianoMotion10M dataset and extracts it.
    
    Args:
        output_dir: Directory to save the dataset. If None, uses relative path.
    
    Returns:
        Path to the extracted dataset.
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GitHub raw content URL for the dataset
    # Using releases or direct GitHub archive download
    github_url = "https://github.com/agnJason/PianoMotion10M/archive/refs/heads/main.zip"
    zip_path = output_dir / "PianoMotion10M.zip"
    
    print(f"📥 Downloading PianoMotion10M dataset from GitHub...")
    print(f"   URL: {github_url}")
    print(f"   Destination: {output_dir}")
    
    try:
        # Download the file with progress reporting
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 // total_size, 100)
            print(f"\r   Progress: {percent}%", end="")
        
        urllib.request.urlretrieve(github_url, zip_path, download_progress)
        print("\n✅ Download complete!")
        
        # Extract the zip file
        print(f"📂 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("✅ Extraction complete!")
        
        # Move contents from PianoMotion10M-main subfolder to parent
        extracted_dir = output_dir / "PianoMotion10M-main"
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                dest = output_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), str(dest))
            shutil.rmtree(extracted_dir)
            print(f"✅ Reorganized dataset structure")
        
        # Clean up zip file
        zip_path.unlink()
        print(f"✅ Cleaned up temporary files")
        
        # List dataset contents
        print(f"\n📊 Dataset contents:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                file_count = len(list(item.glob("**/*")))
                print(f"   📁 {item.name}/ ({file_count} items)")
            else:
                print(f"   📄 {item.name}")
        
        return str(output_dir)
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise


def verify_dataset(dataset_dir: str) -> bool:
    """
    Verifies that the dataset has been properly extracted.
    
    Args:
        dataset_dir: Path to the dataset directory.
    
    Returns:
        True if dataset is valid, False otherwise.
    """
    dataset_path = Path(dataset_dir)
    
    # Check for expected directories
    expected_dirs = ["data", "scripts"]  # Common structure in motion capture datasets
    
    print(f"🔍 Verifying dataset structure at {dataset_path}...")
    
    if dataset_path.exists():
        contents = list(dataset_path.iterdir())
        print(f"   Found {len(contents)} items in dataset directory")
        for item in contents:
            print(f"     - {item.name}")
        return True
    else:
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False


if __name__ == "__main__":
    import sys
    
    # Determine output directory
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = None
    
    try:
        dataset_dir = download_pianomotion10m(output_path)
        verify_dataset(dataset_dir)
        print(f"\n🎉 Dataset ready at: {dataset_dir}")
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        sys.exit(1)
