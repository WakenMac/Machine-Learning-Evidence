"""
setup_real_dataset.py
Automated setup script for downloading and processing real PianoMotion10M dataset.
Handles dependencies and creates complete training pipeline.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SetupHelper:
    """Helps setup the real dataset pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "Data" / "PianoMotion10M"
        self.code_dir = self.project_root / "Code" / "Data_Pipeline"
    
    def check_dependencies(self) -> bool:
        """Check and install required dependencies."""
        print("\n" + "="*80)
        print("🔧 CHECKING DEPENDENCIES")
        print("="*80 + "\n")
        
        required_packages = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'mido': 'mido',  # For MIDI parsing
        }
        
        missing = []
        
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name} (missing)")
                missing.append(package_name)
        
        if missing:
            print(f"\n⚠️  Installing missing packages: {', '.join(missing)}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                print("✅ Dependencies installed successfully!")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                return False
        
        print("\n✅ All dependencies installed!")
        return True
    
    def download_dataset(self) -> bool:
        """Download the real PianoMotion10M dataset."""
        print("\n" + "="*80)
        print("📥 DOWNLOADING REAL PIANOMOTION10M DATASET")
        print("="*80)
        
        print("\nThis will:")
        print("  • Download ~1-2 GB from GitHub")
        print("  • Extract to Data/PianoMotion10M/")
        print("  • Parse motion capture and MIDI files")
        print("\nEstimated time: 5-15 minutes (depends on internet speed)")
        
        response = input("\nProceed with download? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Download cancelled.")
            return False
        
        try:
            script_path = self.code_dir / "DownloadRealPianoMotion10M.py"
            subprocess.check_call([sys.executable, str(script_path)])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def prepare_features(self) -> bool:
        """Prepare features from downloaded dataset."""
        print("\n" + "="*80)
        print("⚙️  PREPARING FEATURES")
        print("="*80 + "\n")
        
        try:
            # Check if dataset was downloaded
            if not (self.data_dir / "data").exists():
                print("❌ Dataset not found. Please download first using --download")
                return False
            
            print("Preparing features from real dataset...")
            print("This includes:")
            print("  • Aligning 3D hand coordinates with MIDI events")
            print("  • Computing motion features (velocity, acceleration, etc.)")
            print("  • Creating labeled training dataset")
            
            script_path = self.code_dir / "DownloadRealPianoMotion10M.py"
            subprocess.check_call([sys.executable, str(script_path)])
            
            # Check output
            feature_file = self.data_dir / "features_real_pianomotion10m.csv"
            if feature_file.exists():
                print(f"\n✅ Features prepared: {feature_file}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return False
    
    def train_models(self) -> bool:
        """Train ML models on real dataset."""
        print("\n" + "="*80)
        print("🤖 TRAINING ML MODELS")
        print("="*80 + "\n")
        
        print("Training SVM and Random Forest models...")
        print("This includes:")
        print("  • Hyperparameter tuning with RandomizedSearchCV")
        print("  • 3-fold cross-validation")
        print("  • Performance evaluation")
        print("\nEstimated time: 5-10 minutes")
        
        try:
            script_path = self.code_dir / "ML_Pipeline_Prep.py"
            subprocess.check_call([sys.executable, str(script_path)])
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def print_status(self):
        """Print current setup status."""
        print("\n" + "="*80)
        print("📊 SETUP STATUS")
        print("="*80 + "\n")
        
        checks = {
            "Dataset Downloaded": (self.data_dir / "data").exists(),
            "Features Prepared": (self.data_dir / "features_real_pianomotion10m.csv").exists(),
            "Models Trained": (self.data_dir / "models").exists(),
        }
        
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
        
        print("\n" + "="*80)


def main():
    """Main setup menu."""
    print("\n" + "="*80)
    print("🎹 REAL PIANOMOTION10M - AUTOMATED SETUP")
    print("="*80)
    print("\nThis script will help you:")
    print("  1. Download the real PianoMotion10M dataset from GitHub")
    print("  2. Parse 3D hand coordinates and MIDI annotations")
    print("  3. Extract ML features")
    print("  4. Train SVM and Random Forest models")
    
    helper = SetupHelper()
    
    while True:
        print("\n" + "-"*80)
        print("\nOPTIONS:")
        print("  1. Check dependencies")
        print("  2. Download dataset")
        print("  3. Prepare features")
        print("  4. Train models")
        print("  5. Full setup (1-4)")
        print("  6. Print status")
        print("  7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            helper.check_dependencies()
        
        elif choice == '2':
            helper.download_dataset()
        
        elif choice == '3':
            helper.prepare_features()
        
        elif choice == '4':
            helper.train_models()
        
        elif choice == '5':
            print("\n🚀 Starting full setup...\n")
            if helper.check_dependencies():
                if helper.download_dataset():
                    helper.prepare_features()
                    helper.train_models()
            print("\n✅ Setup complete!")
        
        elif choice == '6':
            helper.print_status()
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
