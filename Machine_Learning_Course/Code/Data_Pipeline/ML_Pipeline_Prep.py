"""
ML_Pipeline_Prep.py
Trains and compares SVM and Random Forest models for piano motion classification.
Includes hyperparameter tuning via RandomizedSearchCV and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import time
import logging

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PianoMotionMLPipeline:
    """
    Machine Learning pipeline for piano motion classification.
    Compares SVM and Random Forest models with comprehensive evaluation.
    """
    
    def __init__(self, features_csv: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize ML pipeline.
        
        Args:
            features_csv: Path to CSV file with extracted features
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.features_csv = Path(features_csv)
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        self.models = {}
        self.results = {}
        self.inference_times = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features from CSV and split into train/test sets.
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info(f"Loading dataset from {self.features_csv}")
        
        if not self.features_csv.exists():
            logger.error(f"Features file not found: {self.features_csv}")
            raise FileNotFoundError(f"Features CSV not found: {self.features_csv}")
        
        # Load data
        df = pd.read_csv(self.features_csv)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Separate features and labels
        label_col = 'ground_truth_label'
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")
        
        # Drop non-feature columns
        drop_cols = ['frame_index', label_col]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[label_col]
        
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info("✅ Data loading and preparation complete")
        
        return X_train_scaled, X_test_scaled
    
    def train_svm_with_tuning(self, X_train: np.ndarray, X_test: np.ndarray) -> SVC:
        """
        Train SVM with hyperparameter tuning via RandomizedSearchCV.
        
        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled)
            
        Returns:
            Trained SVM model
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING: Support Vector Machine (SVM)")
        logger.info("="*60)
        
        # Hyperparameter grid for RandomizedSearchCV
        param_dist_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4],
        }
        
        # Base SVM model
        svm_base = SVC(probability=True, random_state=self.random_state)
        
        # RandomizedSearchCV for hyperparameter tuning
        logger.info("Running RandomizedSearchCV for SVM (20 iterations)...")
        svm_search = RandomizedSearchCV(
            svm_base, param_dist_svm, n_iter=20, cv=3, 
            scoring='f1_weighted', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        start_time = time.time()
        svm_search.fit(X_train, self.y_train)
        train_time = time.time() - start_time
        
        logger.info(f"✅ SVM training complete ({train_time:.2f}s)")
        logger.info(f"   Best parameters: {svm_search.best_params_}")
        logger.info(f"   Best CV F1 score: {svm_search.best_score_:.4f}")
        
        # Get best model
        svm_model = svm_search.best_estimator_
        self.models['SVM'] = svm_model
        
        return svm_model
    
    def train_rf_with_tuning(self, X_train: np.ndarray, X_test: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest with hyperparameter tuning via RandomizedSearchCV.
        
        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled)
            
        Returns:
            Trained Random Forest model
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING: Random Forest (RF)")
        logger.info("="*60)
        
        # Hyperparameter grid for RandomizedSearchCV
        param_dist_rf = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
        
        # Base Random Forest model
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # RandomizedSearchCV for hyperparameter tuning
        logger.info("Running RandomizedSearchCV for RF (20 iterations)...")
        rf_search = RandomizedSearchCV(
            rf_base, param_dist_rf, n_iter=20, cv=3, 
            scoring='f1_weighted', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        start_time = time.time()
        rf_search.fit(X_train, self.y_train)
        train_time = time.time() - start_time
        
        logger.info(f"✅ RF training complete ({train_time:.2f}s)")
        logger.info(f"   Best parameters: {rf_search.best_params_}")
        logger.info(f"   Best CV F1 score: {rf_search.best_score_:.4f}")
        
        # Get best model
        rf_model = rf_search.best_estimator_
        self.models['Random Forest'] = rf_model
        
        return rf_model
    
    def evaluate_model(self, model_name: str, model, X_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation with all metrics.
        
        Args:
            model_name: Name of the model
            model: Trained model
            X_test: Test features
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Probabilities for ROC-AUC
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_proba)
        
        # Inference speed (FPS) - protect against zero division
        fps = len(X_test) / inference_time if inference_time > 0 else float('inf')
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fps': fps,
            'inference_time_ms': (inference_time / len(X_test)) * 1000,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }
        
        self.results[model_name] = metrics
        
        # Log results
        logger.info(f"✅ {model_name} Results:")
        logger.info(f"   Accuracy:    {accuracy:.4f}")
        logger.info(f"   Precision:   {precision:.4f}")
        logger.info(f"   Recall:      {recall:.4f}")
        logger.info(f"   F1-Score:    {f1:.4f}")
        logger.info(f"   ROC-AUC:     {roc_auc:.4f}")
        logger.info(f"   Inference:   {fps:.2f} FPS ({metrics['inference_time_ms']:.2f} ms/frame)")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comprehensive comparison table of all models.
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("\n" + "="*60)
        logger.info("📋 MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'FPS': f"{metrics['fps']:.2f}",
                'Inference (ms)': f"{metrics['inference_time_ms']:.2f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def visualize_results(self, output_dir: str = None) -> str:
        """
        Create visualization plots comparing models.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Path to saved figure
        """
        if output_dir is None:
            output_dir = self.features_csv.parent / "plots"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n📈 Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SVM vs Random Forest - Piano Motion Classification', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrices
        models_list = list(self.results.keys())
        
        for idx, model_name in enumerate(models_list[:2]):  # Only 2 models
            ax = axes[0, idx]
            cm = self.results[model_name]['confusion_matrix']
            
            im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
            ax.set_title(f'{model_name} - Confusion Matrix')
            
            # Labels
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['Hover', 'Press'])
            ax.set_yticklabels(['Hover', 'Press'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # 3. Metrics Comparison Bar Chart
        ax = axes[1, 0]
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_names))
        width = 0.35
        
        for idx, model_name in enumerate(models_list[:2]):
            metrics = self.results[model_name]
            values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
            ax.bar(x + idx*width, values, width, label=model_name, alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Inference Speed Comparison
        ax = axes[1, 1]
        
        fps_values = [self.results[name]['fps'] for name in models_list[:2]]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(models_list[:2], fps_values, color=colors, alpha=0.8)
        ax.set_ylabel('Frames Per Second (FPS)')
        ax.set_title('Inference Speed Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Comparison plot saved: {output_path}")
        
        # Additional ROC curves plot
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        
        for model_name in models_list[:2]:
            metrics = self.results[model_name]
            y_proba = metrics['y_proba']
            
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc_val = metrics['roc_auc']
            
            ax2.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_val:.4f})', linewidth=2)
        
        # Diagonal line
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Piano Motion Classification')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        roc_path = output_dir / "roc_curves.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ ROC curves saved: {roc_path}")
        
        plt.close('all')
        
        return str(output_dir)
    
    def run_pipeline(self, output_dir: str = None) -> str:
        """
        Execute complete ML pipeline.
        
        Args:
            output_dir: Directory to save results and plots
            
        Returns:
            Path to output directory with results
        """
        if output_dir is None:
            output_dir = self.features_csv.parent / "results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING PIANOMOTION ML PIPELINE")
        logger.info("="*60)
        
        try:
            # Load and prepare data
            X_train_scaled, X_test_scaled = self.load_and_prepare_data()
            
            # Train models
            svm_model = self.train_svm_with_tuning(X_train_scaled, X_test_scaled)
            rf_model = self.train_rf_with_tuning(X_train_scaled, X_test_scaled)
            
            # Evaluate models
            self.evaluate_model('SVM', svm_model, X_test_scaled)
            self.evaluate_model('Random Forest', rf_model, X_test_scaled)
            
            # Compare models
            comparison_df = self.compare_models()
            comparison_csv = output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            logger.info(f"✅ Comparison table saved: {comparison_csv}")
            
            # Visualize results
            self.visualize_results(str(output_dir))
            
            logger.info("\n" + "="*60)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("="*60)
            logger.info(f"Results saved to: {output_dir}")
            
            return str(output_dir)
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Execute the complete ML pipeline."""
    
    # Define paths
    features_csv = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M" / "features.csv"
    output_dir = Path(__file__).parent.parent.parent / "Data" / "PianoMotion10M" / "results"
    
    # Create pipeline
    pipeline = PianoMotionMLPipeline(str(features_csv), test_size=0.2, random_state=42)
    
    # Run pipeline
    results_dir = pipeline.run_pipeline(str(output_dir))
    
    print(f"\n🎉 All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
