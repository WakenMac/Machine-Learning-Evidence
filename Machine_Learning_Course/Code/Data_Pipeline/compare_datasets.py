"""
compare_datasets.py
Compares synthetic vs real PianoMotion10M datasets.
Shows statistics, distributions, and feature differences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


class DatasetComparator:
    """Compare synthetic and real datasets."""
    
    def __init__(self):
        self.data_dir = Path("Data/PianoMotion10M")
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load synthetic and real datasets."""
        print("📊 LOADING DATASETS\n")
        
        # Load or generate synthetic
        synthetic_path = self.data_dir / "features.csv"
        if synthetic_path.exists():
            synthetic_df = pd.read_csv(synthetic_path)
            print(f"✅ Synthetic data: {synthetic_path}")
            print(f"   Shape: {synthetic_df.shape}")
        else:
            print("❌ Synthetic data not found")
            synthetic_df = None
        
        # Load real
        real_path = self.data_dir / "features_real_pianomotion10m.csv"
        if real_path.exists():
            real_df = pd.read_csv(real_path)
            print(f"✅ Real data: {real_path}")
            print(f"   Shape: {real_df.shape}")
        else:
            print("❌ Real data not found (run DownloadRealPianoMotion10M.py first)")
            real_df = None
        
        return synthetic_df, real_df
    
    def basic_statistics(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Print basic statistics for both datasets."""
        print("\n" + "="*80)
        print("📈 BASIC STATISTICS")
        print("="*80 + "\n")
        
        if synthetic_df is not None:
            print("SYNTHETIC DATA:")
            print(f"  Total samples: {len(synthetic_df)}")
            print(f"  Features: {list(synthetic_df.columns[:-1])}")
            print(f"  Label distribution:\n{synthetic_df['label'].value_counts()}\n")
        
        if real_df is not None:
            print("REAL DATA:")
            print(f"  Total samples: {len(real_df)}")
            print(f"  Features: {list(real_df.columns[:-1])}\n")
            print(f"  Label distribution:\n{real_df['label'].value_counts()}\n")
        
        # Class balance comparison
        if synthetic_df is not None and real_df is not None:
            print("CLASS BALANCE:")
            synthetic_balance = (synthetic_df['label'].sum() / len(synthetic_df)) * 100
            real_balance = (real_df['label'].sum() / len(real_df)) * 100
            
            print(f"  Synthetic - Positive class: {synthetic_balance:.1f}%")
            print(f"  Real      - Positive class: {real_balance:.1f}%")
            print(f"  Difference: {abs(synthetic_balance - real_balance):.1f}%\n")
    
    def feature_statistics(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Compare feature statistics."""
        print("\n" + "="*80)
        print("📊 FEATURE STATISTICS")
        print("="*80 + "\n")
        
        if synthetic_df is None or real_df is None:
            print("Cannot compare - one or both datasets missing")
            return
        
        features = [col for col in synthetic_df.columns if col != 'label']
        
        print("Feature Ranges and Means:\n")
        print(f"{'Feature':<30} {'Synthetic':<40} {'Real':<40}")
        print("-" * 110)
        
        for feature in features:
            syn_mean = synthetic_df[feature].mean()
            syn_std = synthetic_df[feature].std()
            real_mean = real_df[feature].mean()
            real_std = real_df[feature].std()
            
            syn_range = f"Mean: {syn_mean:.4f}±{syn_std:.4f}"
            real_range = f"Mean: {real_mean:.4f}±{real_std:.4f}"
            
            print(f"{feature:<30} {syn_range:<40} {real_range:<40}")
        
        print("\n")
    
    def plot_distributions(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Plot feature distributions side-by-side."""
        if synthetic_df is None or real_df is None:
            print("Cannot plot - one or both datasets missing")
            return
        
        print("\n📊 Creating distribution plots...\n")
        
        features = [col for col in synthetic_df.columns if col != 'label']
        n_features = len(features)
        
        # Create figure
        fig, axes = plt.subplots(n_features, 2, figsize=(14, 3*n_features))
        if n_features == 1:
            axes = axes.reshape(1, 2)
        
        fig.suptitle('Feature Distribution Comparison: Synthetic vs Real', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, feature in enumerate(features):
            # Synthetic
            axes[idx, 0].hist(synthetic_df[feature], bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[idx, 0].set_title(f'Synthetic - {feature}', fontweight='bold')
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].grid(alpha=0.3)
            
            # Real
            axes[idx, 1].hist(real_df[feature], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[idx, 1].set_title(f'Real - {feature}', fontweight='bold')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.results_dir / "distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def plot_boxplots(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Plot boxplot comparison."""
        if synthetic_df is None or real_df is None:
            print("Cannot plot - one or both datasets missing")
            return
        
        print("📊 Creating boxplot comparison...\n")
        
        features = [col for col in synthetic_df.columns if col != 'label']
        n_features = len(features)
        
        fig, axes = plt.subplots(1, n_features, figsize=(4*n_features, 5))
        if n_features == 1:
            axes = [axes]
        
        fig.suptitle('Feature Range Comparison: Synthetic vs Real', 
                     fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(features):
            data_to_plot = [synthetic_df[feature], real_df[feature]]
            axes[idx].boxplot(data_to_plot, labels=['Synthetic', 'Real'])
            axes[idx].set_title(feature, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.results_dir / "boxplot_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def plot_class_distribution(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Plot class distribution comparison."""
        if synthetic_df is None or real_df is None:
            print("Cannot plot - one or both datasets missing")
            return
        
        print("📊 Creating class distribution plot...\n")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Synthetic
        synthetic_counts = synthetic_df['label'].value_counts()
        axes[0].pie(synthetic_counts.values, labels=['Hover', 'Press'], 
                   autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'],
                   startangle=90)
        axes[0].set_title('Synthetic Data Class Distribution', fontweight='bold', fontsize=12)
        
        # Real
        real_counts = real_df['label'].value_counts()
        axes[1].pie(real_counts.values, labels=['Hover', 'Press'], 
                   autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'],
                   startangle=90)
        axes[1].set_title('Real Data Class Distribution', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        output_path = self.results_dir / "class_distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def statistical_tests(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Perform statistical tests to identify significant differences."""
        from scipy import stats
        
        print("\n" + "="*80)
        print("📉 STATISTICAL SIGNIFICANCE TESTS")
        print("="*80 + "\n")
        
        if synthetic_df is None or real_df is None:
            print("Cannot test - one or both datasets missing")
            return
        
        features = [col for col in synthetic_df.columns if col != 'label']
        
        print("Kolmogorov-Smirnov Test (p-value < 0.05 = significant difference):\n")
        print(f"{'Feature':<30} {'KS Statistic':<15} {'p-value':<15} {'Significant':<15}")
        print("-" * 75)
        
        significant_count = 0
        for feature in features:
            ks_stat, p_value = stats.ks_2samp(synthetic_df[feature], real_df[feature])
            is_significant = "Yes" if p_value < 0.05 else "No"
            if p_value < 0.05:
                significant_count += 1
            
            print(f"{feature:<30} {ks_stat:<15.6f} {p_value:<15.6f} {is_significant:<15}")
        
        print(f"\nFeatures with significant differences: {significant_count}/{len(features)}")
        
        if significant_count > 0:
            print("\n⚠️  Found significant differences!")
            print("This means the real data distribution differs from synthetic.")
            print("Models trained on real data may perform differently than synthetic benchmarks.")
        else:
            print("\n✅ No significant differences found.")
            print("Synthetic and real data have similar distributions.")
    
    def create_summary_report(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame):
        """Create a summary report."""
        print("\n" + "="*80)
        print("📋 SUMMARY REPORT")
        print("="*80 + "\n")
        
        report = []
        
        report.append("DATASET COMPARISON SUMMARY")
        report.append("-" * 40)
        
        if synthetic_df is not None and real_df is not None:
            report.append(f"\nSamples:")
            report.append(f"  Synthetic: {len(synthetic_df)}")
            report.append(f"  Real: {len(real_df)}")
            report.append(f"  Ratio: {len(real_df) / len(synthetic_df):.1f}x larger")
            
            syn_positive = (synthetic_df['label'].sum() / len(synthetic_df)) * 100
            real_positive = (real_df['label'].sum() / len(real_df)) * 100
            
            report.append(f"\nClass Balance (% positive class):")
            report.append(f"  Synthetic: {syn_positive:.1f}%")
            report.append(f"  Real: {real_positive:.1f}%")
            
            report.append(f"\nExpected Performance:")
            report.append(f"  Models trained on real data may show:")
            report.append(f"  • Lower accuracy than synthetic benchmarks")
            report.append(f"  • Better generalization to new data")
            report.append(f"  • More realistic performance estimates")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = self.results_dir / "dataset_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✅ Report saved: {report_path}")
    
    def run_full_comparison(self):
        """Run complete comparison."""
        print("\n" + "="*80)
        print("🎹 REAL VS SYNTHETIC PIANOMOTION10M DATASET COMPARISON")
        print("="*80)
        
        # Load datasets
        synthetic_df, real_df = self.load_datasets()
        
        if synthetic_df is None or real_df is None:
            print("\n❌ Cannot run comparison - missing dataset(s)")
            return
        
        # Run analyses
        self.basic_statistics(synthetic_df, real_df)
        self.feature_statistics(synthetic_df, real_df)
        self.plot_distributions(synthetic_df, real_df)
        self.plot_boxplots(synthetic_df, real_df)
        self.plot_class_distribution(synthetic_df, real_df)
        self.statistical_tests(synthetic_df, real_df)
        self.create_summary_report(synthetic_df, real_df)
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}/")
        print("\nGenerated files:")
        print("  • distribution_comparison.png")
        print("  • boxplot_comparison.png")
        print("  • class_distribution_comparison.png")
        print("  • dataset_comparison_report.txt")


def main():
    """Main entry point."""
    comparator = DatasetComparator()
    comparator.run_full_comparison()


if __name__ == "__main__":
    main()
