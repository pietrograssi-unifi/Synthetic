"""
SYNTHETIC DATA EVALUATION FRAMEWORK (FEST)
==========================================

Description:
    This module implements a comparative assessment framework to benchmark 
    synthetic data generation architectures (CTGAN, TVAE, TTVAE) against 
    ground truth clinical data.

Methodology:
    The framework employs a "Dual-Axis" evaluation strategy:
    1. Statistical Fidelity (Utility):
       - Marginal Distributions: Kolmogorov-Smirnov (KS) Test.
       - Structural Integrity: Frobenius Norm of Correlation Matrices.
       - First-Order Moments: Mean Absolute Percentage Error (MAPE).
       - Causal Dynamics: Parallel Trends Stability for DiD designs.
       
    2. Privacy Preservation (Disclosure Risk):
       - Overfitting: Distance to Closest Record (DCR).
       - Distinguishability: Adversarial Accuracy (Propensity to discriminate).



Author: Research Team
Publication Context: Q1 Journal Submission
Date: February 2026
"""

import os
import warnings
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

# Suppress non-critical warnings to maintain clean console output
warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION A: BASE COMPUTATIONAL LOGIC
# ==============================================================================

class BaseCalculator:
    """
    Base class for metric calculators.
    
    Responsibility:
    Ensures feature space alignment between Real ($D_{real}$) and Synthetic ($D_{syn}$) 
    datasets. It handles schema intersections and applies necessary preprocessing 
    (imputation/encoding) to ensure metric stability.
    """
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 real_name: str = "Real", synthetic_name: str = "Synthetic"):
        
        # Store raw synthetic data to access auxiliary columns (Anchors) 
        # that may not exist in the Real dataset.
        self.syn_raw = synthetic_data.copy()
        
        # Ensure Anchor columns are numeric for correlation analysis
        for anchor in ['dep_anchor', 'wealth_anchor', 'dep_score', 'wealth_log']:
            if anchor in self.syn_raw.columns:
                self.syn_raw[anchor] = pd.to_numeric(self.syn_raw[anchor], errors='coerce')
        
        # Sanitize NaNs in Raw Data
        self.syn_raw.fillna(0, inplace=True)
        
        # 1. Feature Space Alignment (Intersection Logic)
        # Metrics are computed only on the subset of columns present in BOTH datasets.
        self.common_cols = real_data.columns.intersection(synthetic_data.columns)
        
        self.real = real_data[self.common_cols].copy()
        self.syn = synthetic_data[self.common_cols].copy()
        self.r_name = real_name
        self.s_name = synthetic_name
        
        # 2. Variable Type Detection
        self.num_cols = self.real.select_dtypes(include=np.number).columns
        self.cat_cols = self.real.select_dtypes(exclude=np.number).columns
        
        # 3. Preprocessing for Metric Stability
        # Note: We use central tendency imputation to allow distance-based metrics
        # (KS, DCR) to function. This does not alter the generative model itself.
        if len(self.num_cols) > 0:
            self.real[self.num_cols] = self.real[self.num_cols].fillna(self.real[self.num_cols].mean())
            self.syn[self.num_cols] = self.syn[self.num_cols].fillna(self.syn[self.num_cols].mean())
        
        if len(self.cat_cols) > 0:
            for col in self.cat_cols:
                mode_val = self.real[col].mode()[0]
                self.real[col] = self.real[col].fillna(mode_val)
                self.syn[col] = self.syn[col].fillna(mode_val)


class MetricManager:
    """Abstract orchestrator for executing a batch of metric calculators."""
    def __init__(self): 
        self.metrics = []
        
    def add_metric(self, m: Union[BaseCalculator, List[BaseCalculator]]): 
        if isinstance(m, list): 
            self.metrics.extend(m)
        else: 
            self.metrics.append(m)
            
    def evaluate_all(self) -> Dict[str, float]:
        results = {}
        for metric in self.metrics: 
            results.update(metric.calculate())
        return results


# ==============================================================================
# SECTION B: FIDELITY METRICS (UTILITY)
# ==============================================================================

class BasicStatsCalculator(BaseCalculator):
    """
    Evaluates the preservation of First Moments (Mean).
    Metric: Mean Absolute Percentage Error (MAPE).
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [Fidelity] Calculating First Moment Preservation (MAPE)...")
        if len(self.num_cols) == 0: return {}
        
        real_mean = self.real[self.num_cols].mean()
        syn_mean = self.syn[self.num_cols].mean()
        
        # Epsilon added to denominator to prevent division by zero
        mape = np.mean(np.abs((real_mean - syn_mean) / (real_mean + 1e-6)))
        return {f"{self.s_name}_Mean_MAPE": mape}


class KSCalculator(BaseCalculator):
    """
    Kolmogorov-Smirnov (KS) Test.
    Measures the maximum distance between the Empirical Cumulative Distribution 
    Functions (ECDF) of real and synthetic attributes.
    Output: Avg Score (1.0 - statistic). 1.0 implies perfect overlap.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [Fidelity] Calculating KS Test (Marginal Distributions)...")
        if len(self.num_cols) == 0: return {}
        
        ks_scores = []
        for col in self.num_cols:
            stat, _ = ks_2samp(self.real[col], self.syn[col])
            ks_scores.append(1 - stat) 
            
        return {f"{self.s_name}_KS_Score_Avg": np.mean(ks_scores)}


class CorrelationCalculator(BaseCalculator):
    """
    Structural Correlation Analysis.
    Computes the Frobenius Norm of the difference between correlation matrices
    ($||R_{real} - R_{syn}||_F$) to assess inter-variable dependency preservation.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [Fidelity] Calculating Structural Correlation (Frobenius Norm)...")
        
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        r_enc = self.real.copy()
        s_enc = self.syn.copy()
        
        if len(self.cat_cols) > 0:
            # Fit on combined data for consistent encoding
            full_cat = pd.concat([r_enc[self.cat_cols].astype(str), s_enc[self.cat_cols].astype(str)])
            enc.fit(full_cat)
            r_enc[self.cat_cols] = enc.transform(r_enc[self.cat_cols].astype(str))
            s_enc[self.cat_cols] = enc.transform(s_enc[self.cat_cols].astype(str))

        corr_r = r_enc.corr()
        corr_s = s_enc.corr()
        
        diff_norm = np.linalg.norm(corr_r.fillna(0) - corr_s.fillna(0))
        return {f"{self.s_name}_Correlation_Diff": diff_norm}
    

class CausalFidelityCalculator(BaseCalculator):
    """
    Longitudinal Causal Fidelity Assessment.
    
    Evaluates the suitability of synthetic data for Difference-in-Differences (DiD).
    
    

    Metrics:
    1. Anchor Correlation: Verifies if trajectories respect the individual's baseline history.
    2. Parallel Trends Stability: Quantifies the divergence between Treated and Control 
       groups in the pre-treatment period (should be ~0).
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [Fidelity] Calculating Causal Metrics (Trends & Anchors)...")
        results = {}

        # Use syn_raw to access 'Anchor' columns not present in real data
        df_eval = self.syn_raw
        
        # 1. Anchor Consistency Check
        for anchor_col in ['dep_anchor', 'wealth_anchor']:
            target_col = 'dep_score' if 'dep' in anchor_col else 'wealth_log'
            
            if anchor_col in df_eval.columns and target_col in df_eval.columns:
                
                # Isolate Baseline (t=-1)
                if 'rel_time' in df_eval.columns:
                    base_slice = df_eval[df_eval['rel_time'] == -1]
                else:
                    base_slice = df_eval
                
                if not base_slice.empty:
                    corr = base_slice[target_col].corr(base_slice[anchor_col])
                    results[f"{self.s_name}_{target_col}_Anchor_Corr"] = corr
        
        # 2. Parallel Trends Stability (Pre-Trend Slope)
        req_cols = ['rel_time', 'treat_group', 'dep_score']
        if all(c in df_eval.columns for c in req_cols):
            trends = df_eval.groupby(['rel_time', 'treat_group'])['dep_score'].mean().unstack()
            
            # Check for slope between t=-3 and t=-1
            if 0 in trends.columns and 1 in trends.columns:
                trends['diff'] = trends[1] - trends[0]
                if -3 in trends.index and -1 in trends.index:
                    slope = abs(trends.loc[-1, 'diff'] - trends.loc[-3, 'diff'])
                    results[f"{self.s_name}_PreTrend_Stability_Slope"] = slope 
        
        return results


# ==============================================================================
# SECTION C: PRIVACY METRICS (DISCLOSURE RISK)
# ==============================================================================

class DCRCalculator(BaseCalculator):
    """
    Distance to Closest Record (DCR).
    Measures the Euclidean distance to the nearest neighbor in the real dataset.
    Low DCR indicates potential overfitting/memorization.
    """
    def calculate(self) -> Dict[str, float]:
        print(f"   [Privacy] Calculating Distance to Closest Record (DCR)...")
        if len(self.num_cols) == 0: return {}
        
        scaler = MinMaxScaler()
        r_scaled = scaler.fit_transform(self.real[self.num_cols])
        s_scaled = scaler.transform(self.syn[self.num_cols])
        
        # Subsampling for efficiency if N > 3000
        if len(s_scaled) > 3000:
            idx = np.random.choice(len(s_scaled), 3000, replace=False)
            s_scaled = s_scaled[idx]
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(r_scaled)
        distances, _ = nbrs.kneighbors(s_scaled)
        
        return {f"{self.s_name}_DCR_Mean": np.mean(distances)}


class AdversarialAccuracyCalculator(BaseCalculator):
    """
    Adversarial Accuracy (AA).
    
    Simulates a "Linkage Attack" using a Random Forest discriminator.
    AA ~ 0.5 implies indistinguishability; AA > 0.9 implies high disclosure risk.
    """
    
    

    def calculate(self) -> Dict[str, float]:
        print(f"   [Privacy] Calculating Adversarial Accuracy...")
        X_real = self.real.copy()
        X_syn = self.syn.copy()
        X_real['label'] = 0
        X_syn['label'] = 1
        
        combined = pd.concat([X_real, X_syn], axis=0).sample(frac=1.0, random_state=42)
        y = combined['label']
        X = combined.drop('label', axis=1)
        
        for col in X.select_dtypes(include='object').columns:
            X[col] = pd.factorize(X[col])[0]
        X = X.fillna(0)
        
        if len(X) > 10000:
            X = X.iloc[:10000]
            y = y.iloc[:10000]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        return {f"{self.s_name}_Adversarial_Accuracy": accuracy_score(y_test, clf.predict(X_test))}


# ==============================================================================
# SECTION D: VISUALIZATION
# ==============================================================================

def plot_feature_distribution(real_df: pd.DataFrame, models_dict: Dict[str, pd.DataFrame], 
                              target_col: str, output_name: str = "Distribution_Comparison.png"):
    """Generates a comparative Kernel Density Estimation (KDE) plot."""
    if target_col not in real_df.columns: return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_df[target_col], label='Real Data', fill=True, color='black', alpha=0.1, linewidth=2)
    
    colors = {'CTGAN': 'blue', 'TVAE': 'green', 'TTVAE': 'red'}
    for name, df in models_dict.items():
        if target_col in df.columns:
            sns.kdeplot(df[target_col], label=f'{name} (Synthetic)', 
                        color=colors.get(name, 'gray'), linestyle='--')
            
    plt.title(f"Distributional Consistency: {target_col}", fontsize=14)
    plt.xlabel(target_col, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"   [Visualization] Saved distribution plot to '{output_name}'")


# ==============================================================================
# SECTION E: MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=======================================================")
    print("   FEST FRAMEWORK: SYNTHETIC DATA EVALUATION MODULE    ")
    print("=======================================================")
    
    # 1. Load Ground Truth
    if not os.path.exists('BaseDataset.csv'):
        print("❌ Critical Error: 'BaseDataset.csv' not found.")
        return
    
    real_df = pd.read_csv('BaseDataset.csv')
    print(f"1. Ground Truth Loaded. Shape: {real_df.shape}")

    # 2. Load Synthetic Candidates
    candidate_files = {
        'CTGAN': 'Augmented_CTGAN.csv',
        'TVAE': 'Augmented_TVAE.csv',
        'TTVAE': 'Augmented_TTVAE.csv'
    }
    
    models_data = {}
    for model_name, file_path in candidate_files.items():
        if os.path.exists(file_path):
            print(f"   -> Found Synthetic Data: {model_name}")
            models_data[model_name] = pd.read_csv(file_path)

    if not models_data:
        print("No synthetic data found. Please run the generation script first.")
        return

    # 3. Evaluation Loop
    final_results = []

    for model_name, syn_df in models_data.items():
        print(f"\n--- Evaluating Model Architecture: {model_name} ---")
        
        # Fidelity (Utility) Assessment
        utility_manager = MetricManager()
        utility_manager.add_metric([
            BasicStatsCalculator(real_df, syn_df, synthetic_name=model_name),
            KSCalculator(real_df, syn_df, synthetic_name=model_name),
            CorrelationCalculator(real_df, syn_df, synthetic_name=model_name),
            CausalFidelityCalculator(real_df, syn_df, synthetic_name=model_name)
        ])
        u_res = utility_manager.evaluate_all()
        
        # Privacy (Risk) Assessment
        privacy_manager = MetricManager()
        privacy_manager.add_metric([
            DCRCalculator(real_df, syn_df, synthetic_name=model_name),
            AdversarialAccuracyCalculator(real_df, syn_df, synthetic_name=model_name)
        ])
        p_res = privacy_manager.evaluate_all()
        
        # Consolidate Results
        full_metrics = {**u_res, **p_res}
        
        record = {
            'Model': model_name,
            'KS Score (Fidelity)': full_metrics.get(f"{model_name}_KS_Score_Avg", 0),
            'MAPE (Mean Error)': full_metrics.get(f"{model_name}_Mean_MAPE", 0),
            'Corr. Diff (Structure)': full_metrics.get(f"{model_name}_Correlation_Diff", 999),
            'DCR (Privacy)': full_metrics.get(f"{model_name}_DCR_Mean", 0),
            'Adv. Accuracy': full_metrics.get(f"{model_name}_Adversarial_Accuracy", 0.5),
            'Pre-Trend Slope': full_metrics.get(f"{model_name}_PreTrend_Stability_Slope", 999),
            'Dep Anchor Corr': full_metrics.get(f"{model_name}_dep_score_Anchor_Corr", 0),      
            'Wealth Anchor Corr': full_metrics.get(f"{model_name}_wealth_log_Anchor_Corr", 0) 
        }
        final_results.append(record)

    # 4. Generate Reports
    results_df = pd.DataFrame(final_results)
    
    print("\n=======================================================")
    print("               FINAL COMPARATIVE REPORT                ")
    print("=======================================================")
    print(results_df.round(4).to_string(index=False))
    
    results_df.to_csv("FEST_Evaluation_Results.csv", index=False)
    
    # 5. Determine Optimal Model
    if not results_df.empty:
        winner = results_df.loc[results_df['KS Score (Fidelity)'].idxmax()]
        print(f"\n🏆 OPTIMAL MODEL: {winner['Model']}")
        print(f"   Reasoning: Highest marginal fidelity score (KS = {winner['KS Score (Fidelity)']:.4f})")

    # 6. Visualization
    target_col = 'maxgrip' 
    if target_col not in real_df.columns:
        numerics = real_df.select_dtypes(include=np.number).columns
        if len(numerics) > 0:
            target_col = numerics[0]
            
    plot_feature_distribution(real_df, models_data, target_col)

if __name__ == "__main__":
    main()
