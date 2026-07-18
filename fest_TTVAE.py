"""
SYNTHETIC DATA EVALUATION FRAMEWORK (FEST) - CAUSAL ITE-INJECTION EDITION
=========================================================================

Description:
    This module implements an assessment framework to evaluate the 
    Causal Transformer-based Time-series VAE (C-TTVAE) against 
    ground truth clinical data.

Methodology (Iteration 6 - ITE Injection & G-Computation):
    The framework evaluates the model based on its ability to act as a Causal Extractor:
    1. Statistical Fidelity (Utility):
       - Marginal Distributions: Kolmogorov-Smirnov (KS) Test (Pre-treatment only).
       - Structural Integrity: Frobenius Norm of Correlation Matrices.
       - Causal Dynamics: Parallel Trends Stability and Exact Anchor Correlation.
       
    2. Causal Signal Extraction:
       - Computes the Average Treatment Effect on the Treated (ATT) directly from 
         the injected Individual Treatment Effects (ITE).

Author: Pietro Grassi
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
import random
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. REPRODUCIBILITY SETUP
# ==============================================================================
RANDOM_SEED = 2026

def set_all_seeds(seed: int = 2026):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(RANDOM_SEED)

# ==============================================================================
# SECTION A: BASE COMPUTATIONAL LOGIC (CAUSAL SLICING)
# ==============================================================================

class BaseCalculator:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                 real_name: str = "Real", synthetic_name: str = "Synthetic"):
        
        self.syn_raw = synthetic_data.copy()
        self.real_raw = real_data.copy()
        
        exclude_cols = ['survivor_id', 'deceased_id', 'mergeid', 'coupleid', 'is_synthetic', 
                        'imputation_id', 'aug_factor', 'dep_anchor', 'wealth_anchor']
        self.common_cols = [c for c in self.real_raw.columns.intersection(self.syn_raw.columns) if c not in exclude_cols]
        
        # 1. PRE-TREATMENT SLICE (Splicing Diagnostics - t < 0)
        self.r_pre = self.real_raw[self.real_raw['rel_time'] < 0][self.common_cols].copy()
        self.s_pre = self.syn_raw[self.syn_raw['rel_time'] < 0][self.common_cols].copy()
        
        self.s_name = synthetic_name
        self.target_col = 'dep_score' if 'dep_score' in self.common_cols else None

class MetricManager:
    def __init__(self): self.metrics = []
    def add_metric(self, m_list): self.metrics.extend(m_list)
    def evaluate_all(self):
        res = {}
        for m in self.metrics: res.update(m.calculate())
        return res

class CorrelationCalculator(BaseCalculator):
    def calculate(self):
        r_post = self.real_raw[self.real_raw['rel_time'] >= 0][self.common_cols]
        s_post = self.syn_raw[self.syn_raw['rel_time'] >= 0][self.common_cols]
        corr_r = r_post.corr(numeric_only=True)
        corr_s = s_post.corr(numeric_only=True)
        return {f"{self.s_name}_Corr_Diff": np.linalg.norm(corr_r.fillna(0) - corr_s.fillna(0))}

# ==============================================================================
# SECTION B: FIDELITY METRICS (UTILITY & DIAGNOSTICS)
# ==============================================================================

class SplicingDiagnosticsCalculator(BaseCalculator):
    def calculate(self) -> Dict[str, float]:
        """Tautological checks to prove the Exact Splicing worked."""
        if not self.target_col or self.r_pre.empty or self.s_pre.empty: return {}
        stat, _ = ks_2samp(self.r_pre[self.target_col].fillna(0), self.s_pre[self.target_col].fillna(0))
        return {f"{self.s_name}_Diag_PreKS": 1 - stat}

class GenerativeFidelityCalculator(BaseCalculator):
    def calculate(self) -> Dict[str, float]:
        if not self.target_col: return {}
        
        # Confronto Marginale: I Controlli Reali (Sani) vs i Controlli Sintetici (I trattati deprivati delle cure)
        # NB: Non cerchiamo 1.0 qui. Un punteggio più basso è giusto perché riflette il Selection Bias iniziale!
        r_controls = self.real_raw[(self.real_raw['rel_time'] >= 0) & (self.real_raw['treat_group'] == 0)]
        s_controls = self.syn_raw[(self.syn_raw['rel_time'] >= 0) & (self.syn_raw['treat_group'] == 0)]
        
        if r_controls.empty or s_controls.empty: return {}
        
        stat, _ = ks_2samp(r_controls[self.target_col].fillna(0), s_controls[self.target_col].fillna(0))
        post_ks = 1 - stat
        
        r_pivot = r_controls.pivot_table(index='survivor_id', columns='rel_time', values=self.target_col, aggfunc='mean')
        s_pivot = s_controls.pivot_table(index='survivor_id', columns='rel_time', values=self.target_col, aggfunc='mean')
        
        temp_corr_error = 999.0
        if 0 in r_pivot.columns and 1 in r_pivot.columns and 0 in s_pivot.columns and 1 in s_pivot.columns:
            r_temp_corr = r_pivot[0].corr(r_pivot[1])
            s_temp_corr = s_pivot[0].corr(s_pivot[1])
            temp_corr_error = abs((r_temp_corr if pd.notna(r_temp_corr) else 0) - (s_temp_corr if pd.notna(s_temp_corr) else 0))
            
        return {f"{self.s_name}_Gen_PostKS": post_ks, f"{self.s_name}_Gen_TempCorrErr": temp_corr_error}

# ==============================================================================
# SECTION C: PRIVACY METRICS & CAUSAL SIGNAL (ATT)
# ==============================================================================

class DCRCalculator(BaseCalculator):
    def calculate(self) -> Dict[str, float]:
        if not self.target_col: return {}
        r_post = self.real_raw[self.real_raw['rel_time'] >= 0]
        s_post = self.syn_raw[self.syn_raw['rel_time'] >= 0]
        
        if r_post.empty or s_post.empty: return {}
        
        scaler = MinMaxScaler()
        num_cols = [c for c in ['dep_score', 'wealth_log', 'adl_score', 'maxgrip'] if c in self.common_cols]
        r_scaled = scaler.fit_transform(r_post[num_cols].fillna(0))
        s_scaled = scaler.transform(s_post[num_cols].fillna(0))
        
        if len(s_scaled) > 2000:
            idx = np.random.choice(len(s_scaled), 2000, replace=False)
            s_scaled = s_scaled[idx]
            
        nbrs = NearestNeighbors(n_neighbors=1).fit(r_scaled)
        distances, _ = nbrs.kneighbors(s_scaled)
        
        return {f"{self.s_name}_Gen_DCR_Mean": np.mean(distances)}

class CausalSignalCalculator(BaseCalculator):
    def calculate(self) -> Dict[str, float]:
        """Misura l'Effetto Causale sui Trattati (ATT) tramite G-Computation."""
        if not self.target_col: return {}
        
        # Isoliamo i TRATTATI REALI (D=1)
        r_data = self.real_raw[(self.real_raw['rel_time'] >= 0) & (self.real_raw['treat_group'] == 1)][['survivor_id', 'rel_time', self.target_col]]
        # Isoliamo i loro GEMELLI SINTETICI (Forzati a Controllo, D=0)
        s_data = self.syn_raw[(self.syn_raw['rel_time'] >= 0) & (self.syn_raw['treat_group'] == 0)][['survivor_id', 'rel_time', self.target_col]]
        
        if r_data.empty or s_data.empty: return {}
        
        # Gestiamo imputazioni multiple con la media
        s_data_mean = s_data.groupby(['survivor_id', 'rel_time'], as_index=False)[self.target_col].mean()
        merged = r_data.merge(s_data_mean, on=['survivor_id', 'rel_time'], suffixes=('_real', '_syn'))
        
        if merged.empty: return {f"{self.s_name}_Causal_Signal_ATT": 0.0}
        
        # ATT = Y_Fattuale (1) - Y_Controfattuale (0)
        effect = (merged[f'{self.target_col}_real'] - merged[f'{self.target_col}_syn']).mean()
        return {f"{self.s_name}_Causal_Signal_ATT": effect}

# ==============================================================================
# SECTION D: VISUALIZATION
# ==============================================================================

def plot_causal_trajectories(real_df: pd.DataFrame, models_dict: Dict[str, pd.DataFrame], output_name: str = "FEST_Causal_Trajectories.png"):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.8", "grid.color": "0.9"})
    
    real_treated = real_df[real_df['treat_group'] == 1.0]
    sns.lineplot(data=real_treated, x='rel_time', y='dep_score', label='Factual (Real Treated)', 
                 color='#2980b9', marker='o', linewidth=2.5, errorbar=('ci', 95))
    
    if 'TTVAE' in models_dict:
        syn_ttvae = models_dict['TTVAE']
        # Il controfattuale dei trattati sono i sintetici forzati a 0
        syn_control = syn_ttvae[syn_ttvae['treat_group'] == 0.0]
        sns.lineplot(data=syn_control, x='rel_time', y='dep_score', label='Counterfactual (C-TTVAE Synthetic Twins)', 
                     color='#e74c3c', marker='X', linewidth=2.5, errorbar=('ci', 95), linestyle='--')
        
    plt.axvline(x=-0.5, color='#2c3e50', linestyle=':', linewidth=2, label='Spousal Bereavement (t=0)')
    plt.title("G-Computation Fidelity: Factual vs Counterfactual Trajectories", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Time Since Event (Waves)", fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel("Average EURO-D Depression Score", fontsize=13, fontweight='bold', labelpad=15)
    plt.xticks(np.arange(-3, 3, 1))
    
    plt.legend(fontsize='11', loc='lower right', framealpha=0.9)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_marginal_distributions(real_df: pd.DataFrame, syn_df: pd.DataFrame, target_model: str, output_name: str = "FEST_Distributions.png"):
    r_post = real_df[real_df['rel_time'] >= 0]
    s_post = syn_df[syn_df['rel_time'] >= 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.set_theme(style="whitegrid")
    
    col_real = "#2980b9" 
    col_syn = "#e74c3c"  
    
    sns.histplot(data=r_post[r_post['treat_group'] == 0], x='dep_score', color=col_real, stat='density', discrete=True, alpha=0.6, label='Real Controls (D=0)', ax=axes[0])
    sns.histplot(data=s_post[s_post['treat_group'] == 0], x='dep_score', color=col_syn, stat='density', discrete=True, alpha=0.6, label='Synthetic Twins (Counterfactual of D=1)', ax=axes[0])
    axes[0].set_title("Post-Intervention: Control Cohort (D=0)", fontweight='bold')
    axes[0].set_xlabel("EURO-D Depression Score")
    axes[0].legend()

    sns.histplot(data=r_post[r_post['treat_group'] == 1], x='dep_score', color=col_real, stat='density', discrete=True, alpha=0.6, label='Real Treated (D=1)', ax=axes[1])
    sns.histplot(data=s_post[s_post['treat_group'] == 1], x='dep_score', color=col_syn, stat='density', discrete=True, alpha=0.6, label='Synthetic Twins (Counterfactual of D=0)', ax=axes[1])
    axes[1].set_title("Post-Intervention: Treated Cohort (D=1)", fontweight='bold')
    axes[1].set_xlabel("EURO-D Depression Score")
    axes[1].legend()

    plt.suptitle(f"Marginal Fidelity Check: {target_model} vs Real Distribution", fontsize=16, fontweight='bold', y=1.02)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# SECTION E: MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=======================================================")
    print("   FEST FRAMEWORK: C-TTVAE G-COMPUTATION EVALUATION    ")
    print("=======================================================")
    
    if not os.path.exists('BaseDataset.csv'):
        print("❌ Critical Error: 'BaseDataset.csv' not found.")
        return
    
    real_df = pd.read_csv('BaseDataset.csv')
    print(f"1. Ground Truth Loaded. Shape: {real_df.shape}")

    target_file = 'Augmented_TTVAE_GComp.csv'
    model_name = 'TTVAE'
    
    if not os.path.exists(target_file):
        print(f"❌ Synthetic data not found: {target_file}. Please run generate_TTVAE.py first.")
        return

    full_df = pd.read_csv(target_file)
    if 'is_synthetic' in full_df.columns:
        syn_df = full_df[full_df['is_synthetic'] == 1].copy()
    else:
        syn_df = full_df.copy()
        
    print(f"   -> Found G-Computation Data | Extracted {len(syn_df)} Synthetic Twins.")
    models_data = {model_name: syn_df}

    print(f"\n--- Evaluating Model Architecture: {model_name} ---")
    
    utility_manager = MetricManager()
    utility_manager.add_metric([
        SplicingDiagnosticsCalculator(real_df, syn_df, synthetic_name=model_name),
        GenerativeFidelityCalculator(real_df, syn_df, synthetic_name=model_name),
        CorrelationCalculator(real_df, syn_df, synthetic_name=model_name)
    ])
    u_res = utility_manager.evaluate_all()
    
    privacy_manager = MetricManager()
    privacy_manager.add_metric([
        DCRCalculator(real_df, syn_df, synthetic_name=model_name),
        CausalSignalCalculator(real_df, syn_df, synthetic_name=model_name)
    ])
    p_res = privacy_manager.evaluate_all()
    
    full_metrics = {**u_res, **p_res}
    
    record = {
        'Model': model_name,
        'Diag Pre-KS (Expected 1.0)': full_metrics.get(f"{model_name}_Diag_PreKS", 0),
        'CF Divergence (KS)': full_metrics.get(f"{model_name}_Gen_PostKS", 0), 
        'Temp Corr Err': full_metrics.get(f"{model_name}_Gen_TempCorrErr", 999),
        'Covariance Frob Norm': full_metrics.get(f"{model_name}_Corr_Diff", 999), 
        'Gen DCR Mean': full_metrics.get(f"{model_name}_Gen_DCR_Mean", 0),
        'Learned Effect (ATT)': full_metrics.get(f"{model_name}_Causal_Signal_ATT", 0.0)
    }

    results_df = pd.DataFrame([record])
    
    print("\n=========================================================================================")
    print("               FINAL CAUSAL REPORT (G-COMPUTATION EDITION)                               ")
    print("=========================================================================================")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv("FEST_Evaluation_Results_TTVAE.csv", index=False)
    
    att_val = results_df['Learned Effect (ATT)'].iloc[0]
    print(f"\n✅ CAUSAL DIAGNOSTICS FOR C-TTVAE:")
    print(f"   -> Structural Causal Signal detected: ATT = {att_val:.4f}")

    print("\n📊 Generating Post-Intervention Distribution Plot for TTVAE...")
    plot_marginal_distributions(real_df, syn_df, target_model=model_name, output_name="FEST_Distributions_TTVAE.png")
    
    print("📈 Generating Causal Trajectories Plot...")
    plot_causal_trajectories(real_df, models_data, output_name="FEST_Causal_Trajectories_TTVAE.png")

if __name__ == "__main__":
    main()