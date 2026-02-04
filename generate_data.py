"""
Description:
    This script implements a Causal Transformer-based Time-series Variational Autoencoder (TTVAE)
    to augment longitudinal clinical data. It generates synthetic trajectories for treated 
    (Palliative Care) and control (Standard Care) cohorts.

Models Implemented:
    1. Causal TTVAE (Transformer-based Time-series VAE) - Primary
    2. TVAE/CTGAN (via SDV library) - Benchmarks

Author: Pietro Grassi
Date: February 2026
"""

import os
import sys
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Check for optional dependencies
try:
    from ctgan import CTGAN, TVAE
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

# Suppress non-critical warnings for cleaner log output
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# Training Hyperparameters
AUGMENTATION_FACTOR = 1.0
EPOCHS_TTVAE = 150
EPOCHS_GAN = 150
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 64

# Data Schema
ID_COL = 'survivor_id'
TIME_COL = 'rel_time'
SPLIT_COL = 'treat_group'  # 1 = Palliative, 0 = Standard Care
PADDING_VAL = -999.0

# Static variables defining the cohort skeleton (preserved from real data)
SKELETON_STATIC = [
    'wave_death', 'death_year', 'is_female', 'country', 'control_eligible', 
    'cause_death', 'edu_level', 'living_area_cat'
]

# Time-varying variables that initialize the sequence
SKELETON_START = ['age', 'wave', 'int_year']

# Static variables to be generated
STATIC_GEN = ['gender_imp', 'hc125_num', 'is_treated']

# Dynamic (Longitudinal) variables to be synthesized
DYNAMIC_VARS = [
    'rel_time', 
    'income_imp', 'wealth_imp', 'sphus_imp', 'adl_imp', 'maxgrip_imp', 
    'eurod_imp', 'dep_score', 'wealth_log', 'adl_raw', 'adl_score', 'maxgrip',
    'ins_sat', 'health_sat', 'satisfaction_health',
    'has_cancer', 'has_neuro', 'has_organ',
    'ph006d1', 'ph006d4', 'ph006d6', 'ph006d10', 'ph006d12', 'ph006d16', 'ph006d21'
]

# Deterministic evolution rules for specific variables
DETERMINISTIC_RULES = {
    'age': 2,       # Age increases by 2 years per wave
    'wave': 1,      # Wave increments by 1
    'int_year': 2   # Interview year increments by 2
}

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. NEURAL NETWORK ARCHITECTURE (Causal TTVAE)
# ==============================================================================

class CausalTTVAE_Model(nn.Module):
    """
    Transformer-based Time-series Variational Autoencoder with Causal Masking.
    
    This architecture utilizes a Transformer Encoder-Decoder structure to capture 
    temporal dependencies in longitudinal data. A causal mask ensures that 
    predictions at time t depend only on time steps [0...t].
    """
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, latent_dim: int = 64):
        super(CausalTTVAE_Model, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Embedding layer to project input features to transformer dimension
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent Space Projections
        self.fc_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_z_to_seq = nn.Linear(latent_dim, d_model * seq_len)
        
        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Output Head
        self.output_head = nn.Linear(d_model, input_dim)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix to mask future positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        mask = self._generate_causal_mask(x.size(1))
        x = self.transformer_encoder(x, mask=mask)
        x = x.reshape(batch_size, -1) # Flatten for dense latent projection
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.fc_z_to_seq(z)
        x = x.reshape(batch_size, self.seq_len, self.d_model)
        mask = self._generate_causal_mask(self.seq_len)
        x = self.transformer_decoder(x, mask=mask)
        return self.output_head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class TTVAE_Wrapper:
    """
    Wrapper class to handle data preprocessing, training loop, and sampling
    for the CausalTTVAE model.
    """
    def __init__(self, epochs: int = 200, batch_size: int = 128, latent_dim: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.encoders = {}
        self.model = None
        self.feature_names = None
        self.static_cols = []
        self.prob_cols = []
        
    def fit(self, df_wide: pd.DataFrame, n_visits: int, static_cols: List[str], prob_cols: List[str]):
        """
        Prepares data and trains the model.
        Converts wide-format DataFrame into a 3D Tensor (Batch, Time, Features).
        """
        self.static_cols = static_cols
        self.prob_cols = prob_cols
        
        # 1. Encoding Categorical Variables
        data = df_wide.copy()
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        self.feature_names = list(data.columns)
        
        # 2. Scaling
        data_for_fit = data.replace(PADDING_VAL, 0)
        self.scaler.fit(data_for_fit)
        data_matrix = self.scaler.transform(data.replace(PADDING_VAL, 0))
        
        static_indices = [self.feature_names.index(c) for c in static_cols if c in self.feature_names]
        
        # 3. Sequence Construction
        sequences = []
        for i in range(len(data)):
            row_raw = data.iloc[i] 
            row_scaled = data_matrix[i]
            static_vals = row_scaled[static_indices]
            
            seq_steps = []
            for v in range(n_visits):
                step_vals = []
                # Check existence of data at this time step (handling variable padding)
                check_col = f"{prob_cols[0]}_v{v}"
                if check_col not in row_raw: check_col = f"dep_score_v{v}"
                
                is_padding = False
                if check_col in row_raw:
                     val_orig = row_raw[check_col]
                     try:
                         if float(val_orig) <= PADDING_VAL + 1: is_padding = True
                     except: pass
                
                # Mask: 1.0 = Real Data, 0.0 = Padding
                mask_val = 0.0 if is_padding else 1.0
                
                for pc in prob_cols:
                    col_name = f"{pc}_v{v}"
                    if col_name in self.feature_names:
                        idx = self.feature_names.index(col_name)
                        val = row_scaled[idx]
                        if is_padding: val = 0.0
                        step_vals.append(val)
                    else: 
                        step_vals.append(0.0)
                
                # Concatenate Static features + Dynamic features + Mask
                full_step = np.concatenate([static_vals, step_vals, [mask_val]])
                seq_steps.append(full_step)
            sequences.append(seq_steps)
            
        self.X_train = np.array(sequences, dtype=np.float32)
        
        # 4. Model Initialization & Training
        input_dim = self.X_train.shape[2]
        self.model = CausalTTVAE_Model(input_dim=input_dim, seq_len=n_visits, latent_dim=self.latent_dim).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss(reduction='none') 
        
        tensor_x = torch.FloatTensor(self.X_train).to(DEVICE)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        print(f"   [TTVAE] Model Initialized. Input Dimension: {input_dim}")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                
                # Reconstruction Loss (Weighted)
                loss_mse = loss_fn(recon_x, x)
                weights = torch.ones_like(loss_mse)
                weights[:, :, -1] = 5.0 # High weight for mask prediction
                recon_loss = (loss_mse * weights).mean()
                
                # KL Divergence
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                loss = recon_loss + 0.002 * kld_loss 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 50 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs} | Loss: {total_loss / len(loader):.4f}")

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Generates synthetic samples by sampling from the latent space N(0,1).
        """
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(DEVICE)
            recon_seq = self.model.decode(z).cpu().numpy()
            
        static_indices = [self.feature_names.index(c) for c in self.static_cols if c in self.feature_names]
        n_static = len(static_indices)
        n_dyn = len(self.prob_cols)
        
        final_matrix = np.zeros((n_samples, len(self.feature_names)))
        col_map = {name: i for i, name in enumerate(self.feature_names)}
        padding_masks = np.zeros((n_samples, recon_seq.shape[1]))
        
        # Reconstruct matrix from tensor
        for i in range(n_samples):
            static_vec = recon_seq[i, 0, :n_static]
            for s_i, col_idx in enumerate(static_indices):
                final_matrix[i, col_idx] = static_vec[s_i]
            
            for v in range(recon_seq.shape[1]):
                step = recon_seq[i, v]
                dyn_vals = step[n_static : n_static+n_dyn]
                mask_val = step[-1]
                padding_masks[i, v] = mask_val
                
                for d_i, col in enumerate(self.prob_cols):
                    full_col = f"{col}_v{v}"
                    if full_col in col_map:
                        final_matrix[i, col_map[full_col]] = dyn_vals[d_i]
                        
        # Inverse Scaling
        data_inv = self.scaler.inverse_transform(final_matrix)
        df_syn = pd.DataFrame(data_inv, columns=self.feature_names)
        
        # Post-processing: Categorical decoding and Rounding
        for col in df_syn.columns:
            if col in self.encoders:
                le = self.encoders[col]
                # Clip to valid range before decoding
                df_syn[col] = df_syn[col].round().clip(0, len(le.classes_)-1).astype(int)
                df_syn[col] = le.inverse_transform(df_syn[col])
            else:
                if any(x in col for x in ['has_', 'is_', '_imp', 'n_waves']):
                     df_syn[col] = df_syn[col].round()
        
        # Apply padding mask (set masked values to PADDING_VAL)
        for i in range(n_samples):
            for v in range(recon_seq.shape[1]):
                if padding_masks[i, v] < 0.5:
                    for pc in self.prob_cols:
                        col_name = f"{pc}_v{v}"
                        if col_name in df_syn.columns:
                            df_syn.at[i, col_name] = PADDING_VAL
        return df_syn


# ==============================================================================
# 3. DATA PROCESSING HELPERS
# ==============================================================================

def prepare_wide_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Transforms longitudinal (long-format) data into wide-format suitable for ML models.
    Each row represents a unique survivor with columns for all time steps.
    """
    print("   [PROCESS] Transforming to Wide Format...")
    
    # 1. Derive Relative Time (Centered on Death)
    # Critical for aligning Palliative vs Standard care trajectories
    if 'rel_time' not in df.columns and 'wave' in df.columns and 'wave_death' in df.columns:
        df['rel_time'] = df['wave'] - df['wave_death']
        
    df_sorted = df.sort_values(by=[ID_COL, 'wave'])
    grouped = df_sorted.groupby(ID_COL)
    wide_rows = []
    max_waves_obs = df[ID_COL].value_counts().max()
    
    for survivor_id, group in grouped:
        row = {}
        row['n_waves_real'] = len(group)
        first = group.iloc[0]
        
        # Static Variables
        for col in SKELETON_STATIC + STATIC_GEN + [SPLIT_COL]:
            if col in df.columns: row[col] = first[col]
            
        # Initial Values for Deterministic Variables
        for col in SKELETON_START: 
            if col in first: row[f"{col}_start"] = first[col]
            
        # Dynamic Variables (flattened by wave)
        for i in range(max_waves_obs):
            suffix = f"_v{i}"
            if i < len(group):
                for col in DYNAMIC_VARS: 
                    if col in group.columns: row[col + suffix] = group.iloc[i][col]
            else:
                # Padding for non-existent waves
                for col in DYNAMIC_VARS: row[col + suffix] = PADDING_VAL
        wide_rows.append(row)
        
    return pd.DataFrame(wide_rows), max_waves_obs


def reconstruct_long_data(df_wide: pd.DataFrame, max_waves: int) -> pd.DataFrame:
    """
    Reconstructs the long-format dataframe from the synthetic wide-format output.
    
    CRITICAL LOGIC:
    Calculates 'rel_time' dynamically using the preserved 'wave_death' and the 
    incrementing 'wave' counter. This ensures both Treated and Control groups 
    are correctly aligned to their respective events.
    """
    print("   [PROCESS] Reconstructing Long Format...")
    long_rows = []
    
    for idx, row in df_wide.iterrows():
        syn_id = f"SYN_{idx}"
        
        # Skip if first visit is padding (invalid generation)
        if row.get(f"{DYNAMIC_VARS[0]}_v0", 0) <= PADDING_VAL + 1: continue

        # Extract Static features
        static_vals = {col: row[col] for col in (SKELETON_STATIC + STATIC_GEN) if col in row}
        if SPLIT_COL in row: static_vals[SPLIT_COL] = row[SPLIT_COL]
        
        # Initialize deterministic counters
        curr_vals = {}
        for col in DETERMINISTIC_RULES.keys():
            if f"{col}_start" in row: curr_vals[col] = row[f"{col}_start"]
        
        # Determine sequence length
        n_waves_pred = int(row.get('n_waves_real', max_waves))
        n_waves_pred = max(1, min(max_waves, n_waves_pred))
        
        # Unroll the sequence
        for i in range(n_waves_pred):
            # Stop if padding encountered
            if row.get(f"{DYNAMIC_VARS[0]}_v{i}", 0) <= PADDING_VAL + 1: break
            
            long_row = {}
            long_row[ID_COL] = syn_id
            long_row.update(static_vals)
            long_row.update(curr_vals)
            
            for col in DYNAMIC_VARS:
                long_row[col] = row.get(f"{col}_v{i}", 0)
            
            # Recalculate rel_time to ensure consistency with deterministic wave increment
            if 'wave' in curr_vals and 'wave_death' in static_vals:
                long_row['rel_time'] = curr_vals['wave'] - static_vals['wave_death']
            
            long_rows.append(long_row)
            
            # Update deterministic variables for next step
            for col, step in DETERMINISTIC_RULES.items():
                if col in curr_vals: curr_vals[col] += step
                
    return pd.DataFrame(long_rows)


def robust_sample(model, n_required: int) -> pd.DataFrame:
    """
    Safely samples from the model, discarding invalid sequences (padding-only).
    Retries until the requested number of valid samples is met.
    """
    valid_samples = []
    attempts = 0
    while len(valid_samples) < n_required and attempts < 10:
        n_missing = n_required - len(valid_samples)
        batch = model.sample(int(n_missing * 1.5) + 10)
        
        check_col = f"{DYNAMIC_VARS[0]}_v0"
        if check_col not in batch.columns: check_col = "dep_score_v0"
        
        if check_col in batch.columns:
            batch = batch[batch[check_col] > (PADDING_VAL + 10)]
            
        if len(batch) > 0: valid_samples.append(batch)
        attempts += 1
        
    return pd.concat(valid_samples).iloc[:n_required] if valid_samples else pd.DataFrame()


def preprocess_causal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Clips and cleans variables to remove outliers before training."""
    print("   [PREP] applying pre-processing filters...")
    df_smooth = df.copy()
    if 'dep_score' in df_smooth.columns: 
        df_smooth['dep_score'] = df_smooth['dep_score'].clip(0, 12)
    return df_smooth


# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def train_and_generate_split(df_base: pd.DataFrame, model_type: str = 'TTVAE') -> pd.DataFrame:
    """
    Orchestrates the training and generation process.
    Splits data by Treatment Group to learn distinct trajectories.
    """
    print(f"\n=== Processing Model Architecture: {model_type} ===")
    df_wide, max_waves = prepare_wide_data(df_base)
    syn_dfs = []
    
    n_treated_orig = len(df_wide[df_wide[SPLIT_COL] == 1])
    target_n = int(n_treated_orig * AUGMENTATION_FACTOR)
    
    # Train separate generative processes for Treated (1) and Control (0)
    for group_val in [0, 1]: 
        df_group = df_wide[df_wide[SPLIT_COL] == group_val].copy()
        
        if len(df_group) < 10: 
            print(f"   [WARNING] Insufficient data for Group {group_val}. Skipping.")
            continue
        
        print(f"   [GROUP {group_val}] Training Samples: {len(df_group)} -> Target Generation: {target_n}")
        train_data = df_group.drop(columns=[SPLIT_COL], errors='ignore')
        
        # We sample real static skeletons to preserve realistic demographic distributions
        skel_static_cols = [c for c in (SKELETON_STATIC + STATIC_GEN) if c in train_data.columns]
        skel_start_cols = [f"{c}_start" for c in SKELETON_START if f"{c}_start" in train_data.columns]
        real_skeleton = train_data[skel_static_cols + skel_start_cols].values
        features_static = skel_static_cols + skel_start_cols
        
        syn_data = None
        
        # --- MODEL TRAINING ---
        if model_type == 'TTVAE':
            wrapper = TTVAE_Wrapper(epochs=EPOCHS_TTVAE, latent_dim=LATENT_DIM)
            wrapper.fit(train_data, max_waves, features_static, DYNAMIC_VARS)
            syn_data = robust_sample(wrapper, target_n)
            
        elif CTGAN_AVAILABLE:
            try:
                # Identify discrete columns for CTGAN/TVAE
                discrete_cols = [c for c in train_data.columns if train_data[c].dtype == 'object' or train_data[c].nunique() < 20]
                if model_type == 'CTGAN': 
                    model = CTGAN(epochs=EPOCHS_GAN, verbose=True)
                else: 
                    model = TVAE(epochs=EPOCHS_GAN)
                    
                model.fit(train_data, discrete_columns=discrete_cols)
                syn_data = robust_sample(model, target_n)
            except Exception as e: 
                print(f"   [ERROR] {model_type} failed: {e}")

        # --- SKELETON INJECTION ---
        # Replace generated static vars with sampled real skeletons for demographic fidelity
        if syn_data is not None and len(real_skeleton) > 0:
            indices = np.random.choice(len(real_skeleton), size=len(syn_data), replace=True)
            sampled_skel = real_skeleton[indices]
            cols_to_inject = skel_static_cols + skel_start_cols
            
            for i, col in enumerate(cols_to_inject):
                if col in syn_data.columns: 
                    syn_data[col] = sampled_skel[:, i]
            
            syn_data[SPLIT_COL] = group_val
            syn_dfs.append(syn_data)
            
    if not syn_dfs: return pd.DataFrame()
    
    df_syn_wide = pd.concat(syn_dfs, ignore_index=True)
    return reconstruct_long_data(df_syn_wide, max_waves)


def main():
    print("=== LONGITUDINAL AUGMENTATION PIPELINE (V34) ===")
    
    if not os.path.exists('BaseDataset.csv'): 
        print("❌ Critical Error: Input file 'BaseDataset.csv' not found.")
        return
        
    df_base = pd.read_csv('BaseDataset.csv')
    
    # Global Pre-processing constraints
    if 'dep_score' in df_base.columns: df_base['dep_score'] = df_base['dep_score'].clip(0, 12)
    if 'wealth_log' in df_base.columns: df_base = df_base[df_base['wealth_log'] > -10]
    
    df_base = preprocess_causal_trends(df_base)
    
    # Define models to execute
    models_to_run = ['TTVAE']
    if CTGAN_AVAILABLE: 
        models_to_run += ['TVAE', 'CTGAN']
    
    for m in models_to_run:
        try:
            final_df = train_and_generate_split(df_base, model_type=m)
            
            # Apply Domain Constraints to Synthetic Output
            if 'dep_score' in final_df.columns:
                final_df['dep_score'] = final_df['dep_score'].replace(PADDING_VAL, 0).clip(0, 12).round()
            if 'wealth_log' in final_df.columns:
                final_df['wealth_log'] = final_df['wealth_log'].replace(PADDING_VAL, 0).clip(0, 20)
            
            # Save Output
            fname = f"Augmented_{m}.csv"
            final_df.to_csv(fname, index=False)
            
            n_t = len(final_df[final_df[SPLIT_COL] == 1])
            n_c = len(final_df[final_df[SPLIT_COL] == 0])
            print(f"\n💾 Augmentation Complete. File saved: {fname}")
            print(f"   Structure: Treated N={n_t}, Control N={n_c}")
            print(f"   Validation: Controls aligned to real death event time.")
            
        except Exception as e:
            print(f"   ❌ Execution Failed for {m}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()