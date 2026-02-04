"""
LONGITUDINAL DATA AUGMENTATION FRAMEWORK (v45)
==============================================
Methodology: Delta Learning with Causal Transformers

Description:
    This script implements an Autoregressive Time-series Variational Autoencoder (TTVAE)
    to generate synthetic longitudinal trajectories for palliative care research.
    
    Major Methodological Innovation (v45):
    To strictly adhere to the 'Parallel Trends' assumption required for Difference-in-Differences 
    analysis, this version implements 'Delta Learning':
    1. Anchoring: A baseline state (t=-1) is established for every subject.
    2. Delta Training: The model learns the *change* (Delta) in value relative to the anchor,
       rather than the absolute value.
    3. Reconstruction: Absolute values are recovered post-generation (Anchor + Generated Delta).
    
    This approach stabilizes the starting point of generated trajectories, preventing
    initial divergences between treated and control groups.

Author: Pietro Grassi
Date: February 2026
"""

import os
import sys
import warnings
import traceback
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Optional Dependency Check
try:
    from ctgan import CTGAN, TVAE
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================

# Training Configuration
AUGMENTATION_FACTOR = 1.0
EPOCHS_TTVAE = 200 
EPOCHS_GAN = 200 
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 64

# Data Schema
ID_COL = 'survivor_id'
SPLIT_COL = 'treat_group'
PADDING_VAL = -999.0

# Temporal Horizon (Fixed Matrix Strategy)
# We model waves 6 through 9 to capture the end-of-life trajectory.
WAVE_RANGE = [6, 7, 8, 9]
MAX_SEQ_LEN = len(WAVE_RANGE)

# Static Variables (Invariant)
SKELETON_STATIC = [
    'wave_death', 'death_year', 'is_female', 'country', 'control_eligible', 
    'cause_death', 'edu_level', 'living_area_cat', 'satisfaction_health',
    'dep_anchor', 'wealth_anchor'  # Anchors included in static context
]

# Initialization Variables (t=0)
SKELETON_START = ['age', 'wave', 'int_year'] 

# Generated Static Attributes
STATIC_GEN = ['gender_imp', 'hc125_num', 'is_treated']

# Delta Mapping: Defines which variables are modeled relative to an anchor
DELTA_VARS_MAP = {
    'dep_score': 'dep_anchor',
    'wealth_log': 'wealth_anchor'
}

# Dynamic Variables (Time-Variant)
DYNAMIC_VARS = [
    'rel_time', 
    'income_imp', 'wealth_imp', 'sphus_imp', 'adl_imp', 'maxgrip_imp', 
    'eurod_imp', 'dep_score', 'wealth_log', 'adl_raw', 'adl_score', 'maxgrip',
    'ins_sat', 'health_sat',
    'has_cancer', 'has_neuro', 'has_organ',
    'ph006d1', 'ph006d4', 'ph006d6', 'ph006d10', 'ph006d12', 'ph006d16', 'ph006d21'
]

# Deterministic Evolution Rules
DETERMINISTIC_RULES = {'age': 2, 'wave': 1, 'int_year': 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. NEURAL ARCHITECTURE: AUTOREGRESSIVE TTVAE
# ==============================================================================

class AutoregressiveTTVAE(nn.Module):
    """
    Transformer-based Time-series VAE with Autoregressive Decoding.
    
    Architecture:
    1. Encoder: Bidirectional Transformer encoding the full sequence into Latent Z.
    2. Latent Space: Reparameterization trick (Z ~ N(mu, sigma)).
    3. Decoder: Causal Transformer Decoder that reconstructs the sequence step-by-step,
       conditioned on Z and the previous time step (t-1).
    """
    
    

    def __init__(self, input_dim: int, seq_len: int, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, latent_dim: int = 64):
        super(AutoregressiveTTVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim 
        
        # --- Encoder ---
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # --- Latent Bottleneck ---
        self.fc_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * seq_len, latent_dim)
        
        # --- Decoder ---
        self.dec_embedding = nn.Linear(input_dim, d_model)
        self.z_projection = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, input_dim)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """Generates upper-triangular mask to prevent look-ahead bias in decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_emb = self.enc_embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x_enc = self.transformer_encoder(x_emb)
        x_flat = x_enc.reshape(x.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x_input: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes sequence conditioned on Latent Z and input history.
        """
        seq_emb = self.dec_embedding(x_input) + self.pos_encoder[:, :x_input.size(1), :]
        
        # Project Z and inject it into the sequence (Global Conditioning)
        z_proj = self.z_projection(z).unsqueeze(1)
        decoder_input = seq_emb + z_proj 
        
        tgt_mask = self._generate_causal_mask(x_input.size(1))
        output = self.transformer_decoder(decoder_input, decoder_input, tgt_mask=tgt_mask)
        return self.output_head(output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Autoregressive Training: Shift input right (Start Token)
        batch_size, seq_len, feats = x.size()
        start_token = torch.zeros(batch_size, 1, feats).to(DEVICE)
        shifted_input = torch.cat([start_token, x[:, :-1, :]], dim=1)
        
        recon_x = self.decode(shifted_input, z)
        return recon_x, mu, logvar

    def generate(self, n_samples: int) -> np.ndarray:
        """Iterative autoregressive generation loop."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(DEVICE)
            curr_seq = torch.zeros(n_samples, 1, self.input_dim).to(DEVICE)
            
            # Step-by-step generation
            for t in range(self.seq_len):
                full_out = self.decode(curr_seq, z)
                next_step = full_out[:, -1:, :]
                if t < self.seq_len - 1:
                    curr_seq = torch.cat([curr_seq, next_step], dim=1)
            return full_out.cpu().numpy()


class TTVAE_Wrapper:
    """Wrapper for data preprocessing, training loop management, and inference."""
    
    def __init__(self, epochs=200, batch_size=128, latent_dim=64):
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
        """Formats data into (Batch, Time, Feats) and runs training loop."""
        self.static_cols = static_cols
        self.prob_cols = prob_cols
        
        data = df_wide.copy()
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        self.feature_names = list(data.columns)
        
        # Scaling
        data_for_fit = data.replace(PADDING_VAL, 0)
        self.scaler.fit(data_for_fit)
        data_matrix = self.scaler.transform(data.replace(PADDING_VAL, 0))
        
        static_indices = [self.feature_names.index(c) for c in static_cols if c in self.feature_names]
        
        # Construct Sequences
        sequences = []
        for i in range(len(data)):
            row_scaled = data_matrix[i]
            static_vals = row_scaled[static_indices]
            seq_steps = []
            for v in range(n_visits):
                step_vals = []
                for pc in prob_cols:
                    col_name = f"{pc}_v{v}"
                    idx = self.feature_names.index(col_name)
                    step_vals.append(row_scaled[idx])
                
                # Append validity mask (1.0 = valid)
                full_step = np.concatenate([static_vals, step_vals, [1.0]])
                seq_steps.append(full_step)
            sequences.append(seq_steps)
            
        self.X_train = np.array(sequences, dtype=np.float32)
        
        # Initialize Model
        input_dim = self.X_train.shape[2]
        self.model = AutoregressiveTTVAE(input_dim=input_dim, seq_len=n_visits, latent_dim=self.latent_dim).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss(reduction='none') 
        
        tensor_x = torch.FloatTensor(self.X_train).to(DEVICE)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        print(f"   [AR-TTVAE] Training started (Delta Learning Mode)...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                
                # MSE Loss
                loss_mse = loss_fn(recon_x, x)
                weights = torch.ones_like(loss_mse)
                weights[:, :, -1] = 1.0 # Mask weight
                recon_loss = (loss_mse * weights).mean()
                
                # KLD Loss
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                loss = recon_loss + 0.001 * kld_loss 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 50 == 0:
                print(f"      Epoch {epoch+1}: Loss {total_loss / len(loader):.4f}")

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generates synthetic samples and inverse transforms scaling."""
        recon_seq = self.model.generate(n_samples)
        
        static_indices = [self.feature_names.index(c) for c in self.static_cols if c in self.feature_names]
        n_static = len(static_indices)
        n_dyn = len(self.prob_cols)
        
        final_matrix = np.zeros((n_samples, len(self.feature_names)))
        col_map = {name: i for i, name in enumerate(self.feature_names)}
        
        for i in range(n_samples):
            static_vec = recon_seq[i, 0, :n_static]
            for s_i, col_idx in enumerate(static_indices):
                final_matrix[i, col_idx] = static_vec[s_i]
            
            for v in range(recon_seq.shape[1]):
                step = recon_seq[i, v]
                dyn_vals = step[n_static : n_static+n_dyn]
                for d_i, col in enumerate(self.prob_cols):
                    full_col = f"{col}_v{v}"
                    if full_col in col_map:
                        final_matrix[i, col_map[full_col]] = dyn_vals[d_i]
                        
        data_inv = self.scaler.inverse_transform(final_matrix)
        df_syn = pd.DataFrame(data_inv, columns=self.feature_names)
        
        # Post-processing / Decoding
        for col in df_syn.columns:
            if col in self.encoders:
                le = self.encoders[col]
                df_syn[col] = df_syn[col].round().clip(0, len(le.classes_)-1).astype(int)
                df_syn[col] = le.inverse_transform(df_syn[col])
            else:
                if any(x in col for x in ['has_', 'is_', '_imp']):
                     df_syn[col] = df_syn[col].round()
        return df_syn


# ==============================================================================
# 3. HELPERS: DELTA TRANSFORMATION & ANCHORING
# ==============================================================================

def compute_anchors(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Computes t=-1 anchors for Delta calculation.
    
    The Anchor is the value of the variable at the wave prior to the sequence start.
    This creates a personalized baseline for every survivor, ensuring
    trajectories start from the correct intercept.
    """
    
    

    # Establish Relative Time if missing
    if 'rel_time' not in df_group.columns:
        if 'wave_death' in df_group.columns:
            df_group['rel_time'] = df_group['wave'] - df_group['wave_death']
        else:
            df_group['rel_time'] = -1 

    # Extract Baseline (t = -1)
    baseline_slice = df_group[df_group['rel_time'] == -1]
    
    dep_anchors = baseline_slice[['survivor_id', 'dep_score']].rename(columns={'dep_score': 'dep_anchor'})
    wealth_anchors = baseline_slice[['survivor_id', 'wealth_log']].rename(columns={'wealth_log': 'wealth_anchor'})
    
    # Merge Anchors
    anchors = pd.merge(dep_anchors, wealth_anchors, on='survivor_id', how='outer')
    
    # Ensure all IDs are present (even if missing baseline)
    unique_ids_arr = df_group['survivor_id'].unique()
    unique_ids_df = pd.DataFrame(unique_ids_arr, columns=['survivor_id'])
    
    df_merged = pd.merge(unique_ids_df, anchors, on='survivor_id', how='left')
    
    # Impute missing anchors with population mean
    mean_dep = df_group['dep_score'].mean()
    mean_wealth = df_group['wealth_log'].mean()
    df_merged['dep_anchor'] = df_merged['dep_anchor'].fillna(mean_dep)
    df_merged['wealth_anchor'] = df_merged['wealth_anchor'].fillna(mean_wealth)
    
    return df_merged


def prepare_wide_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]: 
    """
    Transforms Long-format data into a Wide-matrix of DELTAS.
    
    Transformation:
        Delta_t = Value_t - Anchor
    """
    print("   [PROCESS] Transforming to Wide Format (Delta Mode)...")
    if 'rel_time' not in df.columns: df['rel_time'] = df['wave'] - df['wave_death']
    
    # 1. Compute and Inject Anchors
    anchor_df = compute_anchors(df)
    df = pd.merge(df, anchor_df, on='survivor_id', how='left')
    
    # 2. Convert Target Variables to Deltas
    df_delta = df.copy()
    for col, anchor in DELTA_VARS_MAP.items():
        df_delta[col] = df_delta[col] - df_delta[anchor]
    
    df_sorted = df_delta.sort_values(by=[ID_COL, 'wave'])
    grouped = df_sorted.groupby(ID_COL)
    wide_rows = []
    
    template_waves = pd.DataFrame({'wave': WAVE_RANGE})
    
    # 3. Flatten to Wide Format
    for survivor_id, group in grouped:
        row = {}
        first = group.iloc[0]
        
        # Static Features
        for col in SKELETON_STATIC + STATIC_GEN + [SPLIT_COL]:
            if col in df.columns: row[col] = first[col]
        for col in SKELETON_START: 
            if col in first: row[f"{col}_start"] = first[col]
            
        # Merge with Template (Imputation Strategy)
        group_filled = pd.merge(template_waves, group, on='wave', how='left')
        
        # Impute DELTAS (Forward Fill)
        cols_to_fill = DYNAMIC_VARS
        group_filled[cols_to_fill] = group_filled[cols_to_fill].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        for i, w_val in enumerate(WAVE_RANGE):
            suffix = f"_v{i}"
            curr_step = group_filled.iloc[i]
            for col in DYNAMIC_VARS:
                if col in curr_step: row[col + suffix] = curr_step[col]
                else: row[col + suffix] = 0
        wide_rows.append(row)
        
    return pd.DataFrame(wide_rows), MAX_SEQ_LEN


def reconstruct_long_data(df_wide: pd.DataFrame, max_waves: int) -> pd.DataFrame:
    """
    Reconstructs Long-format data from synthetic Wide-matrix.
    
    Crucial Step:
        Value_t = Anchor + Delta_t
    Includes safety clipping to prevent numerical instability.
    """
    print("   [PROCESS] Reconstructing Long Format & Applying Anchors...")
    long_rows = []
    
    for idx, row in df_wide.iterrows():
        syn_id = f"SYN_{idx}"
        
        # Recover Static Features (including Anchors)
        static_vals = {col: row[col] for col in (SKELETON_STATIC + STATIC_GEN) if col in row}
        if SPLIT_COL in row: static_vals[SPLIT_COL] = row[SPLIT_COL]
        
        # Recover Deterministic Start values
        curr_vals = {}
        for col in DETERMINISTIC_RULES.keys():
            if f"{col}_start" in row: curr_vals[col] = row[f"{col}_start"]
        
        # Unroll Sequence
        for i, wave_val in enumerate(WAVE_RANGE):
            long_row = {}
            long_row[ID_COL] = syn_id
            long_row['wave'] = wave_val
            long_row.update(static_vals)
            long_row.update(curr_vals)
            
            for col in DYNAMIC_VARS:
                val_delta = row.get(f"{col}_v{i}", 0)
                
                # --- DELTA RECONSTRUCTION LOGIC ---
                if col in DELTA_VARS_MAP:
                    anchor_col = DELTA_VARS_MAP[col]
                    anchor_val = static_vals.get(anchor_col, 0)
                    
                    # SAFETY CLIP: Restrict Delta magnitude to prevent unrealistic outliers
                    if col == 'dep_score': val_delta = np.clip(val_delta, -12, 12)
                    if col == 'wealth_log': val_delta = np.clip(val_delta, -10, 10)
                    
                    # Reconstruct Absolute Value
                    long_row[col] = anchor_val + val_delta
                else:
                    long_row[col] = val_delta
            
            if 'wave_death' in static_vals:
                long_row['rel_time'] = wave_val - static_vals['wave_death']
            
            long_rows.append(long_row)
            
            # Evolve deterministic vars
            for col, step in DETERMINISTIC_RULES.items():
                if col in curr_vals: curr_vals[col] += step
                
    return pd.DataFrame(long_rows)


def robust_sample(model, n_required: int) -> pd.DataFrame:
    """Safely samples slightly more than required to create a buffer."""
    batch = model.sample(int(n_required * 1.1))
    return batch.iloc[:n_required]


def preprocess_causal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain constraints to pre-trends."""
    print("   [PREP] Applying domain-specific pre-processing...")
    df_smooth = df.copy()
    if 'dep_score' in df_smooth.columns: 
        df_smooth['dep_score'] = df_smooth['dep_score'].clip(0, 12)
    return df_smooth


# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def train_and_generate_split(df_base: pd.DataFrame, model_type: str = 'TTVAE') -> pd.DataFrame:
    """
    Orchestrates the Split-Training strategy.
    Treated and Control groups are trained separately to preserve unique causal dynamics.
    """
    print(f"\n=== Processing Model Architecture: {model_type} ===")
    
    # 1. Prepare Data (calculate Anchors & Deltas)
    df_wide, max_waves = prepare_wide_data(df_base)
    
    syn_dfs = []
    # Calculate augmentation target based on Treated group size
    target_n = int(len(df_wide[df_wide[SPLIT_COL] == 1]) * AUGMENTATION_FACTOR)
    
    for group_val in [0, 1]: 
        df_group = df_wide[df_wide[SPLIT_COL] == group_val].copy()
        
        if len(df_group) < 10: 
            print(f"   [WARN] Insufficient data for Group {group_val}. Skipping.")
            continue
            
        print(f"   [GROUP {group_val}] Training N={len(df_group)} -> Target Generation={target_n}")
        train_data = df_group.drop(columns=[SPLIT_COL], errors='ignore')
        
        # Identify Static/Skeleton columns for injection
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
                discrete_cols = [c for c in train_data.columns if train_data[c].dtype == 'object' or train_data[c].nunique() < 20]
                if model_type == 'CTGAN': model = CTGAN(epochs=EPOCHS_GAN, verbose=True)
                else: model = TVAE(epochs=EPOCHS_GAN)
                model.fit(train_data, discrete_columns=discrete_cols)
                syn_data = robust_sample(model, target_n)
            except Exception as e:
                print(f"   [ERROR] {model_type}: {e}")

        # --- SKELETON INJECTION ---
        # Inject real static attributes to maintain demographic realism
        if syn_data is not None and len(real_skeleton) > 0:
            indices = np.random.choice(len(real_skeleton), size=len(syn_data), replace=True)
            sampled_skel = real_skeleton[indices]
            for i, col in enumerate(skel_static_cols + skel_start_cols):
                if col in syn_data.columns: syn_data[col] = sampled_skel[:, i]
            
            syn_data[SPLIT_COL] = group_val
            syn_dfs.append(syn_data)
            
    if not syn_dfs: return pd.DataFrame()
    
    # 2. Reconstruct (Anchor + Delta)
    return reconstruct_long_data(pd.concat(syn_dfs, ignore_index=True), max_waves)


def main():
    print("=== LONGITUDINAL AUGMENTATION V45 (DELTA LEARNING) ===")
    
    if not os.path.exists('BaseDataset.csv'): 
        print("❌ Critical Error: BaseDataset.csv not found."); return
        
    df_base = pd.read_csv('BaseDataset.csv')
    
    # Global Pre-processing Constraints
    if 'dep_score' in df_base.columns: df_base['dep_score'] = df_base['dep_score'].clip(0, 12)
    if 'wealth_log' in df_base.columns: df_base['wealth_log'] = df_base['wealth_log'].clip(0, 20)
    
    df_base = preprocess_causal_trends(df_base)
    
    models_to_run = ['TTVAE']
    if CTGAN_AVAILABLE: models_to_run += ['TVAE', 'CTGAN']
    
    for m in models_to_run:
        try:
            final_df = train_and_generate_split(df_base, model_type=m)
            
            # Final Safety Clipping
            if 'dep_score' in final_df.columns: 
                final_df['dep_score'] = final_df['dep_score'].replace(PADDING_VAL, 0).clip(0, 12).round()
            if 'wealth_log' in final_df.columns: 
                final_df['wealth_log'] = final_df['wealth_log'].replace(PADDING_VAL, 0).clip(0, 20)
            if 'income_imp' in final_df.columns: 
                final_df['income_imp'] = final_df['income_imp'].replace(PADDING_VAL, 0).clip(0, 200000)
            
            fname = f"Augmented_{m}.csv"
            final_df.to_csv(fname, index=False)
            print(f"\n💾 Augmentation Complete. Saved: {fname}")
            
        except Exception as e:
            print(f"   ❌ Error executing {m}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()