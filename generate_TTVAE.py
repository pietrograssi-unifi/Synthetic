"""
Description:
    This script implements a TARNet-augmented Causal Transformer VAE (C-TTVAE)
    to augment longitudinal clinical data via G-Computation and ITE Injection.

    Features:
    - Explicit Time-Treatment Interaction (DiD Feature) to prevent Constant Offset.
    - TARNet Dual-Head Architecture to isolate causal differentials.
    - Strict LOCF anchoring to prevent Time Leakage.

Author: Pietro Grassi
Date: February 2026
"""

import os
import sys
import warnings
import traceback
import random
from typing import List, Dict, Optional, Tuple, Any

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. REPRODUCIBILITY SETUP
# ==============================================================================
RANDOM_SEED = 2026

def set_all_seeds(seed: int = 2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

M_IMPUTATIONS = 10
EPOCHS_FINAL = 150
N_TUNING_TRIALS = 15     
EPOCHS_TUNING = 40

ID_COL = 'survivor_id'
TIME_COL = 'rel_time'
SPLIT_COL = 'treat_group'
PADDING_VAL = -999.0

SKELETON_STATIC = [
    'wave_death', 'death_year', 'is_female', 'country', 'control_eligible', 
    'cause_death', 'edu_level', 'living_area_cat', 'treat_group', 'is_treated',
    'has_cancer', 'has_neuro', 'has_organ', 'gender_imp', 'hc125_num'
]

SKELETON_ANCHORS = ['dep_anchor', 'wealth_anchor']
SKELETON_START = ['age', 'wave', 'int_year']

DYNAMIC_VARS = [
    'rel_time', 'dep_score', 'wealth_log', 'adl_score', 
    'maxgrip', 'sphus_imp', 'satisfaction_health'
]

DETERMINISTIC_RULES = {'age': 2, 'wave': 1, 'int_year': 2}
REL_TIME_GRID = [-3, -2, -1, 0, 1, 2]
MAX_WAVES = len(REL_TIME_GRID)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. ARCHITETTURA RETE NEURALE: INTERACTIVE TARNet C-TTVAE
# ==============================================================================

class CausalTTVAE_Model(nn.Module):
    def __init__(self, dyn_dim: int, stat_dim: int, seq_len: int, treat_idx: int, 
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2, 
                 latent_dim: int = 64, dropout: float = 0.1):
        super(CausalTTVAE_Model, self).__init__()
        self.seq_len = seq_len  
        self.d_model = d_model  
        self.dyn_dim = dyn_dim
        self.stat_dim = stat_dim
        self.treat_idx = treat_idx 
        
        # FIX: +1 Dimensione per l'inserimento della feature DiD (Active Treatment)
        self.stat_dim_aug = stat_dim + 1
        
        self.embedding = nn.Linear(dyn_dim + self.stat_dim_aug, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        self.fc_z_to_seq = nn.Linear(latent_dim + self.stat_dim_aug, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.head_cont = nn.Linear(d_model + self.stat_dim_aug, dyn_dim) 
        self.head_dep_0 = nn.Linear(d_model + self.stat_dim_aug, 1)
        self.head_dep_1 = nn.Linear(d_model + self.stat_dim_aug, 1)
        self.head_mask = nn.Linear(d_model + self.stat_dim_aug, 1)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(DEVICE)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)

    def _get_augmented_stat(self, x_stat: torch.Tensor):
        """Crea la feature dinamica 'Active Treatment' (Treat * Post)"""
        x_stat_expanded = x_stat.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Estrae il trattamento (-1 per controllo, +1 per trattato a causa del MinMaxScaler)
        t = x_stat[:, self.treat_idx].unsqueeze(1).unsqueeze(2) 
        
        # Crea maschera temporale (0 per t < 0, 1 per t >= 0)
        # Indici 0,1,2 corrispondono a t=-3,-2,-1. Indici 3,4,5 a t=0,1,2
        time_mask = torch.zeros(1, self.seq_len, 1).to(DEVICE)
        time_mask[:, 3:, :] = 1.0
        
        # Active Treatment = 0 nel passato. Nel futuro è +1 (Trattato) o -1 (Controllo)
        active_treat = t * time_mask 
        
        x_stat_aug = torch.cat([x_stat_expanded, active_treat], dim=-1)
        return x_stat_aug, t

    def forward(self, x_dyn: torch.Tensor, x_stat: torch.Tensor):
        x_stat_aug, t = self._get_augmented_stat(x_stat)
        
        enc_input = self.embedding(torch.cat([x_dyn, x_stat_aug], dim=-1)) + self.pos_encoder[:, :self.seq_len, :]
        x_enc = self.transformer_encoder(enc_input, mask=self._generate_causal_mask(self.seq_len))
        
        mu, logvar = self.fc_mu(x_enc), self.fc_logvar(x_enc)
        logvar = torch.clamp(logvar, min=-3.0, max=5.0) # Previene mode collapse
        z = self.reparameterize(mu, logvar)
        
        dec_input = self.fc_z_to_seq(torch.cat([z, x_stat_aug], dim=-1)) + self.pos_encoder[:, :self.seq_len, :]
        x_dec = self.transformer_decoder(dec_input, mask=self._generate_causal_mask(self.seq_len))
        
        out_features = torch.cat([x_dec, x_stat_aug], dim=-1)
        
        mask_1 = (t > 0).float()
        mask_0 = (t <= 0).float()
        
        pred_0 = self.head_dep_0(out_features)
        pred_1 = self.head_dep_1(out_features)
        pred_dep = pred_1 * mask_1 + pred_0 * mask_0
        
        return self.head_cont(out_features), pred_dep, self.head_mask(out_features), mu, logvar

    def decode(self, z: torch.Tensor, x_stat: torch.Tensor):
        x_stat_aug, t = self._get_augmented_stat(x_stat)
        
        dec_input = self.fc_z_to_seq(torch.cat([z, x_stat_aug], dim=-1)) + self.pos_encoder[:, :self.seq_len, :]
        x_dec = self.transformer_decoder(dec_input, mask=self._generate_causal_mask(self.seq_len))
        
        out_features = torch.cat([x_dec, x_stat_aug], dim=-1)
        
        mask_1 = (t > 0).float()
        mask_0 = (t <= 0).float()
        
        pred_0 = self.head_dep_0(out_features)
        pred_1 = self.head_dep_1(out_features)
        pred_dep = pred_1 * mask_1 + pred_0 * mask_0
        
        return self.head_cont(out_features), pred_dep, self.head_mask(out_features)


class TTVAE_Wrapper:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.encoders = {}
        self.best_params = {}
        
    def _create_tensors(self, data, static_cols, prob_cols, n_visits):
        data_matrix = self.scaler.transform(data.replace(PADDING_VAL, 0))
        static_indices = [self.feature_names.index(c) for c in static_cols if c in self.feature_names]
        
        seq_dyn, seq_stat, seq_dep, seq_mask = [], [], [], []
        for i in range(len(data)):
            row_raw, row_scaled = data.iloc[i], data_matrix[i]
            seq_stat.append(row_scaled[static_indices])
            
            step_dyn, step_dep, step_mask = [], [], []
            for v in range(n_visits):
                check_col = f"{prob_cols[0]}_v{v}" if f"{prob_cols[0]}_v{v}" in row_raw else f"dep_score_v{v}"
                
                try:
                    is_pad = float(row_raw[check_col]) <= PADDING_VAL + 1
                except (ValueError, TypeError):
                    is_pad = True
                    
                step_mask.append([0.0 if is_pad else 1.0])
                
                dep_col = f"dep_score_v{v}"
                try:
                    dep_val = float(row_raw[dep_col]) if not is_pad else 0.0
                except (ValueError, TypeError):
                    dep_val = 0.0
                step_dep.append(dep_val) 
                
                step_vals = []
                for pc in prob_cols:
                    col_name = f"{pc}_v{v}"
                    step_vals.append(0.0 if is_pad else row_scaled[self.feature_names.index(col_name)] if col_name in self.feature_names else 0.0)
                step_dyn.append(step_vals)
                
            seq_dyn.append(step_dyn); seq_dep.append(step_dep); seq_mask.append(step_mask)
            
        return map(lambda x: torch.FloatTensor(np.array(x, dtype=np.float32)).to(DEVICE), [seq_dyn, seq_stat, seq_dep, seq_mask])

    def tune_and_fit(self, df_wide: pd.DataFrame, n_visits: int, static_cols: List[str], prob_cols: List[str]):
        self.static_cols, self.prob_cols = static_cols, prob_cols
        data = df_wide.copy()
        
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        self.feature_names = list(data.columns)
        
        self.treat_idx = static_cols.index('treat_group')
        self.scaler.fit(data.replace(PADDING_VAL, 0))
        
        t_dyn, t_stat, t_dep, t_mask = self._create_tensors(data, static_cols, prob_cols, n_visits)
        dataset = TensorDataset(t_dyn, t_stat, t_dep, t_mask)
        
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        print(f"\n   [AUTO-TUNING] Starting Non-Parametric Optimization ({N_TUNING_TRIALS} Trials)...")
        best_val_loss = float('inf')
        best_hyperparams = None
        
        search_space = {
            'latent_dim': [16, 32],
            'd_model': [64, 128],
            'nhead': [2, 4],
            'num_layers': [1, 2],
            'lr': [1e-3, 2e-3],
            'batch_size': [64, 128],
            'dropout': [0.1, 0.2]
        }
        
        mse_loss = nn.MSELoss(reduction='none')
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        for trial in range(N_TUNING_TRIALS):
            params = {k: random.choice(v) for k, v in search_space.items()}
            
            model = CausalTTVAE_Model(dyn_dim=t_dyn.shape[2], stat_dim=t_stat.shape[1], seq_len=n_visits, 
                                      treat_idx=self.treat_idx, 
                                      d_model=params['d_model'], nhead=params['nhead'], 
                                      num_layers=params['num_layers'], latent_dim=params['latent_dim'], 
                                      dropout=params['dropout']).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
            train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
            
            model.train()
            for epoch in range(EPOCHS_TUNING):
                for b_dyn, b_stat, b_dep, b_mask in train_loader:
                    optimizer.zero_grad()
                    r_cont, r_dep, r_mask, mu, logvar = model(b_dyn, b_stat)
                    
                    l_cont = (mse_loss(r_cont, b_dyn) * b_mask).mean() 
                    l_dep = (mse_loss(r_dep.squeeze(-1), b_dep) * b_mask.squeeze(-1)).mean()
                    l_mask = bce_loss(r_mask, b_mask).mean()
                    l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_dyn.size(0)
                    
                    loss = (l_cont * 10.0) + l_dep + (l_mask * 5.0) + (0.001 * l_kld)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            val_loss_tot = 0
            with torch.no_grad():
                for b_dyn, b_stat, b_dep, b_mask in val_loader:
                    r_cont, r_dep, r_mask, mu, logvar = model(b_dyn, b_stat)
                    l_cont = (mse_loss(r_cont, b_dyn) * b_mask).mean() 
                    l_dep = (mse_loss(r_dep.squeeze(-1), b_dep) * b_mask.squeeze(-1)).mean()
                    l_mask = bce_loss(r_mask, b_mask).mean()
                    l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_dyn.size(0)
                    val_loss_tot += ((l_cont * 10.0) + l_dep + (l_mask * 5.0) + (0.001 * l_kld)).item()
            
            val_loss_avg = val_loss_tot / len(val_loader)
            print(f"      Trial {trial+1} | Val Loss: {val_loss_avg:.4f} | Params: {params}")
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_hyperparams = params

        self.best_params = best_hyperparams
        print(f"   [AUTO-TUNING] Best Architecture Found: {self.best_params}")
        
        print(f"   [TRAINING] Fitting Final TTVAE Model on 100% of data for {EPOCHS_FINAL} epochs...")
        self.model = CausalTTVAE_Model(dyn_dim=t_dyn.shape[2], stat_dim=t_stat.shape[1], seq_len=n_visits, 
                                       treat_idx=self.treat_idx, 
                                       d_model=self.best_params['d_model'], nhead=self.best_params['nhead'], 
                                       num_layers=self.best_params['num_layers'], latent_dim=self.best_params['latent_dim'],
                                       dropout=self.best_params['dropout']).to(DEVICE)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params['lr'], weight_decay=1e-5)
        full_loader = DataLoader(dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        
        self.model.train()
        for epoch in range(EPOCHS_FINAL):
            total_loss = 0
            for b_dyn, b_stat, b_dep, b_mask in full_loader:
                optimizer.zero_grad()
                r_cont, r_dep, r_mask, mu, logvar = self.model(b_dyn, b_stat)
                
                l_cont = (mse_loss(r_cont, b_dyn) * b_mask).mean() 
                l_dep = (mse_loss(r_dep.squeeze(-1), b_dep) * b_mask.squeeze(-1)).mean()
                l_mask = bce_loss(r_mask, b_mask).mean()
                l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_dyn.size(0)
                
                loss = (l_cont * 10.0) + l_dep + (l_mask * 5.0) + (0.001 * l_kld)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 50 == 0: 
                print(f"      Epoch {epoch+1}/{EPOCHS_FINAL} | Final ELBO Loss: {total_loss / len(full_loader):.4f}")


# ==============================================================================
# 3. HELPER PER IL PROCESSO DATI
# ==============================================================================

def prepare_wide_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    print("   [PROCESS] Transforming to Wide Format on strict rel_time grid...")
    if 'rel_time' not in df.columns and 'wave' in df.columns and 'wave_death' in df.columns:
        df['rel_time'] = df['wave'] - df['wave_death']
        
    grouped = df.groupby(ID_COL)
    wide_rows = []
    
    for survivor_id, group in grouped:
        row = {ID_COL: survivor_id}
        first = group.iloc[0]
        
        for col in SKELETON_STATIC + SKELETON_ANCHORS + SKELETON_START:
            if col in df.columns: row[col] = first[col]

        for col in SKELETON_START:
            if col in df.columns: row[f"{col}_start"] = first[col]
        
        for i, t_val in enumerate(REL_TIME_GRID):
            suffix = f"_v{i}"
            step_data = group[group['rel_time'] == t_val]
            if not step_data.empty:
                for col in DYNAMIC_VARS:
                    if col in step_data.columns: row[col + suffix] = step_data.iloc[0][col]
            else:
                for col in DYNAMIC_VARS:
                    row[col + suffix] = PADDING_VAL
                    
        wide_rows.append(row)
    return pd.DataFrame(wide_rows), MAX_WAVES

def preprocess_causal_trends(df: pd.DataFrame) -> pd.DataFrame:
    print("   [PREP] removing aggressive binning to preserve variance...")
    df_smooth = df.copy()
    if 'dep_score' in df_smooth.columns: 
        df_smooth['dep_score'] = df_smooth['dep_score'].clip(0, 12).round() 
    return df_smooth

def compute_anchors_simple(df: pd.DataFrame) -> pd.DataFrame:
    if 'rel_time' not in df.columns:
        if 'wave_death' in df.columns: df['rel_time'] = df['wave'] - df['wave_death']
        else: return df 
        
    anchors = df[df['rel_time'] == -1][['survivor_id', 'dep_score', 'wealth_log']]
    anchors = anchors.rename(columns={'dep_score': 'dep_anchor', 'wealth_log': 'wealth_anchor'})
    
    if 'dep_anchor' not in df.columns:
        df = df.merge(anchors, on='survivor_id', how='left')
        
    past_data = df[df['rel_time'] < 0].sort_values(['survivor_id', 'rel_time'])
    last_known_past = past_data.groupby('survivor_id').last().reset_index()
    
    if 'dep_anchor' in df.columns:
        df['dep_anchor'] = df['dep_anchor'].fillna(df['survivor_id'].map(last_known_past.set_index('survivor_id')['dep_score']))
        df['wealth_anchor'] = df['wealth_anchor'].fillna(df['survivor_id'].map(last_known_past.set_index('survivor_id')['wealth_log']))
        
        prior_mean_dep = past_data['dep_score'].mean()
        prior_mean_wealth = past_data['wealth_log'].mean()
        
        df['dep_anchor'] = df['dep_anchor'].fillna(prior_mean_dep)
        df['wealth_anchor'] = df['wealth_anchor'].fillna(prior_mean_wealth)
        
    return df

# ==============================================================================
# 4. PIPELINE DI CAUSAL INJECTION
# ==============================================================================

def get_clean_val(val):
    try:
        v = float(val)
        return v if not np.isnan(v) and v > (PADDING_VAL + 10) else None
    except (ValueError, TypeError):
        return None

def generate_imputations_from_model(model, df_base_real, df_wide_real, m_imputations: int, gamma_perturbation: float = 0.0):
    print(f"\n   [SAMPLING] True Individual Treatment Effect (ITE) Injection via TARNet...")
    m_datasets = []
    
    original_ids = df_wide_real[ID_COL].values 
    data_factual = df_wide_real.copy()
    
    for col, le in model.encoders.items():
        if col in data_factual.columns:
            known = set(le.classes_)
            data_factual[col] = data_factual[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            data_factual[col] = le.transform(data_factual[col])
            
    mat_factual = model.scaler.transform(data_factual.replace(PADDING_VAL, 0))
    stat_idx = [model.feature_names.index(c) for c in model.static_cols if c in model.feature_names]
    x_stat_factual = torch.FloatTensor(mat_factual[:, stat_idx]).to(DEVICE)
    
    seq_dyn = []
    for i in range(len(data_factual)):
        step_dyn = []
        for v in range(model.model.seq_len):
            step_vals = [mat_factual[i][model.feature_names.index(f"{pc}_v{v}")] if f"{pc}_v{v}" in model.feature_names else 0.0 for pc in model.prob_cols]
            step_dyn.append(step_vals)
        seq_dyn.append(step_dyn)
    x_dyn_factual = torch.FloatTensor(np.array(seq_dyn)).to(DEVICE)
    
    x_dyn_factual_masked = x_dyn_factual.clone()
    x_dyn_factual_masked[:, 3:, :] = 0.0 
    
    data_counter = data_factual.copy()
    if 'treat_group' in data_counter.columns: 
        data_counter['treat_group'] = 1.0 - data_counter['treat_group'] 
    
    mat_counter = model.scaler.transform(data_counter.replace(PADDING_VAL, 0))
    x_stat_counter = torch.FloatTensor(mat_counter[:, stat_idx]).to(DEVICE)
    
    model.model.train() 
    idx_anchor = 2 
    
    for m in range(m_imputations):
        current_seed = RANDOM_SEED + m
        set_all_seeds(current_seed)
        
        with torch.no_grad():
            _, _, _, mu, logvar = model.model(x_dyn_factual_masked, x_stat_factual)
            
            if gamma_perturbation > 0.0:
                std = torch.exp(0.5 * logvar)
                mu = mu + (gamma_perturbation * std)
                
            z_factual = model.model.reparameterize(mu, logvar)
            
            recon_cont_f, recon_dep_f, _ = model.model.decode(z_factual, x_stat_factual)
            recon_cont_c, recon_dep_c, _ = model.model.decode(z_factual, x_stat_counter)
            
            recon_dep_f = recon_dep_f.squeeze(-1).cpu().numpy()
            recon_dep_c = recon_dep_c.squeeze(-1).cpu().numpy()
            
            recon_cont_f = recon_cont_f.cpu().numpy()
            recon_cont_c = recon_cont_c.cpu().numpy()
        
        final_matrix_f = np.zeros((len(df_wide_real), len(model.feature_names)))
        final_matrix_c = np.zeros((len(df_wide_real), len(model.feature_names)))
        col_map = {name: idx for idx, name in enumerate(model.feature_names)}
        
        for i in range(len(df_wide_real)):
            for v in range(model.model.seq_len):
                for d_i, col in enumerate(model.prob_cols):
                    full_col = f"{col}_v{v}"
                    if full_col in col_map:
                        final_matrix_f[i, col_map[full_col]] = recon_cont_f[i, v, d_i]
                        final_matrix_c[i, col_map[full_col]] = recon_cont_c[i, v, d_i]
                        
        data_inv_f = model.scaler.inverse_transform(final_matrix_f)
        data_inv_c = model.scaler.inverse_transform(final_matrix_c)
        
        df_cf = df_base_real.copy()
        df_cf['is_synthetic'], df_cf['imputation_id'] = 1, m + 1
        if 'treat_group' in df_cf.columns: df_cf['treat_group'] = 1.0 - df_cf['treat_group'] 
        
        df_cf.set_index([ID_COL, 'rel_time'], inplace=True)
        
        for i in range(len(df_wide_real)):
            surv_id = original_ids[i]
            
            bias_dep_anchor = recon_dep_c[i, idx_anchor] - recon_dep_f[i, idx_anchor]
            
            bias_wealth_anchor = 0
            col_wealth = f"wealth_log_v{idx_anchor}"
            if col_wealth in col_map:
                bias_wealth_anchor = data_inv_c[i, col_map[col_wealth]] - data_inv_f[i, col_map[col_wealth]]
            
            for v, t_val in enumerate(REL_TIME_GRID):
                if (surv_id, t_val) in df_cf.index:
                    if t_val >= 0:
                        emp_dep_col = f"dep_score_v{v}"
                        real_dep = get_clean_val(df_wide_real.loc[i, emp_dep_col] if emp_dep_col in df_wide_real.columns else None)
                        
                        delta_dep_t = recon_dep_c[i, v] - recon_dep_f[i, v]
                        ite_dep = delta_dep_t - bias_dep_anchor
                        
                        if real_dep is not None:
                            synth_dep = real_dep + ite_dep
                        else:
                            synth_dep = recon_dep_c[i, v] - bias_dep_anchor
                            
                        df_cf.loc[(surv_id, t_val), 'dep_score'] = np.clip(np.round(synth_dep), 0, 12)
                        
                        for pc in model.prob_cols:
                            if pc != 'rel_time' and pc != 'dep_score':
                                full_col = f"{pc}_v{v}"
                                if full_col in col_map: 
                                    emp_cont_col = f"{pc}_v{v}"
                                    real_cont = get_clean_val(df_wide_real.loc[i, emp_cont_col] if emp_cont_col in df_wide_real.columns else None)
                                    
                                    val_f = data_inv_f[i, col_map[full_col]]
                                    val_c = data_inv_c[i, col_map[full_col]]
                                    
                                    delta_cont_t = val_c - val_f
                                    bias_cont_anchor = bias_wealth_anchor if pc == 'wealth_log' else 0
                                    ite_cont = delta_cont_t - bias_cont_anchor
                                    
                                    if real_cont is not None:
                                        val = real_cont + ite_cont
                                    else:
                                        val = val_c - bias_cont_anchor
                                        
                                    if pc == 'wealth_log': 
                                        val = np.clip(val, 0, 20)
                                    elif pc in ['adl_score', 'maxgrip', 'sphus_imp', 'satisfaction_health', 'adl_raw', 'maxgrip_raw']: 
                                        val = np.round(val)
                                        
                                    df_cf.loc[(surv_id, t_val), pc] = val
        
        df_cf.reset_index(inplace=True)
        m_datasets.append(df_cf)
            
    return m_datasets

def main():
    print("=== LONGITUDINAL AUGMENTATION PIPELINE ===")

    set_all_seeds(RANDOM_SEED)
    print(f"   [SETUP] Global Random Seed set to: {RANDOM_SEED}")
    
    if not os.path.exists('BaseDataset.csv'): 
        print("❌ Critical Error: Input file 'BaseDataset.csv' not found."); return
        
    df_base = pd.read_csv('BaseDataset.csv')
    df_base['is_synthetic'] = 0
    
    if 'dep_score' in df_base.columns: df_base['dep_score'] = df_base['dep_score'].clip(0, 12)
    if 'wealth_log' in df_base.columns: df_base['wealth_log'] = df_base['wealth_log'].clip(0, 20)
    
    numeric_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_clamp = [c for c in numeric_cols if c not in [ID_COL, 'is_synthetic', 'treat_group']]
    
    real_ranges = {}
    for col in cols_to_clamp:
        valid_vals = df_base[df_base[col] > -990][col] 
        if not valid_vals.empty:
            real_ranges[col] = (valid_vals.min(), valid_vals.max())
        else:
            real_ranges[col] = (0, 1)
            
    print(f"   [SETUP] Constraints calculated for {len(real_ranges)} variables.")

    df_prep = preprocess_causal_trends(df_base)
    df_prep = compute_anchors_simple(df_prep)
    df_wide_real, max_waves = prepare_wide_data(df_prep)

    m = 'TTVAE'
    try:
        skel_cols = [c for c in df_wide_real.columns if c in SKELETON_STATIC + SKELETON_ANCHORS + SKELETON_START + [f"{x}_start" for x in SKELETON_START]]
        
        wrapper = TTVAE_Wrapper()
        wrapper.tune_and_fit(df_wide_real, max_waves, skel_cols, DYNAMIC_VARS)
        trained_model = wrapper
        
        list_of_m_datasets = generate_imputations_from_model(trained_model, df_prep, df_wide_real, M_IMPUTATIONS)
        
        if list_of_m_datasets:
            processed_m_datasets = []
            df_prep_real = df_prep.copy()
            df_prep_real['is_synthetic'] = 0
            
            for idx_m, syn_df in enumerate(list_of_m_datasets):
                df_prep_real['imputation_id'] = idx_m + 1 
                common_cols = df_prep_real.columns.intersection(syn_df.columns)
                df_imputation = pd.concat([df_prep_real[common_cols], syn_df[common_cols]], ignore_index=True)
                processed_m_datasets.append(df_imputation)
            
            final_df = pd.concat(processed_m_datasets, ignore_index=True)
            
            for col in cols_to_clamp:
                if col in final_df.columns:
                    min_v, max_v = real_ranges[col]
                    final_df[col] = final_df[col].clip(lower=min_v, upper=max_v)
                    if col in ['dep_score', 'adl_score', 'maxgrip', 'adl_raw', 'maxgrip_raw', 'sphus_imp', 'satisfaction_health']: 
                        final_df[col] = final_df[col].round()
            
            fname = f"Augmented_{m}_GComp.csv"
            final_df.to_csv(fname, index=False)
            print(f"   💾 Saved G-Computation Dataset: {fname} (Total rows: {len(final_df)})\n")
            
            # SENSITIVITY
            print(f"\n   [SENSITIVITY] Executing Deep Latent Perturbation (Gamma = 0.5) for TTVAE...")
            list_stress_datasets = generate_imputations_from_model(trained_model, df_prep, df_wide_real, M_IMPUTATIONS, gamma_perturbation=0.5)
                    
            if list_stress_datasets:
                processed_stress_datasets = []
                df_prep_real = df_prep.copy()
                df_prep_real['is_synthetic'] = 0
                        
                for idx_m, syn_df in enumerate(list_stress_datasets):
                    df_prep_real['imputation_id'] = idx_m + 1 
                    common_cols = df_prep_real.columns.intersection(syn_df.columns)
                    df_imputation = pd.concat([df_prep_real[common_cols], syn_df[common_cols]], ignore_index=True)
                    processed_stress_datasets.append(df_imputation)
                    
                df_stress_full = pd.concat(processed_stress_datasets, ignore_index=True)
                        
                for col in cols_to_clamp:
                    if col in df_stress_full.columns:
                        min_v, max_v = real_ranges[col]
                        df_stress_full[col] = df_stress_full[col].clip(lower=min_v, upper=max_v)
                        if col in ['dep_score', 'adl_score', 'maxgrip', 'adl_raw', 'maxgrip_raw', 'sphus_imp', 'satisfaction_health']: 
                            df_stress_full[col] = df_stress_full[col].round()
                        
                stress_file = f"Augmented_{m}_StressTest.csv"
                df_stress_full.to_csv(stress_file, index=False)
                print(f"   ☣️ Poisoned Dataset successfully saved: {stress_file}")

    except Exception as e:
        print(f"   ❌ Execution Failed for {m}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()