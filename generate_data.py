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
# Nel tuo script, cerca questa lista e assicurati che contenga le ancore:
SKELETON_STATIC = [
    'wave_death', 'death_year', 'is_female', 'country', 'control_eligible', 
    'cause_death', 'edu_level', 'living_area_cat', 
    'dep_anchor', 'wealth_anchor'
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
# 2. ARCHITETTURA RETE NEURALE (Causal TTVAE)
# ==============================================================================

class CausalTTVAE_Model(nn.Module):
    """
    Transformer-based Time-series Variational Autoencoder (TTVAE) con Mascheramento Causale.
    
    OBIETTIVO: Imparare la distribuzione complessa dei dati longitudinali per generare
    nuove traiettorie sintetiche realistiche.
    
    COMPONENTI CHIAVE:
    1. Transformer Encoder: Comprime la sequenza temporale in una rappresentazione astratta.
    2. Causal Mask: Impedisce al modello di "vedere il futuro" durante l'addestramento.
       Al tempo 't', il modello può usare solo le informazioni da 0 a t.
    3. Latent Space (VAE): Introduce probabilisticità per permettere la generazione.
    4. Transformer Decoder: Ricostruisce la sequenza temporale partendo dallo spazio latente.
    """
    def __init__(self, input_dim: int, seq_len: int, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, latent_dim: int = 64):
        super(CausalTTVAE_Model, self).__init__()
        self.seq_len = seq_len  # Lunghezza massima della sequenza (es. 7 wave)
        self.d_model = d_model  # Dimensione interna del Transformer (es. 128 neuroni)
        
        # 1. Input Projection
        # Trasforma le feature grezze (input_dim) in vettori di dimensione d_model
        # necessari per il Transformer.
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding Imparabile
        # I Transformer non hanno nozione di "ordine" (a differenza delle RNN).
        # Aggiungiamo un vettore che dice "questo è il passo t=1", "questo è t=2", ecc.
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # 3. Encoder Transformer
        # Analizza le relazioni temporali complesse tra gli step.
        # batch_first=True significa che l'input è (Batch, Time, Features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Spazio Latente (VAE)
        # L'encoder produce un output piatto. Lo proiettiamo in due vettori:
        # - mu: La media della distribuzione latente.
        # - logvar: Il logaritmo della varianza.
        self.fc_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(d_model * seq_len, latent_dim)
        
        # Per tornare indietro (decoding), proiettiamo z nella dimensione del Transformer
        self.fc_z_to_seq = nn.Linear(latent_dim, d_model * seq_len)
        
        # 5. Decoder Transformer
        # Prende il vettore latente e prova a ricostruire la sequenza temporale originale.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 6. Output Head
        # Riconverte la dimensione interna (d_model) nelle feature originali (input_dim)
        self.output_head = nn.Linear(d_model, input_dim)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """
        Genera una matrice triangolare superiore per mascherare il futuro.
        
        Esempio (sz=3):
        [[0, -inf, -inf],  <- Al tempo 0, vedo solo t0
         [0, 0,    -inf],  <- Al tempo 1, vedo t0 e t1
         [0, 0,    0   ]]  <- Al tempo 2, vedo t0, t1 e t2
         
        I valori '-inf' diventano zero dopo la Softmax, annullando l'attenzione sul futuro.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passaggio in avanti dell'Encoder."""
        batch_size = x.size(0)
        
        # Aggiunta encoding posizionale
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        
        # Maschera causale: fondamentale per preservare la logica temporale
        mask = self._generate_causal_mask(x.size(1))
        x = self.transformer_encoder(x, mask=mask)
        
        # Appiattimento per passare ai layer lineari del VAE
        x = x.reshape(batch_size, -1) 
        
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Trucco della riparametrizzazione (Reparameterization Trick).
        
        Per poter calcolare il gradiente attraverso un campionamento casuale,
        invece di campionare Z direttamente da N(mu, var), facciamo:
        Z = mu + sigma * epsilon
        dove epsilon è rumore standard fisso N(0,1).
        Questo permette alla backpropagation di funzionare.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Passaggio in avanti del Decoder."""
        batch_size = z.size(0)
        
        # Espande il vettore latente z nella forma (Batch, Seq_Len, D_Model)
        x = self.fc_z_to_seq(z)
        x = x.reshape(batch_size, self.seq_len, self.d_model)
        
        # Anche il decoder usa la maschera per mantenere la coerenza autoregressiva
        mask = self._generate_causal_mask(self.seq_len)
        x = self.transformer_decoder(x, mask=mask)
        
        return self.output_head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ciclo completo: Input -> Encoder -> Latent -> Decoder -> Output"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class TTVAE_Wrapper:
    """
    Classe Wrapper per gestire la complessità dei dati.
    Si occupa di trasformare i DataFrame Pandas in Tensori PyTorch, addestrare il modello
    e gestire il campionamento (generazione).
    """
    def __init__(self, epochs: int = 200, batch_size: int = 128, latent_dim: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) # Normalizzazione essenziale per le reti neurali
        self.encoders = {}
        self.model = None
        
    def fit(self, df_wide: pd.DataFrame, n_visits: int, static_cols: List[str], prob_cols: List[str]):
        """
        Prepara i dati e addestra il modello.
        Converte il DataFrame 'Wide' (1 riga per paziente) in un Tensore 3D.
        Struttura Tensore: (Numero Pazienti, Numero Visite, Numero Features)
        """
        self.static_cols = static_cols
        self.prob_cols = prob_cols
        
        # 1. Encoding Variabili Categoriche (Stringhe -> Numeri)
        data = df_wide.copy()
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        self.feature_names = list(data.columns)
        
        # 2. Scaling (MinMax tra -1 e 1)
        # Sostituiamo il padding (-999) con 0 temporaneamente per lo scaling
        data_for_fit = data.replace(PADDING_VAL, 0)
        self.scaler.fit(data_for_fit)
        data_matrix = self.scaler.transform(data.replace(PADDING_VAL, 0))
        
        static_indices = [self.feature_names.index(c) for c in static_cols if c in self.feature_names]
        
        # 3. Costruzione Sequenze (Creazione del Tensore 3D)
        # Qui uniamo le variabili statiche (ripetute) con quelle dinamiche (che cambiano nel tempo)
        sequences = []
        for i in range(len(data)):
            row_raw = data.iloc[i] 
            row_scaled = data_matrix[i]
            static_vals = row_scaled[static_indices]
            
            seq_steps = []
            for v in range(n_visits):
                step_vals = []
                
                # Gestione Padding: Se il dato originale era -999, creiamo una maschera
                # La maschera (ultima colonna) varrà 0 se è padding, 1 se è dato reale.
                check_col = f"{prob_cols[0]}_v{v}"
                if check_col not in row_raw: check_col = f"dep_score_v{v}" # Fallback
                
                is_padding = False
                if check_col in row_raw:
                     val_orig = row_raw[check_col]
                     # Check robusto per il padding
                     try:
                         if float(val_orig) <= PADDING_VAL + 1: is_padding = True
                     except: pass
                
                mask_val = 0.0 if is_padding else 1.0
                
                for pc in prob_cols:
                    col_name = f"{pc}_v{v}"
                    if col_name in self.feature_names:
                        idx = self.feature_names.index(col_name)
                        val = row_scaled[idx]
                        if is_padding: val = 0.0 # Se è padding, forza a 0 nel tensore
                        step_vals.append(val)
                    else: 
                        step_vals.append(0.0)
                
                # Step finale = [Statiche, Dinamiche al tempo t, Maschera]
                full_step = np.concatenate([static_vals, step_vals, [mask_val]])
                seq_steps.append(full_step)
            sequences.append(seq_steps)
            
        self.X_train = np.array(sequences, dtype=np.float32)
        
        # 4. Inizializzazione e Training Loop
        input_dim = self.X_train.shape[2]
        self.model = CausalTTVAE_Model(input_dim=input_dim, seq_len=n_visits, latent_dim=self.latent_dim).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss(reduction='none') 
        
        tensor_x = torch.FloatTensor(self.X_train).to(DEVICE)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        print(f"   [TTVAE] Modello Inizializzato. Input Dimension: {input_dim}")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                
                # Loss di Ricostruzione (MSE)
                loss_mse = loss_fn(recon_x, x)
                
                # Ponderazione della Loss:
                # Diamo un peso molto alto (5.0) all'ultima feature (la maschera di padding)
                # Questo insegna al modello a capire esattamente QUANDO un paziente deve "uscire" (morire/censura).
                weights = torch.ones_like(loss_mse)
                weights[:, :, -1] = 5.0 
                recon_loss = (loss_mse * weights).mean()
                
                # KL Divergence: Regolarizza lo spazio latente per renderlo una normale standard
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                # Loss Totale = Ricostruzione + peso * KLD
                loss = recon_loss + 0.002 * kld_loss 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 50 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs} | Loss: {total_loss / len(loader):.4f}")

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Genera nuovi pazienti sintetici.
        1. Campiona vettori casuali Z dalla distribuzione Normale(0, 1).
        2. Il Decoder trasforma Z in sequenze temporali.
        3. Post-processing per riconvertire i numeri in categorie e gestire il padding.
        """
        self.model.eval()
        with torch.no_grad():
            # 1. Generazione dallo spazio latente
            z = torch.randn(n_samples, self.latent_dim).to(DEVICE)
            recon_seq = self.model.decode(z).cpu().numpy()
            
        static_indices = [self.feature_names.index(c) for c in self.static_cols if c in self.feature_names]
        n_static = len(static_indices)
        n_dyn = len(self.prob_cols)
        
        final_matrix = np.zeros((n_samples, len(self.feature_names)))
        col_map = {name: i for i, name in enumerate(self.feature_names)}
        padding_masks = np.zeros((n_samples, recon_seq.shape[1]))
        
        # 2. Ricostruzione della Matrice Dati
        for i in range(n_samples):
            # Le variabili statiche sono prese dal primo step (sono uguali per tutti gli step)
            static_vec = recon_seq[i, 0, :n_static]
            for s_i, col_idx in enumerate(static_indices):
                final_matrix[i, col_idx] = static_vec[s_i]
            
            for v in range(recon_seq.shape[1]):
                step = recon_seq[i, v]
                dyn_vals = step[n_static : n_static+n_dyn]
                mask_val = step[-1] # L'ultima feature è la predizione della maschera
                padding_masks[i, v] = mask_val
                
                for d_i, col in enumerate(self.prob_cols):
                    full_col = f"{col}_v{v}"
                    if full_col in col_map:
                        final_matrix[i, col_map[full_col]] = dyn_vals[d_i]
                        
        # 3. Inverse Scaling (Da [-1, 1] ai valori reali)
        data_inv = self.scaler.inverse_transform(final_matrix)
        df_syn = pd.DataFrame(data_inv, columns=self.feature_names)
        
        # 4. Post-processing finale
        for col in df_syn.columns:
            if col in self.encoders:
                # Decodifica le categorie (es. 0.8 -> 1 -> "Male")
                le = self.encoders[col]
                df_syn[col] = df_syn[col].round().clip(0, len(le.classes_)-1).astype(int)
                df_syn[col] = le.inverse_transform(df_syn[col])
            else:
                # Arrotonda variabili intere (es. conteggio malattie)
                if any(x in col for x in ['has_', 'is_', '_imp', 'n_waves']):
                     df_syn[col] = df_syn[col].round()
        
        # 5. Applicazione Maschera di Padding
        # Se il modello ha predetto maschera < 0.5, considera quel punto come "non esistente"
        for i in range(n_samples):
            for v in range(recon_seq.shape[1]):
                if padding_masks[i, v] < 0.5:
                    for pc in self.prob_cols:
                        col_name = f"{pc}_v{v}"
                        if col_name in df_syn.columns:
                            df_syn.at[i, col_name] = PADDING_VAL
        return df_syn


# ==============================================================================
# 3. HELPER PER IL PROCESSO DATI (Data Processing)
# ==============================================================================

def prepare_wide_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Trasforma i dati longitudinali (formato Lungo) in formato Wide (Largo).
    
    PERCHÉ SERVE:
    Le reti neurali standard e i modelli come VAE/GAN preferiscono ricevere 
    un singolo vettore fisso per ogni paziente, invece di N righe sparse.
    
    INPUT (Lungo):
    ID | Wave | Dep_Score
    1  | 1    | 5
    1  | 2    | 6
    
    OUTPUT (Largo):
    ID | Dep_Score_v0 | Dep_Score_v1 | ...
    1  | 5            | 6            | ...
    """
    print("   [PROCESS] Transforming to Wide Format...")
    
    # 1. Calcolo Tempo Relativo (Centrato sulla Morte)
    # Fondamentale per confrontare "pere con pere": allinea i pazienti rispetto
    # al loro evento di morte (t=0) invece che all'anno solare (es. 2015).
    if 'rel_time' not in df.columns and 'wave' in df.columns and 'wave_death' in df.columns:
        df['rel_time'] = df['wave'] - df['wave_death']
        
    # Ordiniamo per ID e Tempo per garantire che v0 sia prima di v1
    df_sorted = df.sort_values(by=[ID_COL, 'wave'])
    grouped = df_sorted.groupby(ID_COL)
    
    wide_rows = []
    # Troviamo la lunghezza massima di una storia clinica nel dataset reale
    max_waves_obs = df[ID_COL].value_counts().max()
    
    for survivor_id, group in grouped:
        row = {}
        # Salviamo quanto è lunga la storia vera di questo paziente
        row['n_waves_real'] = len(group)
        
        # Prendiamo la prima riga del gruppo per estrarre le variabili statiche
        # (Sesso, Educazione, ecc. non cambiano nel tempo)
        first = group.iloc[0]
        
        # --- A. Variabili Statiche ---
        for col in SKELETON_STATIC + STATIC_GEN + [SPLIT_COL]:
            if col in df.columns: row[col] = first[col]
            
        # --- B. Valori Iniziali per variabili Deterministiche ---
        # Il modello non predice l'età a ogni step (sarebbe difficile),
        # ma predice l'età iniziale. Poi noi aggiungeremo +2 anni matematicamente.
        for col in SKELETON_START: 
            if col in first: row[f"{col}_start"] = first[col]
            
        # --- C. Variabili Dinamiche (Appiattimento) ---
        # Creiamo colonne _v0, _v1, _v2... fino a max_waves_obs
        for i in range(max_waves_obs):
            suffix = f"_v{i}"
            
            if i < len(group):
                # CASO 1: Il paziente è VIVO a questo step
                # Copiamo i valori reali (es. depressione, ricchezza)
                for col in DYNAMIC_VARS: 
                    if col in group.columns: row[col + suffix] = group.iloc[i][col]
            else:
                # CASO 2: Il paziente è MORTO o uscito dallo studio
                # Riempiamo con PADDING_VAL (-999).
                # Il modello imparerà che dopo un certo punto arrivano solo -999.
                for col in DYNAMIC_VARS: row[col + suffix] = PADDING_VAL
                
        wide_rows.append(row)
        
    return pd.DataFrame(wide_rows), max_waves_obs


def reconstruct_long_data(df_wide: pd.DataFrame, max_waves: int) -> pd.DataFrame:
    """
    Operazione Inversa: Da Wide (Sintetico) a Long (Analizzabile).
    
    LOGICA CRUCIALE:
    Il modello genera una riga lunga con valori e padding (-999).
    Questa funzione legge la riga e si ferma quando incontra il padding,
    simulando così la morte del paziente sintetico.
    """
    print("   [PROCESS] Reconstructing Long Format...")
    long_rows = []
    
    for idx, row in df_wide.iterrows():
        syn_id = f"SYN_{idx}"
        
        # Controllo Validità: Se il primo step (v0) è già padding, 
        # il modello ha generato un "paziente fantasma". Lo scartiamo.
        if row.get(f"{DYNAMIC_VARS[0]}_v0", 0) <= PADDING_VAL + 1: continue

        # 1. Recupero Variabili Statiche (Uguali per tutte le righe di questo ID)
        static_vals = {col: row[col] for col in (SKELETON_STATIC + STATIC_GEN) if col in row}
        if SPLIT_COL in row: static_vals[SPLIT_COL] = row[SPLIT_COL]
        
        # 2. Inizializzazione Contatori Deterministici
        # (Prendiamo Age_start, Wave_start, ecc.)
        curr_vals = {}
        for col in DETERMINISTIC_RULES.keys():
            if f"{col}_start" in row: curr_vals[col] = row[f"{col}_start"]
        
        # Determiniamo quanto deve essere lunga la sequenza
        n_waves_pred = int(row.get('n_waves_real', max_waves))
        n_waves_pred = max(1, min(max_waves, n_waves_pred))
        
        # 3. Srotolamento della Sequenza (Unrolling)
        for i in range(n_waves_pred):
            # STOP CONDIZIONALE: Se incontriamo -999, fermiamo la generazione.
            # Questo ricrea la struttura "staggered" (pazienti che muoiono a tempi diversi).
            if row.get(f"{DYNAMIC_VARS[0]}_v{i}", 0) <= PADDING_VAL + 1: break
            
            long_row = {}
            long_row[ID_COL] = syn_id
            long_row.update(static_vals) # Aggiunge sesso, educazione...
            long_row.update(curr_vals)   # Aggiunge età corrente, anno corrente...
            
            # Copia i valori dinamici generati dal modello (es. depressione)
            for col in DYNAMIC_VARS:
                long_row[col] = row.get(f"{col}_v{i}", 0)
            
            # 4. Calcolo Dinamico di 'rel_time'
            # Invece di far predire rel_time al modello, lo calcoliamo matematicamente:
            # Rel_Time = Wave_Corrente - Wave_Morte_Statica
            # Questo garantisce coerenza logica perfetta.
            if 'wave' in curr_vals and 'wave_death' in static_vals:
                long_row['rel_time'] = curr_vals['wave'] - static_vals['wave_death']
            
            long_rows.append(long_row)
            
            # 5. Aggiornamento Regole Deterministiche per il prossimo giro
            # Es: Age diventa Age + 2, Wave diventa Wave + 1
            for col, step in DETERMINISTIC_RULES.items():
                if col in curr_vals: curr_vals[col] += step
                
    return pd.DataFrame(long_rows)


def robust_sample(model, n_required: int) -> pd.DataFrame:
    """
    Campionamento Robusto.
    
    PROBLEMA: I modelli generativi (specie i GAN) a volte falliscono e generano 
    righe fatte solo di -999 (padding).
    
    SOLUZIONE: Questa funzione chiede al modello PIÙ dati del necessario (1.5x),
    filtra via quelli non validi, e si ferma solo quando ha n_required dati buoni.
    """
    valid_samples = []
    attempts = 0
    
    # Continua a provare finché non abbiamo abbastanza dati o raggiungiamo 10 tentativi
    while len(valid_samples) < n_required and attempts < 10:
        n_missing = n_required - len(valid_samples)
        
        # Chiediamo un extra (+50% + 10) per compensare scarti
        batch = model.sample(int(n_missing * 1.5) + 10)
        
        # Controllo qualità: verifichiamo che la prima colonna non sia -999
        check_col = f"{DYNAMIC_VARS[0]}_v0"
        if check_col not in batch.columns: check_col = "dep_score_v0" # fallback
        
        if check_col in batch.columns:
            # Teniamo solo le righe dove il valore è > -990 (quindi dati reali)
            batch = batch[batch[check_col] > (PADDING_VAL + 10)]
            
        if len(batch) > 0: valid_samples.append(batch)
        attempts += 1
        
    # Concatena tutto e taglia esattamente al numero richiesto
    return pd.concat(valid_samples).iloc[:n_required] if valid_samples else pd.DataFrame()


def preprocess_causal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulizia finale pre-training.
    Applica vincoli di dominio (es. la depressione Euro-D deve stare tra 0 e 12).
    Serve per aiutare il modello a non dover imparare vincoli ovvi da zero.
    """
    print("   [PREP] applying pre-processing filters...")
    df_smooth = df.copy()
    if 'dep_score' in df_smooth.columns: 
        # Clip forza i valori a restare nel range [0, 12]
        df_smooth['dep_score'] = df_smooth['dep_score'].clip(0, 12)
    return df_smooth

def compute_anchors_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola le ancore basilari per l'iniezione nello scheletro."""
    if 'rel_time' not in df.columns:
        if 'wave_death' in df.columns: df['rel_time'] = df['wave'] - df['wave_death']
        else: return df # Non possiamo calcolare nulla
        
    # Prendi t=-1
    anchors = df[df['rel_time'] == -1][['survivor_id', 'dep_score', 'wealth_log']]
    anchors = anchors.rename(columns={'dep_score': 'dep_anchor', 'wealth_log': 'wealth_anchor'})
    
    # Unisci al df originale
    if 'dep_anchor' not in df.columns:
        df = df.merge(anchors, on='survivor_id', how='left')
        
    # Imputazione media per chi non ha t=-1
    if 'dep_anchor' in df.columns:
        df['dep_anchor'] = df['dep_anchor'].fillna(df['dep_score'].mean())
        df['wealth_anchor'] = df['wealth_anchor'].fillna(df['wealth_log'].mean())
        
    return df


# ==============================================================================
# 4. PIPELINE DI ESECUZIONE PRINCIPALE (Main Execution)
# ==============================================================================

def train_and_generate_split(df_base: pd.DataFrame, model_type: str = 'TTVAE') -> pd.DataFrame:
    print(f"\n=== Processing Model Architecture: {model_type} ===")
    
    # --- Calcoliamo le ancore, così finiscono nello scheletro ---
    df_base = compute_anchors_simple(df_base)
    
    # Trasforma in Wide
    df_wide, max_waves = prepare_wide_data(df_base)
    syn_dfs = []
    
    n_treated_orig = len(df_wide[df_wide[SPLIT_COL] == 1])
    target_n = int(n_treated_orig * AUGMENTATION_FACTOR)
    
    for group_val in [0, 1]: 
        df_group = df_wide[df_wide[SPLIT_COL] == group_val].copy()
        
        if len(df_group) < 10: 
            print(f"   [WARNING] Insufficient data for Group {group_val}. Skipping.")
            continue
        
        print(f"   [GROUP {group_val}] Training Samples: {len(df_group)} -> Target Generation: {target_n}")
        train_data = df_group.drop(columns=[SPLIT_COL], errors='ignore')
        
        # --- SKELETON EXTRACTION ---
        skel_static_cols = [c for c in (SKELETON_STATIC + STATIC_GEN) if c in train_data.columns]
        skel_start_cols = [f"{c}_start" for c in SKELETON_START if f"{c}_start" in train_data.columns]
        
        real_skeleton = train_data[skel_static_cols + skel_start_cols].values
        features_static = skel_static_cols + skel_start_cols
        
        syn_data = None
        
        # --- TRAINING ---
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
                print(f"   [ERROR] {model_type} failed: {e}")

        # --- SKELETON INJECTION (Reinserisce le ancore reali nei dati sintetici) ---
        if syn_data is not None and len(real_skeleton) > 0:
            indices = np.random.choice(len(real_skeleton), size=len(syn_data), replace=True)
            sampled_skel = real_skeleton[indices]
            
            # Sovrascriviamo le colonne statiche (incluse le ancore)
            for i, col in enumerate(skel_static_cols + skel_start_cols):
                if col in syn_data.columns: 
                    syn_data[col] = sampled_skel[:, i]
            
            syn_data[SPLIT_COL] = group_val
            syn_dfs.append(syn_data)
            
    if not syn_dfs: return pd.DataFrame()
    
    df_syn_wide = pd.concat(syn_dfs, ignore_index=True)
    return reconstruct_long_data(df_syn_wide, max_waves)


def main():
    print("=== LONGITUDINAL AUGMENTATION PIPELINE (V34 - FINAL DYNAMIC FIX) ===")
    
    if not os.path.exists('BaseDataset.csv'): 
        print("❌ Critical Error: Input file 'BaseDataset.csv' not found."); return
        
    df_base = pd.read_csv('BaseDataset.csv')
    df_base['is_synthetic'] = 0
    
    # 1. Pre-processing iniziale sui dati REALI
    if 'dep_score' in df_base.columns: df_base['dep_score'] = df_base['dep_score'].clip(0, 12)
    if 'wealth_log' in df_base.columns: df_base['wealth_log'] = df_base['wealth_log'].clip(0, 20)
    
    # --- SETUP DYNAMIC CLAMPING ---
    # Calcoliamo i range validi (Min, Max) dai dati reali per OGNI colonna numerica.
    # Questo creerà una "gabbia" che impedirà al CTGAN di generare -999 o valori assurdi.
    numeric_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_clamp = [c for c in numeric_cols if c not in [ID_COL, 'is_synthetic', 'treat_group']]
    
    real_ranges = {}
    for col in cols_to_clamp:
        # Nota: calcoliamo min/max ignorando eventuali -999 se presenti per errore nel reale
        valid_vals = df_base[df_base[col] > -990][col] 
        if not valid_vals.empty:
            real_ranges[col] = (valid_vals.min(), valid_vals.max())
        else:
            real_ranges[col] = (0, 1) # Fallback
            
    print(f"   [SETUP] Constraints calculated for {len(real_ranges)} variables.")

    df_prep = preprocess_causal_trends(df_base)
    
    models_to_run = ['TTVAE']
    if CTGAN_AVAILABLE: models_to_run += ['TVAE', 'CTGAN']
    
    for m in models_to_run:
        try:
            # Generazione
            syn_df = train_and_generate_split(df_prep, model_type=m)
            
            if not syn_df.empty:
                syn_df['is_synthetic'] = 1
                
                # --- APPLICAZIONE CLAMPING ---
                print(f"   [POST-PROC] Fixing MAPE: Clamping {m} output to real ranges...")
                
                for col in cols_to_clamp:
                    if col in syn_df.columns:
                        min_v, max_v = real_ranges[col]
                        
                        # 1. FIX PADDING (-999): Sostituiamo i -999 con il valore minimo reale
                        # (es. per una dummy 0/1, -999 diventa 0)
                        syn_df.loc[syn_df[col] <= -990, col] = min_v
                        
                        # 2. FIX OUTLIER: Clippiamo tutto nel range reale
                        syn_df[col] = syn_df[col].clip(lower=min_v, upper=max_v)
                        
                        # 3. FIX INTERI: Arrotondiamo se necessario
                        if col in ['dep_score', 'hc125_num', 'eurod_imp'] or 'has_' in col:
                             syn_df[col] = syn_df[col].round()

                # Merge Finale
                common_cols = df_base.columns.intersection(syn_df.columns)
                final_df = pd.concat([df_base[common_cols], syn_df[common_cols]], ignore_index=True)
                
                print(f"   [MERGE] Combined Real ({len(df_base)}) + Synthetic ({len(syn_df)})")
            else:
                final_df = df_base.copy()
                print("   [WARN] No synthetic data generated.")

            # Salvataggio
            fname = f"Augmented_v1new_{m}.csv"
            final_df.to_csv(fname, index=False)
            print(f"\n💾 Hybrid Dataset Saved: {fname}")
            
        except Exception as e:
            print(f"   ❌ Execution Failed for {m}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()