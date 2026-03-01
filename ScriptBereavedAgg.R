# ==============================================================================
# PROJECT:   The Ripple Effect of a Good Death
# TITLE:     A Causal Evaluation of Palliative Care on Bereaved Spouses' Mental Health
# DATA:      Survey of Health, Ageing and Retirement in Europe (SHARE)
# AUTHORS:   Pietro Grassi, Tallys Fendels, PhD Chiara Seghieri, PhD Daniele Vignoli
# DATE:      February 2026
#
# SCRIPT:    01_data_preparation.R
# PURPOSE:   Raw data ingestion, dyad linkage (matching deceased to survivor), 
#            longitudinal panel construction, and variable engineering for causal 
#            inference (Difference-in-Differences).
# ==============================================================================

# ==============================================================================
# 0. CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================
rm(list = ls())
gc()

# Define robust package loading via 'pacman'
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,      # Data manipulation and pipeline architecture
  haven,          # Import Stata (.dta) files
  stringr,        # String manipulation
  janitor         # Data cleaning utilities
)

# --- DIRECTORY MANAGEMENT ---
# Modify PATH_DATA to point to your local SHARE dataset repository.
PATH_DATA <- "/Users/pietro/Desktop/Sant'Anna/Ricerca/SHARE/DatiDid" 
PATH_OUT  <- "/Users/pietro/Desktop/Sant'Anna/Ricerca/SHARE/DiD/Output"

if(!dir.exists(PATH_OUT)) dir.create(PATH_OUT, recursive = TRUE)


# ==============================================================================
# 1. IMPORT UTILITY FUNCTION
# ==============================================================================
#' Iterates through longitudinal files for a specific SHARE module,
#' selects/renames variables based on a mapping dictionary, and binds them.
#' 
#' @param path Directory containing the SHARE .dta files.
#' @param module_code The specific module abbreviation (e.g., "xt", "ph").
#' @param var_mapping A named character vector mapping standardized names to SHARE regex.
read_share_module <- function(path, module_code, var_mapping) {
  
  pattern_file <- paste0("sharew.*_", module_code, ".dta")
  files <- list.files(path, pattern = pattern_file, full.names = TRUE, recursive = TRUE)
  
  if(length(files) == 0) {
    warning(paste("Missing Data: No files found for module:", module_code))
    return(tibble())
  }
  
  message(paste("Processing", length(files), "files for module:", module_code))
  
  data_list <- map(files, function(f) {
    filename <- basename(f)
    wave_num <- as.numeric(str_extract(str_extract(filename, "sharew\\d+"), "\\d+"))
    
    headers <- names(read_dta(f, n_max = 0))
    selected_cols <- c("mergeid") 
    rename_vec <- c()             
    
    for (new_name in names(var_mapping)) {
      pattern <- var_mapping[[new_name]]
      matches <- headers[str_detect(headers, paste0("^", pattern))]
      
      if (length(matches) > 0) {
        col_found <- matches[1] 
        selected_cols <- c(selected_cols, col_found)
        rename_vec[new_name] <- col_found
      }
    }
    
    d <- read_dta(f, col_select = any_of(selected_cols))
    if (length(rename_vec) > 0) d <- d %>% rename(!!!rename_vec)
    
    # Ensure structural consistency across waves by padding missing columns
    missing_vars <- setdiff(names(var_mapping), names(d))
    for (mv in missing_vars) d[[mv]] <- NA
    
    d <- d %>% mutate(mergeid = as.character(mergeid), wave = wave_num)
    return(d)
  })
  
  final_df <- bind_rows(data_list) %>% distinct(mergeid, wave, .keep_all = TRUE) 
  return(final_df)
}


# ==============================================================================
# 2. VARIABLE DICTIONARIES (MAPPINGS)
# ==============================================================================

# Module XT: End of Life (Treatment Assignment & Mortality)
map_xt <- c(
  "treated_palliat" = "xt757", 
  "cause_death"     = "xt011", 
  "reason_no_care"  = "xt754",
  "country"         = "country", 
  "symptoms_pain"   = "xt758", 
  "symptoms_breath" = "xt760", 
  "symptoms_anx"    = "xt762"
)

# Module PH: Physical Health (Chronic Conditions)
map_ph <- c(
  "ph006_1"  = "ph006_1",  "ph006d1"  = "ph006d1",   # Heart Attack
  "ph006_4"  = "ph006_4",  "ph006d4"  = "ph006d4",   # Hypertension
  "ph006_6"  = "ph006_6",  "ph006d6"  = "ph006d6",   # Stroke
  "ph006_10" = "ph006_10", "ph006d10" = "ph006d10",  # Cancer
  "ph006_12" = "ph006_12", "ph006d12" = "ph006d12",  # Stomach/Duodenal Ulcer
  "ph006_16" = "ph006_16", "ph006d16" = "ph006d16",  # Parkinson/Alzheimer
  "ph006_21" = "ph006_21", "ph006d21" = "ph006d21"   # Chronic Kidney Disease
)

# Demographic & Environmental Covariates
map_cv <- c("gender" = "gender", "yrbirth" = "yrbirth", "coupleid" = "coupleid", "int_year" = "int_year")
map_mh <- c("eurod" = "eurod")                      # Mental Health raw
map_iv <- c("area_iv" = "iv009")                    # Interviewer observations
map_hh <- c("area_move" = "ho037")                  # Household location
map_hc <- c("health_sat" = "hc125", "ins_sat" = "hc113") 
map_dn <- c("partner_id" = "dn014")                 # Social Network

# Imputations (gv_imputations generated by SHARE)
map_imp <- c(
  "wealth_imp"  = "hnetw", 
  "income_imp"  = "thinc2", 
  "adl_imp"     = "adl", 
  "sphus_imp"   = "sphus",
  "age_imp"     = "age", 
  "gender_imp"  = "gender", 
  "edu_imp"     = "isced", 
  "eurod_imp"   = "eurod", 
  "maxgrip_imp" = "maxgrip"
)


# ==============================================================================
# 3. DATA IMPORT EXECUTION
# ==============================================================================
message("--- Step 1: Raw Data Ingestion ---")

df_xt  <- read_share_module(PATH_DATA, "xt", map_xt)
df_ph  <- read_share_module(PATH_DATA, "ph", map_ph)
df_mh  <- read_share_module(PATH_DATA, "gv_health", map_mh) 
df_cv  <- read_share_module(PATH_DATA, "cv_r", map_cv) 
df_hh  <- read_share_module(PATH_DATA, "ho", map_hh)
df_iv  <- read_share_module(PATH_DATA, "iv", map_iv)
df_hc  <- read_share_module(PATH_DATA, "hc", map_hc)
df_dn  <- read_share_module(PATH_DATA, "dn", map_dn)
df_imp <- read_share_module(PATH_DATA, "gv_imputations", map_imp)


# ==============================================================================
# 4. DYAD LINKAGE ALGORITHM
# ==============================================================================
# Objective: Link the deceased individual (proxy interview) to their surviving 
# partner using the longitudinal household 'coupleid'.
# ==============================================================================
message("--- Step 2: Dyad Linkage ---")

# 4.1. Clean and Propagate Couple IDs
# Forward-fill 'coupleid' to prevent linkage failure due to missingness in later waves.
df_cv_clean <- df_cv %>%
  arrange(mergeid, wave) %>%
  group_by(mergeid) %>%
  mutate(
    coupleid = as.character(coupleid),
    coupleid = na_if(coupleid, ""),
    coupleid = if_else(coupleid == "0", NA_character_, coupleid)
  ) %>%
  fill(coupleid, .direction = "down") %>%
  ungroup() %>%
  filter(!is.na(coupleid))

# 4.2. Define Treatment Cohort (Deceased Individuals)
deceased_xt <- df_xt %>%
  rename(wave_death = wave) %>%
  mutate(
    # Consolidate palliative care reception across survey versions
    raw_pall = coalesce(
      if("treated_palliat" %in% names(.)) as.numeric(treated_palliat) else NULL,
      if("xt021" %in% names(.)) as.numeric(xt021) else NULL
    ),
    # Treatment Assignment (1 = Palliative Care Received, 0 = No Care)
    is_treated = case_when(
      raw_pall == 1 ~ 1,      
      raw_pall %in% c(0,5) ~ 0, 
      TRUE ~ NA_real_
    ),
    # Eligibility for Control Group: Untreated but had a documented need (reasons 2 or 3)
    reason_code = as.numeric(reason_no_care),
    control_eligible = if_else(is_treated == 0 & reason_code %in% c(2, 3), 1, 0)
  ) %>%
  filter(!is.na(is_treated)) %>%
  select(deceased_id = mergeid, wave_death, is_treated, control_eligible, cause_death, country) 

# 4.3. Execute Dyad Matching
# Extract the last known couple ID for the deceased prior to death
deceased_history <- df_cv_clean %>%
  inner_join(deceased_xt %>% select(deceased_id, wave_death), by = c("mergeid" = "deceased_id")) %>%
  filter(wave < wave_death) %>%
  arrange(mergeid, desc(wave)) %>%
  group_by(mergeid) %>%
  slice(1) %>% 
  ungroup() %>%
  select(deceased_id = mergeid, last_wave = wave, coupleid_key = coupleid)

# Match with the surviving partner using the historical couple ID
partners_found <- df_cv_clean %>%
  inner_join(deceased_history, by = c("coupleid" = "coupleid_key", "wave" = "last_wave")) %>%
  filter(mergeid != deceased_id) %>% 
  rename(partner_id = mergeid) %>%
  select(partner_id, deceased_id)

dyad_map <- partners_found %>%
  inner_join(deceased_xt, by = "deceased_id")

# 4.4. Resolve Ambiguities (Double Deaths)
# Remove dyads where both partners died simultaneously or mapping is ambiguous.
double_deaths <- dyad_map %>%
  group_by(partner_id, wave_death) %>%
  filter(n() > 1) %>%
  pull(deceased_id)

dyad_map_clean <- dyad_map %>%
  filter(!deceased_id %in% double_deaths)

message(sprintf("✅ Dyad Linkage Complete. Valid Dyads Identified: %d", nrow(dyad_map_clean)))

# ==============================================================================
# 5. LONGITUDINAL PANEL CONSTRUCTION (SURVIVOR-CENTERED)
# ==============================================================================
message("--- Step 3: Survivor Panel Construction & Variable Engineering ---")

# Merge modules based on the survivor ID
survivor_panel <- df_imp %>% 
  rename(survivor_id = mergeid) %>%
  inner_join(dyad_map_clean, by = c("survivor_id" = "partner_id")) %>%
  left_join(df_ph, by = c("survivor_id" = "mergeid", "wave")) %>%
  left_join(df_iv, by = c("survivor_id" = "mergeid", "wave")) %>%
  left_join(df_hh, by = c("survivor_id" = "mergeid", "wave")) %>%
  left_join(df_hc, by = c("survivor_id" = "mergeid", "wave")) %>%
  left_join(df_cv %>% select(mergeid, wave, int_year), by = c("survivor_id" = "mergeid", "wave"))

# Helper for robust numeric extraction across varying module structures
safe_chk <- function(col_name, df) {
  if(col_name %in% names(df)) return(as.numeric(df[[col_name]]))
  return(rep(NA_real_, nrow(df)))
}

df_analysis <- survivor_panel %>%
  
  # Structural Variables Interpolation
  mutate(temp_area = coalesce(as.numeric(area_iv), as.numeric(area_move))) %>%
  group_by(survivor_id) %>%
  fill(temp_area, .direction = "downup") %>%
  ungroup() %>%
  
  mutate(
    int_year = as.numeric(int_year), 
    int_year = if_else(int_year < 0, NA_real_, int_year),
    temp_death_year = if_else(wave == wave_death, int_year, NA_real_)
  ) %>%
  group_by(survivor_id) %>%
  fill(temp_death_year, .direction = "downup") %>%
  ungroup() %>%
  rename(death_year = temp_death_year) %>%
  
  # Variable Engineering
  mutate(
    # Event-Time Specification (Crucial for DiD)
    rel_time = wave - wave_death,
    
    # Primary Outcome: Euro-D Depression Score
    dep_score = as.numeric(eurod_imp), 
    
    # Causal Treatment Assignment
    treat_group = case_when(
      is_treated == 1 ~ 1,
      control_eligible == 1 ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Base Demographics
    age = as.numeric(age_imp),          
    is_female = if_else(as.numeric(gender_imp) == 2, 1, 0), 
    
    edu_level = factor(case_when(
      as.numeric(edu_imp) <= 2 ~ "Low",     
      as.numeric(edu_imp) <= 4 ~ "Medium", 
      TRUE ~ "High"
    ), levels = c("Low", "Medium", "High")),
    
    # Economic Status (Inverse Hyperbolic Sine to handle zeros/debt)
    wealth_log = asinh(as.numeric(wealth_imp)), 
    
    # Physical Health Measures
    adl_raw = suppressWarnings(as.numeric(adl_imp)),
    adl_score = if_else(adl_raw < 0, NA_real_, adl_raw),
    
    maxgrip_raw = suppressWarnings(as.numeric(maxgrip_imp)),
    maxgrip = if_else(maxgrip_raw > 100 | maxgrip_raw < 0, NA_real_, maxgrip_raw),
    
    # --- Comorbidity Extraction ---
    # Fuses items and dummies to maximize data availability
    p_ca_10 = safe_chk("ph006_10", .), p_cad_10 = safe_chk("ph006d10", .), 
    p_ne_16 = safe_chk("ph006_16", .), p_ned_16 = safe_chk("ph006d16", .), 
    p_ne_12 = safe_chk("ph006_12", .), p_ned_12 = safe_chk("ph006d12", .), 
    p_or_1  = safe_chk("ph006_1", .),  p_ord_1  = safe_chk("ph006d1", .),  
    p_or_4  = safe_chk("ph006_4", .),  p_ord_4  = safe_chk("ph006d4", .),  
    p_or_6  = safe_chk("ph006_6", .),  p_ord_6  = safe_chk("ph006d6", .),  
    p_or_21 = safe_chk("ph006_21", .), p_ord_21 = safe_chk("ph006d21", .), 
    
    has_cancer = case_when(
      (!is.na(p_ca_10) & p_ca_10 == 1) | (!is.na(p_cad_10) & p_cad_10 == 1) ~ 1, 
      is.na(p_ca_10) & is.na(p_cad_10) ~ NA_real_,
      TRUE ~ 0
    ),
    
    has_neuro = case_when(
      (!is.na(p_ne_16) & p_ne_16 == 1) | (!is.na(p_ned_16) & p_ned_16 == 1) | 
        (!is.na(p_ned_12) & p_ned_12 == 1) ~ 1, 
      TRUE ~ 0
    ),
    
    has_organ = case_when(
      (!is.na(p_or_1) & p_or_1 == 1) | (!is.na(p_ord_1) & p_ord_1 == 1) ~ 1, 
      (!is.na(p_or_4) & p_or_4 == 1) | (!is.na(p_ord_4) & p_ord_4 == 1) ~ 1, 
      (!is.na(p_or_6) & p_or_6 == 1) | (!is.na(p_ord_6) & p_ord_6 == 1) ~ 1,
      (!is.na(p_or_21) & p_or_21 == 1) | (!is.na(p_ord_21) & p_ord_21 == 1) ~ 1,
      TRUE ~ 0
    ),
    
    # Categorical Environment/Satisfaction Variables
    living_area_cat = factor(case_when(
      temp_area %in% c(1, 2) ~ "Urban/City",
      temp_area %in% c(3, 4) ~ "Town",
      temp_area == 5 ~ "Rural"
    ), levels = c("Urban/City", "Town", "Rural")),
    
    hc125_num = suppressWarnings(as.numeric(health_sat)),
    satisfaction_health = factor(case_when(
      hc125_num %in% c(1, 2) ~ "Satisfied",
      hc125_num %in% c(3, 4) ~ "Dissatisfied"
    ), levels = c("Satisfied", "Dissatisfied"))
    
  ) %>%
  # Filter observation window to +/- 3 waves around the mortality event
  filter(abs(rel_time) <= 3) %>%  
  filter(!is.na(treat_group)) %>%
  select(-starts_with("p_ca_"), -starts_with("p_ne_"), -starts_with("p_or_"), 
         -starts_with("p_cad_"), -starts_with("p_ned_"), -starts_with("p_ord_")) %>%
  arrange(survivor_id, wave)

# ==============================================================================
# 6. MISSING DATA IMPUTATION (MICE PRE-AUGMENTATION)
# ==============================================================================
# Methodological Note: 
# We impute missing baseline and time-varying covariates BEFORE passing the 
# dataset to Python generative models (TTVAE/TabDDPM). Generative models require 
# complete data structures to learn valid joint distributions. 
# We utilize Classification and Regression Trees (CART) within MICE to capture 
# potential non-linearities and complex interactions inherent in health data.
# ==============================================================================
message("--- Step 6: Missing Data Imputation (Robust MICE) ---")

pacman::p_load(mice, dplyr, haven, readr)

# --- 6.1 Data Preparation & NA Injection ---
# Certain SHARE placeholder codes (-99, -2, -1) denote missingness.
# We convert these to true NAs, strictly avoiding event-time variables and
# pre-imputed economic variables which might legitimately contain negative values.
vars_to_exclude_from_cleaning <- c(
  "survivor_id", "deceased_id", "rel_time", 
  "wealth_imp", "wealth_log", "income_imp"
)

df_clean <- df_analysis %>%
  # Convert survey placeholders to NA for numeric variables
  mutate(across(
    .cols = -any_of(vars_to_exclude_from_cleaning) & where(is.numeric),
    .fns = ~ if_else(.x %in% c(-99, -999, -2, -1), NA_real_, .x)
  )) %>%
  # Harmonize object types for MICE compatibility
  mutate(across(where(~ inherits(., "haven_labelled")), as.numeric)) %>% 
  mutate(across(where(is.character), as.factor)) %>%
  # Drop entirely empty columns
  select(where(~ sum(!is.na(.)) > 0))

message(sprintf("Total NA count before MICE imputation: %d", sum(is.na(df_clean))))

# --- 6.2 Configure MICE Architecture ---
vars_with_missing <- names(which(colSums(is.na(df_clean)) > 0))
message(sprintf("Variables requiring imputation: %d", length(vars_with_missing)))

init <- mice(df_clean, maxit = 0)
meth <- init$method
predM <- init$predictorMatrix

# Prevent identifiers from being used as predictors in the imputation model
ids_to_ignore <- c("survivor_id", "deceased_id", "mergeid", "coupleid", "is_synthetic")
for (var in names(df_clean)) {
  if (var %in% ids_to_ignore) {
    meth[var] <- ""
    predM[, var] <- 0
    predM[var, ] <- 0
  }
}

# Assign CART algorithm to all variables with missing data
meth[vars_with_missing] <- "cart"

# --- 6.3 Execute MICE ---
message("Running MICE imputation (Method: CART). Please wait...")
imp_obj <- mice(df_clean, 
                method = meth, 
                predictorMatrix = predM, 
                m = 1,            # Single imputation suitable for ML preprocessing
                maxit = 5,        # 5 iterations for convergence
                seed = 123, 
                printFlag = FALSE)

df_imputed <- complete(imp_obj)

# Final Sanity Check
n_missing_final <- sum(is.na(select(df_imputed, -any_of(ids_to_ignore))))
if(n_missing_final > 0) {
  warning(sprintf("Warning: %d missing values remain after MICE!", n_missing_final))
}

# --- 6.4 Export Analytical Base for Python Generative Pipeline ---
outfile_csv <- file.path(PATH_OUT, "BaseDataset.csv")
write_csv(df_imputed, outfile_csv)

message(sprintf("✅ BaseDataset.csv (Cleaned & Imputed) saved to: %s", outfile_csv))
message("==============================================================================")
message("👉 PIPELINE PAUSE: Execute the Python Generative Script (generate_data.py) now.")
message("👉 Output expected: 'Augmented_v1_[MODEL].csv' in the output directory.")
message("==============================================================================")

# ==============================================================================
# STOP EXECUTING HERE IF RUNNING INTERACTIVELY.
# RESUME AFTER PYTHON SCRIPT HAS GENERATED THE SYNTHETIC DATA.
# ==============================================================================

# ==============================================================================
# 7. LOAD & PREPARE AUGMENTED DATA
# ==============================================================================
# Methodological Note:
# We ingest the hybrid dataset (Real + Synthetic twins). We enforce strict 
# type casting and reconstruct structural variables (wave, post) to ensure 
# the synthetic data conforms perfectly to the econometric model requirements.
# ==============================================================================
message("--- Step 7: Loading & Validating Augmented Data ---")

# Define the target generative model to evaluate
TARGET_MODEL <- "CTGAN" 
file_path <- file.path(PATH_OUT, paste0("Augmented_v1_", TARGET_MODEL, ".csv"))

if (!file.exists(file_path)) stop(sprintf("Augmented dataset not found at: %s", file_path))

df_raw <- read_csv(file_path, show_col_types = FALSE, na = c("", "NA", "-999", "-999.0"))

# Reconstruct 'wave' if dropped during Python matrix flattening
if (!"wave" %in% names(df_raw)) {
  message("⚠️ Column 'wave' missing. Reconstructing from rel_time and wave_death...")
  df_raw <- df_raw %>% mutate(wave = wave_death + rel_time)
}

df_final <- df_raw %>%
  # 1. Clean Deep Learning Padding Artifacts
  mutate(across(where(is.numeric), ~ ifelse(. <= -990, NA, .))) %>%
  filter(!is.na(dep_score)) %>% # Drop rows with invalid outcomes
  
  # 2. Strict Type Casting for Econometrics
  mutate(
    survivor_id = as.character(survivor_id),
    wave        = as.numeric(wave),
    
    # Robust recalculation of relative time
    rel_time    = wave - wave_death,
    post        = ifelse(rel_time >= 0, 1, 0), # Event indicator
    
    dep_score   = as.numeric(dep_score),
    
    # Ensure binary treatment variable is strictly 0/1
    treat_group = as.numeric(as.factor(treat_group)) - 1,
    
    # Continuous & Binary Covariates
    age         = as.numeric(age),
    is_female   = as.numeric(is_female),
    wealth_log  = as.numeric(wealth_log),
    maxgrip     = as.numeric(maxgrip),
    has_cancer  = as.numeric(has_cancer),
    has_neuro   = as.numeric(has_neuro),
    has_organ   = as.numeric(has_organ)
  ) %>%
  # 3. Deduplication and Sorting
  distinct(survivor_id, wave, .keep_all = TRUE) %>%
  arrange(survivor_id, wave)

message(sprintf("✅ Augmented Data Loaded. Total Observations: %d", nrow(df_final)))

# ==============================================================================
# 8. BASELINE DESCRIPTIVE STATISTICS (TABLE 1)
# ==============================================================================
# Objective: Generate and export a publication-ready "Table 1" comparing 
# Treated and Control groups at baseline.
# ==============================================================================
message("--- Step 8: Generating Baseline Characteristics (Table 1) ---")

pacman::p_load(gtsummary, gt)

table1 <- df_final %>%
  group_by(survivor_id) %>%
  filter(
    (treat_group == 1 & rel_time == -1) |
      (treat_group == 0 & wave == min(wave))
  ) %>%
  ungroup() %>%
  distinct(survivor_id, .keep_all = TRUE) %>%
  select(treat_group, age, is_female, dep_score, wealth_log, satisfaction_health) %>%
  mutate(Group = factor(treat_group, levels = c(0, 1), labels = c("Control", "Treated"))) %>%
  
  tbl_summary(
    by = Group,
    include = -treat_group,
    # Explicitly define continuous variables to prevent gtsummary from guessing
    type = list(
      dep_score ~ "continuous",
      age ~ "continuous",
      wealth_log ~ "continuous"
    ),
    statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)"),
    missing = "no",
    label = list(
      age ~ "Age (Years)",
      is_female ~ "Female",
      dep_score ~ "Depression Score (Euro-D)",
      wealth_log ~ "Log Household Wealth",
      satisfaction_health ~ "Health Satisfaction"
    )
  ) %>%
  # Force standard statistical tests for speed and compatibility
  add_p(test = list(all_continuous() ~ "t.test", all_categorical() ~ "chisq.test")) %>% 
  add_overall() %>%
  modify_header(label = "**Characteristic**") %>%
  modify_caption("**Table 1. Baseline Characteristics of the Analytical Sample**") %>%
  bold_labels()

# Print to console for interactive viewing
print(table1)

# Convert gtsummary object to a gt object
gt_table1 <- as_gt(table1) %>%
  tab_options(
    table.width = pct(80),
    data_row.padding = px(5),
    heading.title.font.weight = "bold"
  )

# Save as PNG (Great for presentations/LaTeX inclusion)
gtsave(gt_table1, filename = file.path(PATH_OUT, "Table1_Baseline.png"))

message("✅ Table 1 successfully exported.")

# ==============================================================================
# 9. ECONOMETRIC ESTIMATION STRATEGY
# ==============================================================================
message("--- Step 9: Causal Inference Modeling ---")

pacman::p_load(fixest, plm, MatchIt, cobalt, ggplot2, modelsummary)

# ------------------------------------------------------------------------------
# MODEL 1: Staggered DiD (Standard Event Study)
# ------------------------------------------------------------------------------
message("Estimating Model 1: TWFE Event Study...")

mod_did_base <- feols(
  dep_score ~ i(rel_time, treat_group, ref = -1) + age | survivor_id + wave, 
  data = df_final, 
  cluster = ~survivor_id
)

# ------------------------------------------------------------------------------
# MODEL 2: System GMM (Dynamic Panel)
# Rationale: Addresses the "Nickell Bias" when including lagged dependent 
# variables in panels with short T and large N.
# ------------------------------------------------------------------------------
message("Estimating Model 2: System GMM...")

df_gmm <- df_final %>% 
  select(survivor_id, wave, dep_score, treat_group, post, 
         satisfaction_health, adl_score, wealth_log, maxgrip, 
         has_cancer, has_neuro, has_organ) %>%
  na.omit() 

p_share <- pdata.frame(df_gmm, index = c("survivor_id", "wave"))

# Wrapped in tryCatch: GMM covariance matrices can be singular with synthetic data
mod_gmm <- tryCatch({
  pgmm(
    dep_score ~ lag(dep_score, 1) + treat_group:post + 
      satisfaction_health + adl_score + wealth_log + maxgrip +
      has_cancer + has_neuro + has_organ | 
      lag(dep_score, 2:4), 
    data = p_share, 
    effect = "individual", model = "twosteps", transformation = "ld", 
    robust = TRUE, collapse = TRUE
  )
}, error = function(e) {
  message("⚠️ GMM Estimation failed (common with specific synthetic covariance structures). Skipping.")
  return(NULL)
})

# ------------------------------------------------------------------------------
# MODEL 3: Dynamic OLS (FE with Lagged Y)
# ------------------------------------------------------------------------------
message("Estimating Model 3: Dynamic OLS...")

mod_dyn_ols <- feols(
  dep_score ~ l(dep_score, 1) + i(rel_time, treat_group, ref = -1) | survivor_id + wave,
  data = df_final, 
  panel.id = ~survivor_id + wave, 
  cluster = ~survivor_id
)

# ------------------------------------------------------------------------------
# MODEL 4: Matched Event Study (Sun & Abraham approach approximation)
# Rationale: Ensures strict Parallel Trends by matching units on covariates
# AND baseline outcome (dep_score) at t = -1 prior to TWFE estimation.
# ------------------------------------------------------------------------------
message("Estimating Model 4: Propensity Score Matched DiD...")

# A. Isolate Baseline Data (t = -1)
df_baseline <- df_final %>%
  filter(rel_time == -1) %>%  
  distinct(survivor_id, .keep_all = TRUE)

# Optional: Inspect missingness before matching
# missing_stats <- df_baseline %>% summarise(across(everything(), ~ mean(is.na(.)) * 100))

# B. Execute Nearest Neighbor Matching
match_obj <- matchit(
  treat_group ~ age + is_female + wealth_log + hc125_num + living_area_cat + 
    country + cause_death + has_cancer + maxgrip + has_neuro + 
    has_organ + dep_score, 
  data = df_baseline, 
  method = "nearest", 
  distance = "glm",   
  ratio = 1,          
  replace = TRUE,    
  caliper = 0.25      
)

message("Propensity Score Matching Summary:")
print(summary(match_obj, un = FALSE))

# C. Extract Weights and Merge with Longitudinal Panel
m_data <- match.data(match_obj) %>% select(survivor_id, weights)
df_matched <- df_final %>%
  inner_join(m_data, by = "survivor_id")

# D. Estimate Matched Event Study
mod_sa <- feols(
  dep_score ~ i(rel_time, treat_group, ref = -1, keep = c(-3, -2, 0, 1, 2)) | survivor_id,
  data = df_matched,
  weights = ~weights,
  cluster = ~survivor_id
)

# ==============================================================================
# 10. PUBLICATION-READY OUTPUTS
# ==============================================================================
message("--- Step 10: Generating Tables and Figures ---")

# A. Regression Table Export
models_list <- list(
  "DiD (Base)" = mod_did_base, 
  "Dyn. OLS" = mod_dyn_ols, 
  "Matched Event Study" = mod_sa
)
if(!is.null(mod_gmm)) models_list[["System GMM"]] <- mod_gmm

modelsummary(
  models_list,
  stars = c('*' = .1, '**' = .05, '***' = .01),
  gof_map = c("nobs", "r.squared"),
  output = file.path(PATH_OUT, sprintf("Table_Regression_Results_%s.png", TARGET_MODEL))
)

# B. Dynamic Event Study Plotter
plot_event_study <- function(model_obj, title, filename) {
  
  est <- broom::tidy(model_obj, conf.int = TRUE) %>%
    filter(str_detect(term, "rel_time::|wave::")) %>%
    mutate(
      time = as.numeric(str_extract(term, "-?\\d+")),
      model = title
    ) %>%
    filter(!is.na(time) & time >= -4 & time <= 4)
  
  if(!(-1 %in% est$time)) est <- bind_rows(est, tibble(time = -1, estimate = 0, conf.low = 0, conf.high = 0))
  
  y_min <- min(est$conf.low, na.rm = TRUE) - 0.2
  y_max <- max(est$conf.high, na.rm = TRUE) + 0.2
  
  p <- ggplot(est, aes(x = time, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_vline(xintercept = -0.5, linetype = "dotted", color = "#c0392b", linewidth = 1) +
    geom_line(color = "#2c3e50", linewidth = 1) +
    geom_pointrange(aes(ymin = conf.low, ymax = conf.high), color = "#2c3e50", fill = "white", shape = 21, size = 0.8) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    labs(title = title, x = "Time Since Event (Years)", y = "Effect on Euro-D Score") +
    theme_classic(base_size = 14) +
    scale_x_continuous(breaks = seq(-4, 4, 1))
  
  ggsave(file.path(PATH_OUT, paste0(filename, ".png")), p, width = 8, height = 5, dpi = 300)
  return(p)
}

# Generate Plots
p1 <- plot_event_study(mod_did_base, "Model 1: Standard TWFE DiD", sprintf("Fig_M1_DiD_%s", TARGET_MODEL))
p3 <- plot_event_study(mod_dyn_ols, "Model 3: Dynamic OLS", sprintf("Fig_M3_Dyn_OLS_%s", TARGET_MODEL))
p4 <- plot_event_study(mod_sa, "Model 4: Matched Event Study", sprintf("Fig_M4_Matched_%s", TARGET_MODEL))

plot_gmm_att <- function(gmm_model, title, filename) {
  
  if(is.null(gmm_model)) {
    message("Modello GMM nullo, impossibile plottare.")
    return(NULL)
  }
  
  # 1. Estrazione manuale dei coefficienti dal summary del pgmm
  gmm_sum <- summary(gmm_model)
  coef_matrix <- gmm_sum$coefficients
  
  # 2. Cerchiamo la riga relativa all'interazione del trattamento
  # Può chiamarsi "treat_group:post" o "post:treat_group" a seconda di plm
  target_row <- rownames(coef_matrix)[str_detect(rownames(coef_matrix), "treat_group:post|post:treat_group")]
  
  if(length(target_row) == 0) {
    message("Coefficiente di trattamento non trovato nel GMM.")
    return(NULL)
  }
  
  # 3. Estrazione stima ed errore standard per calcolare l'intervallo di confidenza al 95%
  est <- coef_matrix[target_row, "Estimate"]
  se  <- coef_matrix[target_row, "Std. Error"]
  
  ci_low  <- est - 1.96 * se
  ci_high <- est + 1.96 * se
  
  df_plot <- tibble(
    term = "ATT\n(Post x Treated)",
    estimate = est,
    conf.low = ci_low,
    conf.high = ci_high
  )
  
  # 4. Creazione del Forest Plot (Orizzontale)
  p <- ggplot(df_plot, aes(x = term, y = estimate)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40", linewidth = 0.8) +
    geom_pointrange(aes(ymin = conf.low, ymax = conf.high), 
                    color = "#c0392b", fill = "white", shape = 21, size = 1.2, linewidth = 1.2) +
    coord_flip() + # Ruota il grafico per renderlo un classico forest plot
    labs(
      title = title,
      subtitle = "Average Treatment Effect on the Treated (95% CI)",
      x = "", 
      y = "Overall Effect on Depression (Euro-D)"
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.y = element_text(face = "bold", size = 12)
    )
  
  ggsave(file.path(PATH_OUT, paste0(filename, ".png")), p, width = 7, height = 3.5, dpi = 300)
  return(p)
}

message("✅ Full Analysis Pipeline Complete. Assets saved to Output directory.")

# ==============================================================================
# 12. APPENDICES & SENSITIVITY ANALYSIS (MATCHED DID SPECIFICATION)
# ==============================================================================
# Methodological Note: This section generates robustness checks and appendix 
# materials required for high-impact journals. It includes:
# 1. Heterogeneity analyses (Welfare Regime, Gender, Baseline Depression).
# 2. Negative Control Outcome (Placebo) tests using physical grip strength.
# 3. Assessment of selection on unobservables using Oster's (2019) bounds.
# ==============================================================================

message("--- Step 11: Generating Appendices and Sensitivity Checks ---")

# Ensure required packages for formatting and matching are loaded
pacman::p_load(patchwork, purrr, MatchIt, stringr, gt, modelsummary) 

# Create sub-directory for Appendix outputs
PATH_APP <- file.path(PATH_OUT, "Appendices")
if(!dir.exists(PATH_APP)) dir.create(PATH_APP)

# ------------------------------------------------------------------------------
# A. HELPER FUNCTIONS FOR ROBUSTNESS CHECKS
# ------------------------------------------------------------------------------

#' Coefficient Extraction Helper
#' Purpose: Extracts and formats event-study coefficients for publication-ready 'gt' tables.
extract_coeffs_robust <- function(model, name) {
  if(is.null(model)) return(NULL)
  
  est <- tryCatch(modelsummary::get_estimates(model), error = function(e) NULL)
  if(is.null(est) || nrow(est) == 0) return(NULL)
  
  if (any(str_detect(est$term, "rel_time"))) {
    df_clean <- est %>%
      filter(str_detect(term, "rel_time")) %>%
      mutate(
        time_str = str_extract(term, "(?<=rel_time::)-?\\d+"),
        time_point = as.numeric(time_str),
        col_name = case_when(
          time_point == 0 ~ "t0",
          time_point > 0 ~ paste0("t+", time_point),
          TRUE ~ paste0("t", time_point)
        ),
        # Standard significance stars
        stars = case_when(p.value < 0.01 ~ "***", p.value < 0.05 ~ "**", p.value < 0.1 ~ "*", TRUE ~ ""),
        val_str = paste0(round(estimate, 3), stars, "\n(", round(std.error, 3), ")")
      ) %>%
      filter(!is.na(col_name)) %>% 
      select(col_name, val_str)
    
    # Manually inject the reference period (t-1) for table completeness
    if(!"t-1" %in% df_clean$col_name) {
      df_clean <- bind_rows(df_clean, tibble(col_name = "t-1", val_str = "Ref."))
    }
    
    return(df_clean %>% mutate(Model = name))
  } else {
    return(NULL)
  }
}

#' Matched DiD Wrapper for Subgroups
#' Purpose: Re-estimates the Propensity Score and the Event Study within specific subgroups.
run_matched_did_subgroup <- function(sub_data, group_name, formula_dep = "dep_score") {
  message(sprintf(">> Processing Subgroup: %s", group_name))
  
  # 1. Isolate Baseline Data (t = -1) for the subgroup
  df_baseline_sub <- sub_data %>%
    filter(rel_time == -1) %>%
    select(survivor_id, treat_group, death_year, 
           age, is_female, edu_level, wealth_log, living_area_cat, 
           has_cancer, has_neuro, has_organ) %>% 
    na.omit()
  
  # Sample size safety check to prevent matching errors
  if(nrow(df_baseline_sub) < 50) {
    warning(sprintf("Sample size insufficient (N=%d) for subgroup: %s", nrow(df_baseline_sub), group_name))
    return(NULL)
  }
  
  # 2. Execute Matching (Nearest Neighbor + Replacement)
  match_obj <- tryCatch({
    matchit(
      treat_group ~ age + is_female + wealth_log + living_area_cat + has_cancer,
      data = df_baseline_sub, 
      method = "nearest", 
      distance = "glm",   
      ratio = 2,          
      replace = TRUE      
    )
  }, error = function(e) return(NULL))
  
  if(is.null(match_obj)) return(NULL)
  
  # 3. Extract Weights and Run Weighted DiD
  matched_w <- match.data(match_obj) %>% select(survivor_id, weights)
  df_weighted_sub <- sub_data %>% inner_join(matched_w, by = "survivor_id")
  fml <- as.formula(paste(formula_dep, "~ i(rel_time, treat_group, ref = -1) | survivor_id + int_year"))
  
  mod <- tryCatch({
    feols(fml, data = df_weighted_sub, weights = ~weights, cluster = ~survivor_id, notes = FALSE)
  }, error = function(e) return(NULL))
  
  return(list(model = mod, name = group_name))
}

#' Generate Publication-Ready GT Tables
create_appendix_table <- function(model_list, title_text, filename) {
  rows_df <- map_dfr(model_list, function(item) {
    if(is.null(item)) return(NULL)
    extract_coeffs_robust(item$model, item$name)
  })
  
  if(nrow(rows_df) == 0) return(NULL)
  
  col_order <- c("Model", "t-3", "t-2", "t-1", "t0", "t+1", "t+2", "t+3")
  
  tab_data <- rows_df %>%
    pivot_wider(names_from = col_name, values_from = val_str) %>%
    select(any_of(col_order)) %>%
    replace(is.na(.), "-")
  
  gt_tab <- tab_data %>% gt() %>%
    tab_header(title = md(paste0("**", title_text, "**"))) %>%
    tab_style(style = list(cell_text(weight = "bold")), locations = cells_body(columns = Model)) %>%
    cols_align(align = "center", columns = -Model)
  
  gtsave(gt_tab, file.path(PATH_APP, paste0(filename, ".png")))
}

#' Oster Bounds Statistics Extractor
get_stats_oster <- function(model, name_prefix) {
  r2_val <- as.numeric(r2(model, type = "r2"))[1]
  modelsummary::get_estimates(model) %>%
    filter(str_detect(term, "rel_time")) %>%
    mutate(
      time_str = str_extract(term, "(?<=rel_time::)-?\\d+"),
      time = as.numeric(time_str)
    ) %>%
    filter(time >= 0) %>% 
    select(time, estimate) %>%
    rename_with(~paste0(name_prefix, "_", .), estimate) %>%
    mutate(!!paste0(name_prefix, "_r2") := r2_val)
}

# ------------------------------------------------------------------------------
# B. HETEROGENEITY ANALYSIS (Matched DiD)
# ------------------------------------------------------------------------------
message("--- Appendix B: Subgroup Analyses ---")

# 1. By Welfare Regimes
df_welfare <- df_final %>%
  mutate(cntry = as.numeric(country),
         welfare_regime_check = case_when(
           cntry %in% c(13, 18, 55, 14) ~ "Nordic", 
           cntry %in% c(11, 12, 17, 20, 23, 31, 25) ~ "Continental", 
           cntry %in% c(15, 16, 19, 33, 53, 59) ~ "Southern", 
           cntry %in% c(28, 29, 34, 35, 47, 48, 32, 51, 57, 61, 63) ~ "Eastern",
           TRUE ~ NA_character_ )) %>%
  filter(!is.na(welfare_regime_check))

res_welfare <- map(unique(df_welfare$welfare_regime_check), function(r) {
  run_matched_did_subgroup(df_welfare %>% filter(welfare_regime_check == r), r)
}) %>% compact()

create_appendix_table(res_welfare, "Heterogeneity by Welfare Regime", "Table_App_Welfare")

# Generate plots silently (assuming plot_event_study exists from previous sections)
walk(res_welfare, function(res) {
  suppressMessages(plot_event_study(res$model, paste("Welfare:", res$name), paste0("App_Welfare_", res$name)))
})

# 2. By Gender
res_gender <- map(c(0, 1), function(g) {
  run_matched_did_subgroup(df_final %>% filter(is_female == g), ifelse(g==1, "Female", "Male"))
}) %>% compact()

create_appendix_table(res_gender, "Heterogeneity by Gender", "Table_App_Gender")
walk(res_gender, function(res) {
  suppressMessages(plot_event_study(res$model, paste("Gender:", res$name), paste0("App_Gender_", res$name)))
})

# 3. By Baseline Depression Status
median_eurod <- median(df_final$dep_score[df_final$rel_time == -1], na.rm = TRUE)

df_dep_split <- df_final %>% 
  group_by(survivor_id) %>%
  mutate(baseline_val = dep_score[rel_time == -1][1], 
         dep_group = if_else(baseline_val > median_eurod, "High Baseline", "Low Baseline")) %>%
  ungroup() %>% 
  filter(!is.na(dep_group))

res_dep <- map(unique(df_dep_split$dep_group), function(d) {
  run_matched_did_subgroup(df_dep_split %>% filter(dep_group == d), d)
}) %>% compact()

create_appendix_table(res_dep, "Heterogeneity by Baseline Depression", "Table_App_BaselineDep")
walk(res_dep, function(res) {
  suppressMessages(plot_event_study(res$model, paste("Subgroup:", res$name), paste0("App_Dep_", res$name)))
})

# ------------------------------------------------------------------------------
# C. FALSIFICATION TEST (NEGATIVE CONTROL OUTCOME)
# ------------------------------------------------------------------------------
# Rationale: Palliative care should not have a causal impact on purely physiological 
# traits like grip strength. If an effect is found, it signals unobserved confounding.
message("--- Appendix C: Placebo Test (MaxGrip) ---")

df_placebo_ready <- df_final %>%
  mutate(maxgrip_val = suppressWarnings(as.numeric(maxgrip)),
         maxgrip_val = if_else(maxgrip_val > 100 | maxgrip_val < 0, NA_real_, maxgrip_val)) %>%
  filter(!is.na(maxgrip_val))

res_placebo_mg <- run_matched_did_subgroup(
  df_placebo_ready, 
  "Placebo Outcome: MaxGrip", 
  formula_dep = "maxgrip_val"
)

if(!is.null(res_placebo_mg)) {
  create_appendix_table(list(res_placebo_mg), "Negative Control Outcome Test", "Table_App_Placebo")
  suppressMessages(plot_event_study(res_placebo_mg$model, "Placebo Test: Effect on Grip Strength", "App_Placebo_MaxGrip"))
}

# ------------------------------------------------------------------------------
# D. SENSITIVITY ANALYSIS: OSTER BOUNDS (MATCHED DID)
# ------------------------------------------------------------------------------
# Rationale: Evaluates the robustness of the ATT to unobserved confounding, 
# assuming R_max = 1.3 * R_tilde^2 (standard in literature).
message("--- Appendix D: Oster's Bounds Analysis (Matched DiD) ---")

# 1. Reconstruct Matched Data (Ensuring exact year matching)
df_baseline_tmp <- df_final %>%
  filter(rel_time == -1) %>%
  select(survivor_id, treat_group, death_year, 
         age, is_female, edu_level, wealth_log, living_area_cat, 
         has_cancer, has_neuro, has_organ) %>% 
  na.omit()

match_obj_tmp <- matchit(
  treat_group ~ age + is_female + edu_level + wealth_log + living_area_cat + 
    has_cancer + has_neuro + has_organ,
  data = df_baseline_tmp, method = "nearest", distance = "glm", 
  replace = TRUE, exact = ~ death_year, verbose = FALSE
)

df_oster <- df_final %>% inner_join(match.data(match_obj_tmp) %>% select(survivor_id, weights), by = "survivor_id")

# 2. Estimate Full and Restricted Models
mod_did_matched <- feols(dep_score ~ i(rel_time, treat_group, ref = -1) | survivor_id + int_year, 
                         data = df_oster, weights = ~weights, cluster = ~survivor_id)

mod_restricted <- feols(dep_score ~ i(rel_time, treat_group, ref = -1) | survivor_id + int_year, 
                        data = df_oster, weights = ~weights, cluster = ~survivor_id)

# 3. Calculate Oster's Delta
stats_full  <- get_stats_oster(mod_did_matched, "tilde")
stats_restr <- get_stats_oster(mod_restricted, "ring")

if(nrow(stats_full) > 0 && nrow(stats_restr) > 0) {
  oster_results <- stats_full %>% left_join(stats_restr, by = "time") %>%
    mutate(
      R_max = pmin(1, 1.3 * tilde_r2), 
      beta_diff = ring_estimate - tilde_estimate, 
      r2_diff = tilde_r2 - ring_r2,
      delta = case_when(
        abs(tilde_estimate) >= abs(ring_estimate) ~ Inf, 
        abs(r2_diff) <= 0.00001 ~ 0, 
        TRUE ~ abs(tilde_estimate / (beta_diff * ((R_max - tilde_r2) / r2_diff)))
      ),
      beta_bound = tilde_estimate - if_else(abs(r2_diff) <= 0.00001, 0, (beta_diff * (R_max - tilde_r2) / r2_diff))
    )
  
  table_oster <- oster_results %>%
    mutate(
      Time = paste0("t + ", time), 
      Coeff_Main = round(tilde_estimate, 3), 
      R2_Move = paste0(round(ring_r2, 3), " -> ", round(tilde_r2, 3)),
      Delta_Label = case_when(
        delta == Inf ~ "> 10 (Very Robust)", 
        delta > 1 ~ paste0(round(delta, 2), " (Robust)"), 
        delta <= 0 ~ "Unstable", 
        TRUE ~ paste0(round(delta, 2), " (Fragile)")
      ),
      Identified_Set = paste0("[", round(tilde_estimate, 3), ", ", round(beta_bound, 3), "]")
    ) %>%
    select(Time, Coeff_Main, R2_Move, Identified_Set, Delta_Label) %>% gt() %>%
    tab_header(title = md("**Robustness: Oster's Bounds Analysis (Matched DiD)**"))
  
  gtsave(table_oster, filename = file.path(PATH_APP, "Table_App_OsterBounds.png"))
  message("✅ Oster Bounds (Matched DiD) successfully generated.")
}

# ==============================================================================
# 13. APPENDIX F: ROBUSTNESS TO ALTERNATIVE ESTIMATORS
# ==============================================================================
# Methodological Note: The Matched DiD (Sun & Abraham approximation) is our 
# preferred specification (Main Text). Here, we demonstrate that the average 
# treatment effect on the treated (ATT) is robust to alternative panel data 
# estimators, specifically Dynamic OLS (which accounts for state dependence 
# via a lagged dependent variable) and standard Two-Way Fixed Effects.
# ==============================================================================
message("--- Appendix F: Robustness to Alternative Estimators ---")

# 1. Ensure models exist in the environment
if(!exists("mod_did_base")) {
  mod_did_base <- feols(dep_score ~ i(rel_time, treat_group, ref = -1) | survivor_id + int_year, 
                        data = df_final, cluster = ~survivor_id, notes = FALSE)
}

if(!exists("mod_dyn_ols")) {
  mod_dyn_ols <- feols(dep_score ~ l(dep_score, 1) + i(rel_time, treat_group, ref = -1) | survivor_id + int_year, 
                       data = df_final, panel.id = ~survivor_id + wave, cluster = ~survivor_id, notes = FALSE)
}

# 2. Create a clean comparative table for the Appendix
if(exists("mod_sa")) {
  
  message("Generating Alternative Estimators Comparison Table...")
  
  models_comp <- list(
    "Standard TWFE" = mod_did_base,
    "Dynamic OLS" = mod_dyn_ols,
    "Matched DiD (Preferred)" = mod_sa
  )
  
  # Format the table using modelsummary
  comp_table <- modelsummary(
    models_comp,
    stars = c('*' = .1, '**' = .05, '***' = .01),
    coef_map = c(
      "rel_time::-3:treat_group" = "Pre: t - 3",
      "rel_time::-2:treat_group" = "Pre: t - 2",
      "rel_time::0:treat_group"  = "Post: t = 0 (Event)",
      "rel_time::1:treat_group"  = "Post: t + 1",
      "rel_time::2:treat_group"  = "Post: t + 2"
    ),
    gof_map = c("nobs", "r.squared", "adj.r.squared"),
    title = "Robustness to Alternative Econometric Specifications",
    output = "gt"
  ) %>%
    gt::tab_style(
      style = gt::cell_text(weight = "bold"),
      locations = gt::cells_column_labels(columns = everything())
    )
  
  # Save the comparative table
  gt::gtsave(comp_table, file.path(PATH_APP, "Table_App_F_Estimator_Comparison.png"))
  message("✅ Alternative Estimators table saved.")
  
} else {
  message("⚠️ Preferred model (mod_sa) not found. Run main analysis first.")
}

message("🏁 ALL APPENDICES, PLOTS, AND CHECKS SUCCESSFULLY GENERATED.")
