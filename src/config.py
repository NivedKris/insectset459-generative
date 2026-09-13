"""
Global configuration constants and paths for IEEE SPL Reproducibility Kit.
Matches exact specifications from:
'When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale'
"""
import os

# Dataset & Taxonomy constants
NUM_CLASSES = 459
NUM_GENERA = 196
NULL_SPECIES_ID = 459
NULL_GENUS_ID = 196

# Audio processing constants (48 kHz wideband bioacoustics)
SAMPLE_RATE = 48000
WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5
N_FFT = 4096
HOP_LENGTH = 1024
N_MELS = 128
F_MIN = 400.0
F_MAX = 22000.0
EXPECTED_FRAMES = 235  # ceil(5.0 * 48000 / 1024) = 235

# Quartile Tier boundaries by training chunk count
# Tier 1 (Head): >600 chunks (114 species)
# Tier 2 (Mid):  251-600 chunks (115 species)
# Tier 3 (Few):  119-250 chunks (115 species)
# Tier 4 (Tail): <=118 chunks (115 species)
TIER_BOUNDS = {
    "tier_1": (600, float("inf")),
    "tier_2": (250, 600),
    "tier_3": (118, 250),
    "tier_4": (0, 118),
}

# TOST Equivalence Margin (from pre-registered protocol)
TOST_MARGIN = 0.020
ALPHA = 0.05

# Paths (relative to kit root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Pre-trained checkpoint file paths (GitHub-safe, <100MB)
ORACLE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "oracle", "best_oracle.pt")
OT_CFM_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "ot_cfm", "best_ot_cfm.pt")
DDPM_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "ddpm", "best_ddpm.pt")
CVAE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "cvae", "best_cvae.pt")
ACGAN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "acgan", "best_acgan.pt")

# Phase 3 chunks dir
PHASE3_CHUNKS_DIR = os.path.join(CHECKPOINTS_DIR, "phase3_chunks")

# Zenodo download constants
ZENODO_RECORD_ID = "14056458"
ZENODO_CSV_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/InsectSet459_Train_Val_Annotation.csv?download=1"
ZENODO_TRAIN_ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/Train.zip?download=1"
