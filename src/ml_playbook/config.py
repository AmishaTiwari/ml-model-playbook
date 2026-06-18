# config.py

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = BASE_DIR / "data"

# Model paths
MODEL_DIR = BASE_DIR / "models"

# Global configuration
RANDOM_STATE = 42