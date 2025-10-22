"""
Utility functions and configurations for the heart disease prediction project.
"""
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "results"

# Dataset configuration
DATASET_FILENAME = "processed.cleveland.data"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Column names for the Cleveland dataset
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Model parameters dictionary
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf"
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1
    }
}

def setup_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "plot").mkdir(exist_ok=True)

def save_results(metrics, filename="metrics_report.json"):
    """Save evaluation metrics to a JSON file."""
    import json
    results_file = RESULTS_DIR / filename
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
