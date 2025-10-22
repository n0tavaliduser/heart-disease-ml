"""
Model training module for the heart disease prediction project.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .utils import MODEL_PARAMS, RANDOM_STATE

def create_models():
    """
    Create dictionary of models with their configurations.
    
    Returns:
        dict: Dictionary of initialized models
    """
    models = {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE,
            **MODEL_PARAMS["logistic_regression"]
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            **MODEL_PARAMS["random_forest"]
        ),
        "svm": SVC(
            random_state=RANDOM_STATE,
            probability=True,
            **MODEL_PARAMS["svm"]
        ),
        "xgboost": XGBClassifier(
            random_state=RANDOM_STATE,
            **MODEL_PARAMS["xgboost"]
        )
    }
    return models

def train_model(model, X_train, y_train):
    """
    Train a single model.
    
    Args:
        model: The machine learning model to train
        X_train: Training features
        y_train: Training labels
        
    Returns:
        object: Trained model
    """
    model.fit(X_train, y_train)
    return model

def train_all_models(models, X_train, y_train):
    """
    Train all models in the dictionary.
    
    Args:
        models (dict): Dictionary of model names and their instances
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_models[name] = train_model(model, X_train, y_train)
    return trained_models
