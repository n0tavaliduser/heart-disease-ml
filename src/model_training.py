"""
Model training module for the heart disease prediction project.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from .neural_network import NeuralNetworkWrapper
from .utils import MODEL_PARAMS, RANDOM_STATE

def create_models():
    """
    Create dictionary of models with their configurations.
    
    Returns:
        dict: Dictionary of initialized models
    """
    # Base models
    base_models = {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            **MODEL_PARAMS["random_forest"]
        ),
        "svm": SVC(
            random_state=RANDOM_STATE,
            probability=True,
            kernel='rbf',
            C=1.0
        ),
        "xgboost": XGBClassifier(
            random_state=RANDOM_STATE,
            **MODEL_PARAMS["xgboost"]
        ),
        "neural_network": NeuralNetworkWrapper()
    }
    
    # Create stacking ensemble
    estimators = [
        ('rf', base_models['random_forest']),
        ('xgb', base_models['xgboost']),
        ('nn', base_models['neural_network'])
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            random_seed=RANDOM_STATE,
            verbose=0
        ),
        n_jobs=-1
    )
    
    # Create voting ensemble
    voting = VotingClassifier(
        estimators=[
            ('rf', base_models['random_forest']),
            ('xgb', base_models['xgboost']),
            ('nn', base_models['neural_network'])
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Add ensemble models
    base_models["stacking"] = stacking
    base_models["voting"] = voting
    
    return base_models

def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Train a single model.
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
    
    Returns:
        Trained model
    """
    if isinstance(model, NeuralNetworkWrapper) and X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        model.fit(X_train, y_train, validation_data=validation_data)
    else:
        model.fit(X_train, y_train)
    return model

def train_all_models(models, X_train, y_train, X_val=None, y_val=None):
    """
    Train all models in the dictionary.
    
    Args:
        models (dict): Dictionary of model instances
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
    
    Returns:
        dict: Dictionary of trained models
    """
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_models[name] = train_model(model, X_train, y_train, X_val, y_val)
    return trained_models
