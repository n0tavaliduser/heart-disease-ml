"""
Data preprocessing module for the heart disease prediction project.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .utils import COLUMN_NAMES, DATASET_PATH, RANDOM_STATE, TEST_SIZE

def load_data():
    """
    Load the Cleveland heart disease dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset with proper column names
    """
    df = pd.read_csv(DATASET_PATH, names=COLUMN_NAMES, na_values="?")
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, scaling features,
    and splitting into train/test sets.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Handle missing values
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric)
    df = df.fillna(df.median())
    
    # Convert target to binary (0: no disease, 1: disease)
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model if available.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        pd.Series: Feature importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        importance = pd.Series(
            np.abs(model.coef_[0]),
            index=feature_names
        ).sort_values(ascending=False)
    else:
        importance = None
    
    return importance
