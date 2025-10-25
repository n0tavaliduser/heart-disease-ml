"""
Neural network model implementation for heart disease prediction.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def create_neural_network():
    """
    Create a Multilayer Perceptron model optimized for small datasets.
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(13,)),  # 13 features
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0008)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks():
    """
    Get training callbacks for learning rate reduction and early stopping.
    
    Returns:
        List of Keras callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]

class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for neural network to make it compatible with scikit-learn API.
    Inherits from BaseEstimator and ClassifierMixin to properly identify as a classifier.
    """
    def __init__(self, batch_size=16, epochs=100):
        self.model = None
        self.callbacks = get_callbacks()
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes_ = np.array([0, 1])  # Binary classification
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
                
        Returns:
            dict: Parameter names mapped to their values
        """
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Args:
            **parameters: Estimator parameters
            
        Returns:
            self: Estimator instance
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y, validation_data=None):
        """
        Train the neural network model.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Tuple of (X_val, y_val) for validation
            
        Returns:
            self: Returns the instance itself.
        """
        # Store unique classes
        self.classes_ = np.unique(y)
        
        self.model = create_neural_network()
        
        # Convert data to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (0 or 1)
        """
        X = np.array(X)
        return (self.model.predict(X) > 0.5).astype(int).ravel()
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Probability predictions for each class
        """
        X = np.array(X)
        probs = self.model.predict(X).ravel()
        return np.column_stack([1 - probs, probs])  # Return probabilities for both classes