"""
LSTM Deep Learning Model
Implements LSTM for Olympic medal prediction
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class LSTMConfig:
    """LSTM model configuration"""
    sequence_length: int = 3
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    dense_units: int = 32
    epochs: int = 100
    batch_size: int = 32
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64]


class LSTMModel:
    """
    LSTM model for time series forecasting.
    
    Architecture:
    - LSTM layers (128 -> 64 units)
    - Dropout layers (0.2)
    - Dense output layer
    """
    
    def __init__(self, config: LSTMConfig = None):
        """Initialize LSTM model"""
        self.config = config or LSTMConfig()
        self.model = None
        self.is_fitted = False
        self.history = None

    @staticmethod
    def create_sequences(data: np.ndarray, 
                         seq_length: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            data: 1D array of values
            seq_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            self.config.lstm_units[0],
            return_sequences=len(self.config.lstm_units) > 1,
            input_shape=input_shape
        ))
        self.model.add(Dropout(self.config.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.config.lstm_units[1:]):
            return_seq = i < len(self.config.lstm_units) - 2
            self.model.add(LSTM(units, return_sequences=return_seq))
            self.model.add(Dropout(self.config.dropout_rate))
        
        # Dense layers
        self.model.add(Dense(self.config.dense_units, activation='relu'))
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.2) -> dict:
        """
        Train LSTM model.
        
        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Target values
            validation_split: Fraction for validation
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        # Reshape if needed
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model if not built
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=self.config.epochs // 10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=self.config.epochs // 20)
        ]
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return self.model.predict(X, verbose=0).flatten()
