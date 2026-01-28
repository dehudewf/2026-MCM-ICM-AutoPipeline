"""
Model Persistence Module
Handles saving and loading of trained models
"""
import os
import pickle
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelMetadata:
    """Metadata for saved model"""
    model_name: str
    model_type: str
    version: str
    created_at: str
    random_seed: int
    features: list
    metrics: Dict[str, float]


class ModelPersistence:
    """
    Handles model serialization and deserialization.
    
    Supports:
    - pickle for sklearn models
    - h5 for Keras models
    - Version tracking
    - Reproducibility with random seeds
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            models_dir = os.path.join(base_dir, 'models')
        
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.version_history = {}
    
    def get_format_for_model(self, model: Any) -> str:
        """
        Determine appropriate format for model type.
        
        Returns:
            'pickle' for sklearn models, 'h5' for Keras models
        """
        model_type = type(model).__name__
        
        # Check for Keras models
        keras_types = ['Sequential', 'Model', 'Functional']
        if model_type in keras_types or 'keras' in str(type(model).__module__):
            return 'h5'
        
        # Default to pickle for sklearn and other models
        return 'pickle'
    
    def save_model(self, model: Any, name: str,
                   metadata: ModelMetadata = None,
                   format: str = None) -> str:
        """
        Save model to file.
        
        Args:
            model: Trained model object
            name: Model name for filename
            metadata: Optional metadata
            format: 'pickle' or 'h5' (auto-detected if None)
        """
        if format is None:
            format = self.get_format_for_model(model)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'h5':
            filename = f"{name}_{timestamp}.h5"
            filepath = os.path.join(self.models_dir, filename)
            try:
                model.save(filepath)
            except AttributeError:
                # Fallback to pickle if h5 save fails
                format = 'pickle'
                filename = f"{name}_{timestamp}.pkl"
                filepath = os.path.join(self.models_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
        else:
            filename = f"{name}_{timestamp}.pkl"
            filepath = os.path.join(self.models_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            meta_path = filepath.replace('.pkl', '_meta.json').replace('.h5', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'model_name': metadata.model_name,
                    'model_type': metadata.model_type,
                    'version': metadata.version,
                    'created_at': metadata.created_at,
                    'random_seed': metadata.random_seed,
                    'features': metadata.features,
                    'metrics': metadata.metrics
                }, f, indent=2)
        
        # Track version
        if name not in self.version_history:
            self.version_history[name] = []
        self.version_history[name].append(filepath)
        
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load model from file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if filepath.endswith('.h5'):
            try:
                from tensorflow.keras.models import load_model
                return load_model(filepath)
            except ImportError:
                raise ImportError("TensorFlow required to load .h5 models")
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def load_metadata(self, filepath: str) -> Optional[Dict]:
        """Load model metadata"""
        meta_path = filepath.replace('.pkl', '_meta.json').replace('.h5', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        return None
    
    def check_version_compatibility(self, metadata: Dict,
                                    expected_version: str) -> bool:
        """Check if model version is compatible"""
        if metadata is None:
            return True
        return metadata.get('version', '') == expected_version
    
    def get_latest_model(self, name: str) -> Optional[str]:
        """Get path to latest version of a model"""
        if name in self.version_history and self.version_history[name]:
            return self.version_history[name][-1]
        
        # Search in models directory
        models = [f for f in os.listdir(self.models_dir) 
                  if f.startswith(name) and (f.endswith('.pkl') or f.endswith('.h5'))]
        if models:
            models.sort()
            return os.path.join(self.models_dir, models[-1])
        return None
    
    def rollback_to_version(self, name: str, version_idx: int) -> Optional[str]:
        """Rollback to a previous version"""
        if name in self.version_history:
            versions = self.version_history[name]
            if 0 <= version_idx < len(versions):
                return versions[version_idx]
        return None


class DataPersistence:
    """Handles saving and loading of processed data"""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data', 'processed')
        
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_features(self, X, y, name: str,
                      version: str = '1.0',
                      random_seed: int = 42) -> str:
        """Save feature matrix and target vector"""
        import pandas as pd
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            'X': X,
            'y': y,
            'version': version,
            'random_seed': random_seed,
            'created_at': timestamp
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load_features(self, filepath: str) -> Dict:
        """Load saved features"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ReproducibilityManager:
    """Manages random seeds for reproducibility"""
    
    def __init__(self):
        self.seeds = {}
    
    def set_seed(self, name: str, seed: int) -> None:
        """Set and store random seed"""
        import numpy as np
        import random
        
        self.seeds[name] = seed
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def get_seed(self, name: str) -> Optional[int]:
        """Get stored seed"""
        return self.seeds.get(name)
    
    def save_seeds(self, filepath: str) -> None:
        """Save all seeds to file"""
        with open(filepath, 'w') as f:
            json.dump(self.seeds, f)
    
    def load_seeds(self, filepath: str) -> None:
        """Load seeds from file"""
        with open(filepath, 'r') as f:
            self.seeds = json.load(f)
