"""
Configuration file for Olympic Medal Prediction System
Contains all hyperparameters, file paths, and model settings
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


@dataclass
class DataConfig:
    """Data file paths and loading configuration"""
    medals_file: str = 'summerOly_medal_counts.csv'
    hosts_file: str = 'summerOly_hosts.csv'
    programs_file: str = 'summerOly_programs.csv'
    athletes_file: str = 'summerOly_athletes.csv'
    
    # Data validation
    required_medal_columns: List[str] = field(default_factory=lambda: [
        'year', 'country', 'gold', 'silver', 'bronze', 'total'
    ])
    required_host_columns: List[str] = field(default_factory=lambda: [
        'year', 'host_country', 'host_city'
    ])
    
    # Missing value handling
    imputation_strategy: str = 'interpolate'  # 'interpolate', 'knn', 'mean'
    knn_neighbors: int = 5
    
    # Outlier detection
    outlier_method: str = 'zscore'  # 'zscore', 'iqr'
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Lagged features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Rolling statistics
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5])
    
    # Feature lists
    historical_features: List[str] = field(default_factory=lambda: [
        'medals_lag1', 'medals_lag2', 'medals_lag3',
        'medals_ma3', 'medals_ma5', 'medals_std3'
    ])
    economic_features: List[str] = field(default_factory=lambda: [
        'gdp_per_capita', 'gdp_growth', 'gdp_log'
    ])
    event_features: List[str] = field(default_factory=lambda: [
        'events_participated', 'athletes_count', 'events_ratio'
    ])
    interaction_features: List[str] = field(default_factory=lambda: [
        'gdp_pop_interaction', 'investment_events'
    ])


@dataclass
class ARIMAConfig:
    """ARIMA model configuration"""
    p_range: Tuple[int, int] = (0, 5)
    d_range: Tuple[int, int] = (0, 2)
    q_range: Tuple[int, int] = (0, 5)
    criterion: str = 'aic'  # 'aic' or 'bic'
    adf_significance: float = 0.05


@dataclass
class ProphetConfig:
    """Prophet model configuration"""
    yearly_seasonality: bool = True
    changepoint_prior_scale: float = 0.05
    seasonality_mode: str = 'additive'
    regressors: List[str] = field(default_factory=lambda: ['gdp', 'population'])


@dataclass
class LSTMConfig:
    """LSTM model configuration"""
    sequence_length: int = 3
    lstm_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2
    dense_units: int = 32
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5


@dataclass
class XGBoostConfig:
    """XGBoost model configuration"""
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


@dataclass
class LightGBMConfig:
    """LightGBM model configuration"""
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


@dataclass
class RandomForestConfig:
    """Random Forest model configuration"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'


@dataclass
class EnsembleConfig:
    """Ensemble model configuration"""
    # Time series model weights
    arima_weight: float = 0.15
    prophet_weight: float = 0.15
    lstm_weight: float = 0.20
    
    # ML model weights
    xgboost_weight: float = 0.25
    lightgbm_weight: float = 0.15
    random_forest_weight: float = 0.10
    
    # High variance threshold
    variance_threshold: float = 0.20  # 20% of mean prediction


@dataclass
class HostEffectConfig:
    """Host country effect configuration"""
    historical_start_year: int = 2000
    historical_end_year: int = 2024
    expected_effect_range: Tuple[float, float] = (0.15, 0.25)  # 15-25%
    usa_2028_host: bool = True


@dataclass
class BayesianConfig:
    """Bayesian uncertainty quantification configuration"""
    n_samples: int = 2000
    n_tune: int = 1000
    confidence_level: float = 0.95
    wide_ci_threshold: float = 0.20  # 20% of point estimate


@dataclass
class ValidationConfig:
    """Validation and backtesting configuration"""
    n_cv_folds: int = 5
    backtest_years: List[int] = field(default_factory=lambda: [2020, 2024])
    performance_threshold_mape: float = 0.15  # 15% MAPE threshold


@dataclass
class SensitivityConfig:
    """Sensitivity analysis configuration"""
    gdp_variation: float = 0.02  # ±2%
    investment_variation: float = 0.10  # ±10%
    events_variation: int = 5  # ±5 events
    host_effect_variation: float = 0.05  # ±5%
    high_sensitivity_threshold: float = 0.10  # 10% prediction change


@dataclass
class OutputConfig:
    """Output and visualization configuration"""
    figure_dpi: int = 300
    figure_format: str = 'png'
    csv_encoding: str = 'utf-8'
    min_charts: int = 10


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    host_effect: HostEffectConfig = field(default_factory=HostEffectConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Random seed for reproducibility
    random_seed: int = 42


# Global configuration instance
config = Config()
