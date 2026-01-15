# Design Document

## Overview

The Olympic Medal Prediction System is a multi-layered ensemble architecture that combines time series forecasting, machine learning regression, and optimization adjustments to predict medal counts for the 2028 Los Angeles Olympics. The system processes historical Olympic data (1896-2024), extracts predictive features, trains multiple models, and produces final predictions with quantified uncertainty intervals.

The architecture follows a three-layer design:
- **Layer 1 (Time Series)**: Captures temporal trends using ARIMA, Prophet, and LSTM
- **Layer 2 (ML Regression)**: Models feature relationships using XGBoost, LightGBM, and Random Forest
- **Layer 3 (Optimization)**: Applies domain-specific adjustments (host effect, event changes, Bayesian uncertainty)

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
│  CSV Files → Loader → Validator → Cleaner → Feature Engine  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  Layer 1: TS    │            │  Layer 2: ML    │
│  - ARIMA        │            │  - XGBoost      │
│  - Prophet      │            │  - LightGBM     │
│  - LSTM         │            │  - RandomForest │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                ┌────────▼────────┐
                │  Ensemble       │
                │  Fusion         │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  Layer 3:       │
                │  Optimization   │
                │  - Host Effect  │
                │  - Event Adjust │
                │  - Bayesian CI  │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  Output         │
                │  - Predictions  │
                │  - Visualizations│
                │  - Reports      │
                └─────────────────┘
```

### Data Flow

1. **Input Stage**: Load 4 CSV files (medals, hosts, programs, athletes)
2. **Preprocessing Stage**: Clean, validate, merge datasets
3. **Feature Engineering Stage**: Create lagged, rolling, economic, and interaction features
4. **Training Stage**: Train 6 models in parallel (3 TS + 3 ML)
5. **Ensemble Stage**: Combine predictions using weighted averaging
6. **Adjustment Stage**: Apply host effect, event changes, compute confidence intervals
7. **Output Stage**: Generate predictions table, visualizations, validation reports

## Components and Interfaces

### 1. Data Module

**DataLoader**
```python
class DataLoader:
    def load_medals(path: str) -> pd.DataFrame
    def load_hosts(path: str) -> pd.DataFrame
    def load_programs(path: str) -> pd.DataFrame
    def load_athletes(path: str) -> pd.DataFrame
    def merge_datasets(medals, hosts, programs, athletes) -> pd.DataFrame
```

**DataCleaner**
```python
class DataCleaner:
    def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame
    def detect_outliers(df: pd.DataFrame, method: str) -> List[int]
    def remove_outliers(df: pd.DataFrame, indices: List[int]) -> pd.DataFrame
    def validate_data_integrity(df: pd.DataFrame) -> ValidationReport
```

**FeatureEngineer**
```python
class FeatureEngineer:
    def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame
    def create_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame
    def create_economic_features(df: pd.DataFrame) -> pd.DataFrame
    def create_interaction_features(df: pd.DataFrame, pairs: List[Tuple]) -> pd.DataFrame
    def create_host_features(df: pd.DataFrame) -> pd.DataFrame
```

### 2. Time Series Module

**ARIMAModel**
```python
class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int])
    def fit(self, data: pd.Series) -> None
    def predict(self, steps: int) -> np.ndarray
    def get_aic_bic(self) -> Tuple[float, float]
```

**ProphetModel**
```python
class ProphetModel:
    def __init__(self, seasonality_mode: str, changepoint_prior_scale: float)
    def add_regressor(self, name: str) -> None
    def fit(self, df: pd.DataFrame) -> None
    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame
```

**LSTMModel**
```python
class LSTMModel:
    def __init__(self, sequence_length: int, units: List[int], dropout: float)
    def build_model(self) -> tf.keras.Model
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, callbacks: List) -> History
    def predict(self, X: np.ndarray) -> np.ndarray
```

### 3. Machine Learning Module

**XGBoostModel**
```python
class XGBoostModel:
    def __init__(self, params: Dict)
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: List) -> None
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_feature_importance(self) -> pd.DataFrame
```

**LightGBMModel**
```python
class LightGBMModel:
    def __init__(self, params: Dict)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_feature_importance(self) -> pd.DataFrame
```

**RandomForestModel**
```python
class RandomForestModel:
    def __init__(self, n_estimators: int, max_depth: int)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_feature_importance(self) -> pd.DataFrame
```

### 4. Ensemble Module

**EnsemblePredictor**
```python
class EnsemblePredictor:
    def __init__(self, models: List[Model], weights: Dict[str, float])
    def add_model(self, name: str, model: Model, weight: float) -> None
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]
```

**StackingEnsemble**
```python
class StackingEnsemble:
    def __init__(self, base_models: List[Tuple[str, Model]], meta_model: Model)
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int) -> None
    def predict(self, X: pd.DataFrame) -> np.ndarray
```

### 5. Optimization Module

**HostEffectAdjuster**
```python
class HostEffectAdjuster:
    def calculate_historical_effect(self, df: pd.DataFrame) -> float
    def apply_adjustment(self, predictions: np.ndarray, host_countries: List[str], effect: float) -> np.ndarray
```

**EventChangeAdjuster**
```python
class EventChangeAdjuster:
    def estimate_event_impact(self, country: str, new_events: List[str], removed_events: List[str]) -> float
    def apply_adjustment(self, predictions: np.ndarray, adjustments: Dict[str, float]) -> np.ndarray
```

**BayesianUncertainty**
```python
class BayesianUncertainty:
    def __init__(self, prior_params: Dict)
    def fit(self, X: pd.DataFrame, y: pd.Series, n_samples: int) -> None
    def predict_with_intervals(self, X: pd.DataFrame, confidence: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

### 6. Validation Module

**BacktestValidator**
```python
class BacktestValidator:
    def time_series_split(self, df: pd.DataFrame, n_splits: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]
    def evaluate_model(self, model: Model, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]
    def cross_validate(self, model: Model, df: pd.DataFrame, n_splits: int) -> pd.DataFrame
```

**SensitivityAnalyzer**
```python
class SensitivityAnalyzer:
    def vary_feature(self, X: pd.DataFrame, feature: str, range: np.ndarray) -> pd.DataFrame
    def compute_impact(self, model: Model, variations: pd.DataFrame) -> pd.DataFrame
    def plot_sensitivity(self, results: pd.DataFrame, feature: str) -> None
```

### 7. Output Module

**PredictionReporter**
```python
class PredictionReporter:
    def create_prediction_table(self, predictions: Dict, confidence_intervals: Dict) -> pd.DataFrame
    def save_to_csv(self, df: pd.DataFrame, path: str) -> None
    def generate_summary_report(self, predictions: pd.DataFrame, metrics: Dict) -> str
```

**Visualizer**
```python
class Visualizer:
    def plot_historical_trends(self, df: pd.DataFrame, countries: List[str]) -> plt.Figure
    def plot_predictions(self, predictions: pd.DataFrame) -> plt.Figure
    def plot_feature_importance(self, importance: pd.DataFrame) -> plt.Figure
    def plot_sensitivity_analysis(self, sensitivity: pd.DataFrame) -> plt.Figure
    def save_figure(self, fig: plt.Figure, path: str, dpi: int) -> None
```

## Data Models

### Core Data Structures

**MedalData**
```python
@dataclass
class MedalData:
    year: int
    country: str
    gold: int
    silver: int
    bronze: int
    total: int
    is_host: bool
    gdp: float
    population: int
    events_participated: int
```

**Prediction**
```python
@dataclass
class Prediction:
    country: str
    gold_predicted: float
    gold_ci_lower: float
    gold_ci_upper: float
    total_predicted: float
    total_ci_lower: float
    total_ci_upper: float
    rank: int
```

**ModelMetrics**
```python
@dataclass
class ModelMetrics:
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    training_time: float
```

**FeatureSet**
```python
@dataclass
class FeatureSet:
    historical_features: List[str]  # medals_lag1, medals_lag2, etc.
    economic_features: List[str]    # gdp_per_capita, gdp_growth, etc.
    event_features: List[str]       # events_participated, athletes_count, etc.
    interaction_features: List[str] # gdp_pop_interaction, etc.
    host_features: List[str]        # host_country, host_effect, etc.
```

### Database Schema (if persistence needed)

```sql
-- Historical medals table
CREATE TABLE medals (
    id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    country VARCHAR(3) NOT NULL,
    gold INTEGER,
    silver INTEGER,
    bronze INTEGER,
    total INTEGER,
    UNIQUE(year, country)
);

-- Predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    country VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    gold_predicted FLOAT,
    gold_ci_lower FLOAT,
    gold_ci_upper FLOAT,
    total_predicted FLOAT,
    total_ci_lower FLOAT,
    total_ci_upper FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    validation_year INTEGER,
    mae FLOAT,
    rmse FLOAT,
    mape FLOAT,
    r2 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Data Integrity Validation
*For any* loaded dataset with integrity issues (missing values, type mismatches, invalid ranges), the validation system should detect and report all issues without false negatives.
**Validates: Requirements 1.2**

### Property 2: Imputation Completeness
*For any* dataset with missing values, after applying imputation strategies, the resulting dataset should contain zero missing values in all required columns.
**Validates: Requirements 1.3**

### Property 3: Outlier Detection Correctness
*For any* dataset and outlier detection method (Z-score or IQR), values flagged as outliers should satisfy the mathematical definition of the method (e.g., |z| > 3 for Z-score).
**Validates: Requirements 1.4**

### Property 4: Merge Preservation
*For any* set of dataframes with common keys (year, country), the merged dataset should preserve all matching records and the number of rows should equal the number of unique key combinations present in all input dataframes.
**Validates: Requirements 1.5**

### Property 5: Lagged Feature Correctness
*For any* time series and lag value k, the lagged feature at position t should equal the original value at position t-k, and the first k values should be NaN or appropriately handled.
**Validates: Requirements 2.1**

### Property 6: Rolling Statistics Accuracy
*For any* sequence of values and window size w, the rolling mean at position t should equal the arithmetic mean of values from position t-w+1 to t.
**Validates: Requirements 2.2**

### Property 7: Economic Feature Calculation
*For any* GDP and population values, the calculated GDP per capita should equal GDP divided by population, and log-transformed GDP should equal ln(GDP + 1).
**Validates: Requirements 2.3**

### Property 8: Interaction Feature Multiplication
*For any* two features A and B, the interaction feature should equal the element-wise product A × B for all observations.
**Validates: Requirements 2.4**

### Property 9: Host Indicator Correctness
*For any* country-year pair, the host indicator should be 1 if and only if that country hosted the Olympics in that year, otherwise 0.
**Validates: Requirements 2.5**

### Property 10: Stationarity Testing
*For any* time series used in ARIMA modeling, the ADF test should be performed, and if the p-value > 0.05, the series should be differenced before model fitting.
**Validates: Requirements 3.1**

### Property 11: Feature Importance Normalization
*For any* trained tree-based model, the extracted feature importance scores should be non-negative and sum to approximately 1.0 (within numerical precision).
**Validates: Requirements 4.4**

### Property 12: Evaluation Metrics Validity
*For any* set of predictions and actual values, computed metrics should satisfy: MAE ≥ 0, RMSE ≥ MAE, MAPE ≥ 0, and R² ≤ 1.
**Validates: Requirements 4.5**

### Property 13: Weighted Average Correctness
*For any* set of model predictions P₁, P₂, ..., Pₙ with weights w₁, w₂, ..., wₙ (where Σwᵢ = 1), the ensemble prediction should equal Σ(wᵢ × Pᵢ).
**Validates: Requirements 5.4**

### Property 14: High Variance Flagging
*For any* set of model predictions with standard deviation exceeding a threshold (e.g., 20% of mean), the system should flag these predictions for review.
**Validates: Requirements 5.5**

### Property 15: Host Effect Multiplicative Adjustment
*For any* baseline prediction P and host effect rate r, the adjusted prediction should equal P × (1 + r), where r is calculated from historical host country data.
**Validates: Requirements 6.2**

### Property 16: Event Addition Impact
*For any* country and newly added event, the estimated additional medals should be non-negative and based on the country's historical performance in similar events.
**Validates: Requirements 7.1**

### Property 17: Event Removal Impact
*For any* removed event, the adjusted medal prediction should be less than or equal to the original prediction (medals cannot increase when events are removed).
**Validates: Requirements 7.2**

### Property 18: Country-Specific Event Rates
*For any* country, the medal-per-event rate should be calculated exclusively from that country's historical data, not from global averages.
**Validates: Requirements 7.3**

### Property 19: Confidence Interval Coverage
*For any* prediction with 95% confidence interval [L, U], the point estimate should satisfy L ≤ estimate ≤ U, and the interval should be computed from the 2.5th and 97.5th percentiles of the posterior distribution.
**Validates: Requirements 8.1, 8.4**

### Property 20: Wide Interval Flagging
*For any* prediction where (CI_upper - CI_lower) > 0.2 × point_estimate, the system should flag this prediction as having high uncertainty.
**Validates: Requirements 8.5**

### Property 21: Temporal Data Leakage Prevention
*For any* backtesting scenario predicting year Y, the training data should contain only records where year < Y, ensuring no future information leaks into the model.
**Validates: Requirements 9.2, 9.3**

### Property 22: Cross-Validation Metric Aggregation
*For any* k-fold cross-validation, the reported average metric should equal the arithmetic mean of the metric values across all k folds.
**Validates: Requirements 9.4**

### Property 23: Sensitivity Scenario Generation
*For any* baseline feature value V and variation percentage p, the sensitivity analysis should test scenarios with values V × (1 - p) and V × (1 + p).
**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

### Property 24: High Sensitivity Flagging
*For any* feature where a 10% input change causes >10% prediction change, the system should flag this feature as having high sensitivity.
**Validates: Requirements 10.5**

### Property 25: Regional Aggregation Consistency
*For any* region, the sum of country-level predictions should equal the region-level aggregated forecast (within numerical precision).
**Validates: Requirements 11.4**

### Property 26: Correlation Bounds
*For any* two variables, the computed Pearson and Spearman correlation coefficients should be in the range [-1, 1].
**Validates: Requirements 12.1**

### Property 27: Statistical Test Validity
*For any* two samples (before/after coach arrival), the t-test should return a valid p-value in the range [0, 1] and Cohen's d should be calculable.
**Validates: Requirements 13.2, 13.3**

### Property 28: Prediction Output Completeness
*For any* prediction run, the output table should contain all required columns (country, gold_predicted, gold_ci_lower, gold_ci_upper, total_predicted, total_ci_lower, total_ci_upper, rank) for all countries.
**Validates: Requirements 14.1**

### Property 29: File Export Format Correctness
*For any* exported visualization, the file format should be PNG and the DPI should be ≥ 300, verifiable through image metadata.
**Validates: Requirements 14.4**

### Property 30: Model Serialization Format Selection
*For any* trained model, the serialization format should match the model type: pickle for scikit-learn models, h5 for Keras models, ensuring successful deserialization.
**Validates: Requirements 15.1**

### Property 31: Reproducibility with Random Seeds
*For any* two model training runs with identical random seeds and input data, the resulting predictions should be identical (within numerical precision).
**Validates: Requirements 15.4**

## Error Handling

### Data Loading Errors
- **FileNotFoundError**: Raised when CSV files are missing, with clear message indicating which file
- **DataValidationError**: Raised when data integrity checks fail, with detailed report of issues
- **MergeError**: Raised when datasets cannot be merged due to missing keys

### Model Training Errors
- **ConvergenceError**: Raised when ARIMA or LSTM fails to converge, with suggestion to adjust parameters
- **InsufficientDataError**: Raised when time series is too short for requested lag features
- **FeatureError**: Raised when required features are missing or have invalid values

### Prediction Errors
- **ModelNotFittedError**: Raised when attempting to predict before model training
- **InvalidInputError**: Raised when prediction input has wrong shape or missing features
- **UncertaintyError**: Raised when Bayesian sampling fails to converge

### Validation Errors
- **BacktestError**: Raised when backtesting encounters data issues or model failures
- **MetricError**: Raised when evaluation metrics cannot be computed (e.g., division by zero)

### Error Recovery Strategies
1. **Graceful Degradation**: If one model fails, continue with remaining models
2. **Fallback Values**: Use regional averages when country-specific data is unavailable
3. **Logging**: All errors logged with timestamps, context, and stack traces
4. **User Notification**: Clear error messages with actionable suggestions

## Testing Strategy

### Unit Testing
Unit tests verify specific components and edge cases:
- Data loading functions with various CSV formats
- Feature engineering calculations with known inputs/outputs
- Model initialization with valid/invalid parameters
- Metric calculations with edge cases (empty arrays, all zeros, etc.)
- Serialization/deserialization round trips

### Property-Based Testing
Property-based tests verify universal properties across many randomly generated inputs:
- Each correctness property listed above will be implemented as a property-based test
- Tests will use hypothesis library (Python) to generate diverse test cases
- Minimum 100 iterations per property test to ensure thorough coverage
- Each property test will be tagged with format: **Feature: olympic-medal-prediction, Property X: [property description]**

**Property Testing Framework**: pytest with hypothesis plugin

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st
import pytest

@given(
    gdp=st.floats(min_value=1e9, max_value=1e13),
    population=st.integers(min_value=1e6, max_value=1e9)
)
def test_property_7_gdp_per_capita_calculation(gdp, population):
    """
    Feature: olympic-medal-prediction, Property 7: Economic Feature Calculation
    For any GDP and population values, GDP per capita should equal GDP / population
    """
    feature_engineer = FeatureEngineer()
    result = feature_engineer.calculate_gdp_per_capita(gdp, population)
    expected = gdp / population
    assert abs(result - expected) < 1e-6  # numerical precision tolerance
```

### Integration Testing
Integration tests verify component interactions:
- End-to-end pipeline: CSV loading → feature engineering → model training → prediction
- Model ensemble: multiple models → weighted combination → final prediction
- Backtesting workflow: data split → train → predict → evaluate
- Visualization pipeline: predictions → plot generation → file export

### Performance Testing
- Training time benchmarks for each model type
- Memory usage profiling during large dataset processing
- Prediction latency measurements
- Scalability tests with varying dataset sizes

### Validation Testing
- Backtest on 2020 Olympics (train on ≤2016 data)
- Backtest on 2024 Olympics (train on ≤2020 data)
- Cross-validation with 5 folds
- Sensitivity analysis with ±10% input variations
- Comparison against baseline models (simple average, linear trend)

