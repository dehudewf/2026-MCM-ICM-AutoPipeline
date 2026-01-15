"""
Olympic Medal Prediction System - Main Entry Point
2028 Los Angeles Olympics Prediction
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, DATA_DIR, OUTPUT_DIR, MODELS_DIR


def load_and_prepare_data() -> Optional[pd.DataFrame]:
    """Load and prepare all data"""
    from data.loader import DataLoader
    from data.merger import DataMerger
    
    print("Loading data...")
    loader = DataLoader(DATA_DIR)
    
    try:
        data = loader.load_all()
        print(f"  Loaded {len(data)} datasets")
    except Exception as e:
        print(f"  Warning: Could not load all data: {e}")
        return None
    
    # Merge datasets
    print("Merging datasets...")
    merger = DataMerger()
    result = merger.merge_datasets(
        data['medals'],
        data['hosts'],
        data.get('programs'),
        data.get('athletes')
    )
    
    if not result.success:
        print(f"  Merge failed: {result.error_message}")
        return None
    
    print(f"  Merged dataset: {result.rows_after} rows")
    return result.data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    from data.preprocessor import DataPreprocessor
    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = preprocessor.impute_missing(df, numeric_cols, strategy='mean')
    
    print(f"  Preprocessed {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features"""
    from features.engineer import FeatureEngineer
    
    print("Engineering features...")
    engineer = FeatureEngineer()
    
    df = engineer.create_all_features(df, target_col='total', group_by='country')
    
    print(f"  Created {len(df.columns)} features")
    return df


def train_models(df: pd.DataFrame, features: list, target: str) -> Dict[str, Any]:
    """Train all prediction models"""
    from models.xgboost_model import XGBoostModel, XGBoostConfig
    from models.lightgbm_model import LightGBMModel, LightGBMConfig
    from models.random_forest_model import RandomForestModel, RandomForestConfig
    
    print("Training models...")
    models = {}
    
    # Prepare data
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    # XGBoost
    print("  Training XGBoost...")
    try:
        xgb_config = XGBoostConfig(n_estimators=100, max_depth=4)
        xgb_model = XGBoostModel(xgb_config)
        xgb_model.fit(X, y)
        models['xgboost'] = xgb_model
    except Exception as e:
        print(f"    XGBoost failed: {e}")
    
    # LightGBM
    print("  Training LightGBM...")
    try:
        lgb_config = LightGBMConfig(n_estimators=100, max_depth=4)
        lgb_model = LightGBMModel(lgb_config)
        lgb_model.fit(X, y)
        models['lightgbm'] = lgb_model
    except Exception as e:
        print(f"    LightGBM failed: {e}")
    
    # Random Forest
    print("  Training Random Forest...")
    try:
        rf_config = RandomForestConfig(n_estimators=100, max_depth=6)
        rf_model = RandomForestModel(rf_config)
        rf_model.fit(X, y)
        models['random_forest'] = rf_model
    except Exception as e:
        print(f"    Random Forest failed: {e}")
    
    print(f"  Trained {len(models)} models")
    return models


def make_ensemble_predictions(models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Get predictions from all models"""
    from models.ensemble import EnsemblePredictor, EnsembleConfig
    
    print("Making ensemble predictions...")
    predictions = {}
    
    for name, model in models.items():
        try:
            predictions[name] = model.predict(X)
        except Exception as e:
            print(f"  {name} prediction failed: {e}")
    
    # Ensemble
    if predictions:
        config = EnsembleConfig()
        ensemble = EnsemblePredictor(config)
        ensemble_pred = ensemble.predict(predictions)
        predictions['ensemble'] = ensemble_pred
    
    return predictions


def apply_adjustments(predictions: np.ndarray, 
                      countries: list,
                      host_country: str = 'USA') -> np.ndarray:
    """Apply host effect and other adjustments"""
    from optimization.host_effect import HostEffectAdjuster
    
    print("Applying adjustments...")
    adjuster = HostEffectAdjuster()
    
    # Apply host effect
    adjusted = predictions.copy()
    for i, country in enumerate(countries):
        if country == host_country:
            adjusted[i] = adjuster.apply_multiplicative_adjustment(
                predictions[i], 0.20  # 20% host effect
            )
    
    return adjusted


def compute_uncertainty(predictions: Dict[str, np.ndarray]) -> Dict:
    """Compute confidence intervals"""
    from optimization.bayesian import BayesianUncertainty
    
    print("Computing uncertainty...")
    uncertainty = BayesianUncertainty()
    result = uncertainty.compute_confidence_intervals(predictions)
    
    return {
        'point_estimate': result.point_estimate,
        'lower_bound': result.lower_bound,
        'upper_bound': result.upper_bound,
        'high_uncertainty': result.high_uncertainty_indices
    }


def generate_outputs(predictions_df: pd.DataFrame, metrics: Dict = None):
    """Generate all outputs"""
    from output.reporter import PredictionReporter
    from output.visualizer import Visualizer
    
    print("Generating outputs...")
    
    # Reporter
    reporter = PredictionReporter(OUTPUT_DIR)
    
    # Save predictions
    csv_path = reporter.save_to_csv(predictions_df, 'predictions_2028.csv')
    print(f"  Saved predictions to {csv_path}")
    
    # Generate report
    report = reporter.generate_summary_report(predictions_df, metrics)
    report_path = reporter.save_report(report, 'summary_report.txt')
    print(f"  Saved report to {report_path}")
    
    # Visualizations
    visualizer = Visualizer(os.path.join(OUTPUT_DIR, 'figures'))
    
    fig = visualizer.plot_predictions(predictions_df, top_n=20)
    if fig:
        fig_path = visualizer.save_figure(fig, 'predictions_2028.png', dpi=300)
        print(f"  Saved visualization to {fig_path}")


def run_prediction_pipeline(mode: str = 'predict'):
    """Run the complete prediction pipeline"""
    print("=" * 60)
    print("Olympic Medal Prediction System")
    print("2028 Los Angeles Olympics")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        print("\nPlease place data files in the 'data' directory:")
        print("  - summerOly_medal_counts.csv")
        print("  - summerOly_hosts.csv")
        print("  - summerOly_programs.csv (optional)")
        print("  - summerOly_athletes.csv (optional)")
        return None
    
    # Preprocess
    df = preprocess_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Define features
    feature_cols = [col for col in df.columns if col not in 
                    ['year', 'country', 'gold', 'silver', 'bronze', 'total', 
                     'host_country', 'is_host']]
    feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]
    
    if mode == 'train' or mode == 'predict':
        # Train models
        models = train_models(df, feature_cols, 'total')
        
        if not models:
            print("No models trained successfully")
            return None
        
        # Prepare prediction data (latest year for each country)
        latest_data = df.groupby('country').last().reset_index()
        X_pred = latest_data[feature_cols].fillna(0)
        countries = latest_data['country'].tolist()
        
        # Make predictions
        predictions = make_ensemble_predictions(models, X_pred)
        
        if 'ensemble' not in predictions:
            print("Ensemble prediction failed")
            return None
        
        # Apply adjustments
        adjusted = apply_adjustments(predictions['ensemble'], countries, 'USA')
        
        # Compute uncertainty
        uncertainty = compute_uncertainty(predictions)
        
        # Create output dataframe
        from output.reporter import PredictionReporter
        reporter = PredictionReporter()
        
        pred_dict = {c: p for c, p in zip(countries, adjusted)}
        ci_dict = {c: (l, u) for c, l, u in zip(
            countries, uncertainty['lower_bound'], uncertainty['upper_bound']
        )}
        
        predictions_df = reporter.create_prediction_table(pred_dict, ci_dict)
        
        # Generate outputs
        generate_outputs(predictions_df)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        return predictions_df
    
    elif mode == 'validate':
        from validation.backtest import BacktestValidator
        
        print("\nRunning validation...")
        validator = BacktestValidator(n_splits=3)
        
        # Time series splits
        splits = validator.time_series_split(df)
        print(f"  Created {len(splits)} validation splits")
        
        return df
    
    elif mode == 'analyze':
        from analysis.regional import RegionalAnalyzer
        from analysis.event_impact import EventImpactAnalyzer
        
        print("\nRunning analysis...")
        
        # Regional analysis
        regional = RegionalAnalyzer()
        sa_data = regional.get_south_american_data(df)
        print(f"  South American countries: {sa_data['country'].nunique()}")
        
        # Event impact
        if 'events_participated' in df.columns:
            event_analyzer = EventImpactAnalyzer()
            corr = event_analyzer.compute_correlations(
                df['events_participated'].values,
                df['total'].values
            )
            print(f"  Event-Medal correlation: {corr.pearson_r:.3f}")
        
        return df
    
    return df


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description='Olympic Medal Prediction System'
    )
    parser.add_argument(
        '--mode', 
        choices=['train', 'predict', 'validate', 'analyze'],
        default='predict',
        help='Operation mode'
    )
    
    args = parser.parse_args()
    run_prediction_pipeline(args.mode)


if __name__ == "__main__":
    main()
