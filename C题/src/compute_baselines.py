"""
Compute Baseline Model Performance for MCM Paper
Generates real performance metrics for Simple Average and Historical Mean baselines
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_DIR
from data.loader import DataLoader
from data.merger import DataMerger
from features.engineer import FeatureEngineer
from validation.evaluator import ModelEvaluator


class BaselineModels:
    """Baseline models for comparison"""
    
    @staticmethod
    def simple_average(y_train):
        """Return global average of training data"""
        return np.mean(y_train)
    
    @staticmethod
    def historical_mean_per_country(df_train, df_test, country_col='country', target='total'):
        """Return per-country historical mean"""
        country_means = df_train.groupby(country_col)[target].mean()
        predictions = df_test[country_col].map(country_means)
        # Fill missing countries with global mean
        global_mean = df_train[target].mean()
        predictions = predictions.fillna(global_mean)
        return predictions.values


def main():
    print("=" * 70)
    print("BASELINE MODEL PERFORMANCE COMPUTATION")
    print("=" * 70)
    
    # Load data
    loader = DataLoader(DATA_DIR)
    data = loader.load_all()
    
    merger = DataMerger()
    result = merger.merge_datasets(
        data['medals'], data['hosts'],
        data.get('programs'), data.get('athletes')
    )
    
    df = result.data
    
    # Standardize columns
    column_mapping = {
        'NOC': 'country', 'Gold': 'gold', 'Silver': 'silver',
        'Bronze': 'bronze', 'Total': 'total', 'Year': 'year', 'Rank': 'rank'
    }
    df = df.rename(columns=column_mapping)
    
    print(f"Loaded {len(df)} records, {df['year'].nunique()} Olympics")
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df, target_col='total', group_by='country')
    df_features = df_features.dropna()
    
    print(f"After feature engineering: {len(df_features)} records")
    
    # Time-series cross-validation split (same as main model)
    test_years = [2016, 2020, 2024]
    evaluator = ModelEvaluator()
    
    # Store results
    results = {
        'Simple Average': {'r2': [], 'rmse': [], 'mae': []},
        'Historical Mean': {'r2': [], 'rmse': [], 'mae': []}
    }
    
    print("\n" + "-" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("-" * 70)
    
    for test_year in test_years:
        train_df = df_features[df_features['year'] < test_year]
        test_df = df_features[df_features['year'] == test_year]
        
        if len(test_df) == 0:
            continue
        
        y_train = train_df['total'].values
        y_test = test_df['total'].values
        
        print(f"\nFold: Test Year = {test_year}")
        print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Baseline 1: Simple Average
        simple_avg_pred = np.full(len(y_test), np.mean(y_train))
        metrics_simple = evaluator.evaluate(y_test, simple_avg_pred)
        results['Simple Average']['r2'].append(metrics_simple.r2)
        results['Simple Average']['rmse'].append(metrics_simple.rmse)
        results['Simple Average']['mae'].append(metrics_simple.mae)
        print(f"  Simple Average:  R²={metrics_simple.r2:.4f}, RMSE={metrics_simple.rmse:.2f}, MAE={metrics_simple.mae:.2f}")
        
        # Baseline 2: Historical Mean per Country
        hist_mean_pred = BaselineModels.historical_mean_per_country(
            train_df, test_df, country_col='country', target='total'
        )
        metrics_hist = evaluator.evaluate(y_test, hist_mean_pred)
        results['Historical Mean']['r2'].append(metrics_hist.r2)
        results['Historical Mean']['rmse'].append(metrics_hist.rmse)
        results['Historical Mean']['mae'].append(metrics_hist.mae)
        print(f"  Historical Mean: R²={metrics_hist.r2:.4f}, RMSE={metrics_hist.rmse:.2f}, MAE={metrics_hist.mae:.2f}")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL BASELINE PERFORMANCE (Average across folds)")
    print("=" * 70)
    
    final_results = {}
    for model_name, metrics_dict in results.items():
        avg_r2 = np.mean(metrics_dict['r2'])
        avg_rmse = np.mean(metrics_dict['rmse'])
        avg_mae = np.mean(metrics_dict['mae'])
        final_results[model_name] = {'r2': avg_r2, 'rmse': avg_rmse, 'mae': avg_mae}
        print(f"\n{model_name}:")
        print(f"  R² = {avg_r2:.3f}")
        print(f"  RMSE = {avg_rmse:.2f}")
        print(f"  MAE = {avg_mae:.2f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("LATEX TABLE OUTPUT (for paper)")
    print("=" * 70)
    
    print("""
\\begin{table}[h]
\\centering
\\caption{Individual Model vs Ensemble Performance (with Baselines)}
\\label{tab:model_comparison}
\\begin{tabular}{ccccc}
\\toprule
\\textbf{Model} & \\textbf{R$^2$} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{Notes} \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Baselines}} \\\\""")
    
    r = final_results['Simple Average']
    print(f"Simple Average & {r['r2']:.3f} & {r['rmse']:.2f} & {r['mae']:.2f} & Na\\\"ive baseline \\\\")
    r = final_results['Historical Mean']
    print(f"Historical Mean & {r['r2']:.3f} & {r['rmse']:.2f} & {r['mae']:.2f} & Per-country mean \\\\")
    
    print("""\\midrule
\\multicolumn{5}{l}{\\textit{Individual Models}} \\\\
ARIMA & 0.823 & 7.42 & 4.31 & Linear temporal patterns \\\\
... (rest of table)
\\bottomrule
\\end{tabular}
\\end{table}""")
    
    # Compute improvement
    ensemble_r2 = 0.947
    baseline_r2 = final_results['Simple Average']['r2']
    improvement = ((ensemble_r2 - baseline_r2) / baseline_r2) * 100
    print(f"\n\nEnsemble Improvement over Simple Average Baseline: +{improvement:.1f}%")
    
    return final_results


if __name__ == "__main__":
    main()
