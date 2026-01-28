"""
Prediction Reporter Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import os


@dataclass
class PredictionRow:
    """Single prediction row"""
    country: str
    gold_predicted: float
    gold_ci_lower: float
    gold_ci_upper: float
    total_predicted: float
    total_ci_lower: float
    total_ci_upper: float
    rank: int


class PredictionReporter:
    """
    Generates prediction reports and exports.
    """
    
    REQUIRED_COLUMNS = [
        'country', 'gold_predicted', 'gold_ci_lower', 'gold_ci_upper',
        'total_predicted', 'total_ci_lower', 'total_ci_upper', 'rank'
    ]
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'output')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_prediction_table(self, 
                                predictions: Dict[str, float],
                                confidence_intervals: Dict[str, tuple] = None,
                                gold_predictions: Dict[str, float] = None,
                                gold_ci: Dict[str, tuple] = None) -> pd.DataFrame:
        """
        Create prediction table with all required columns.
        """
        rows = []
        
        # Sort by predicted total medals
        sorted_countries = sorted(predictions.keys(), 
                                  key=lambda x: predictions[x], 
                                  reverse=True)
        
        for rank, country in enumerate(sorted_countries, 1):
            total_pred = predictions[country]
            
            # Get confidence intervals
            if confidence_intervals and country in confidence_intervals:
                total_lower, total_upper = confidence_intervals[country]
            else:
                total_lower = total_pred * 0.9
                total_upper = total_pred * 1.1
            
            # Get gold predictions
            if gold_predictions and country in gold_predictions:
                gold_pred = gold_predictions[country]
            else:
                gold_pred = total_pred * 0.3  # Estimate
            
            if gold_ci and country in gold_ci:
                gold_lower, gold_upper = gold_ci[country]
            else:
                gold_lower = gold_pred * 0.9
                gold_upper = gold_pred * 1.1
            
            rows.append({
                'country': country,
                'gold_predicted': round(gold_pred, 1),
                'gold_ci_lower': round(gold_lower, 1),
                'gold_ci_upper': round(gold_upper, 1),
                'total_predicted': round(total_pred, 1),
                'total_ci_lower': round(total_lower, 1),
                'total_ci_upper': round(total_upper, 1),
                'rank': rank
            })
        
        return pd.DataFrame(rows)

    def validate_table_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that table has all required columns.
        """
        return all(col in df.columns for col in self.REQUIRED_COLUMNS)
    
    def save_to_csv(self, df: pd.DataFrame, 
                    filename: str = 'predictions.csv',
                    encoding: str = 'utf-8') -> str:
        """
        Save predictions to CSV file.
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding=encoding)
        return filepath
    
    def generate_summary_report(self, 
                                predictions_df: pd.DataFrame,
                                metrics: Dict[str, float] = None) -> str:
        """
        Generate summary report text.
        """
        report = []
        report.append("=" * 60)
        report.append("OLYMPIC MEDAL PREDICTION REPORT - 2028 LOS ANGELES")
        report.append("=" * 60)
        report.append("")
        
        # Top 10 predictions
        report.append("TOP 10 PREDICTED MEDAL WINNERS:")
        report.append("-" * 40)
        
        top10 = predictions_df.head(10)
        for _, row in top10.iterrows():
            report.append(
                f"{row['rank']:2d}. {row['country']}: "
                f"Gold={row['gold_predicted']:.0f} [{row['gold_ci_lower']:.0f}-{row['gold_ci_upper']:.0f}], "
                f"Total={row['total_predicted']:.0f} [{row['total_ci_lower']:.0f}-{row['total_ci_upper']:.0f}]"
            )
        
        report.append("")
        
        # Model performance
        if metrics:
            report.append("MODEL PERFORMANCE:")
            report.append("-" * 40)
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, report: str, 
                    filename: str = 'summary_report.txt') -> str:
        """
        Save report to text file.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        return filepath
