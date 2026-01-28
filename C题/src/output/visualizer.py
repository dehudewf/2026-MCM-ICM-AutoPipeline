"""
Visualization Module
Creates charts and plots for Olympic medal predictions
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import os


class Visualizer:
    """
    Creates visualizations for Olympic medal predictions.
    """
    
    def __init__(self, output_dir: str = 'output/figures'):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_historical_trends(self, df: pd.DataFrame,
                                countries: List[str],
                                medal_col: str = 'total') -> Any:
        """
        Plot historical medal trends for countries.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for country in countries:
                country_data = df[df['country'] == country]
                ax.plot(country_data['year'], country_data[medal_col], 
                       label=country, marker='o')
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Medals')
            ax.set_title('Historical Medal Trends')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
        except ImportError:
            return None

    def plot_predictions(self, predictions: pd.DataFrame,
                         top_n: int = 20) -> Any:
        """
        Plot predicted medal counts.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            top_countries = predictions.nlargest(top_n, 'total_predicted')
            
            ax.barh(top_countries['country'], top_countries['total_predicted'])
            ax.set_xlabel('Predicted Total Medals')
            ax.set_title(f'Top {top_n} Countries - 2028 Medal Predictions')
            ax.invert_yaxis()
            
            return fig
        except ImportError:
            return None
    
    def plot_feature_importance(self, importance: pd.DataFrame,
                                 top_n: int = 15) -> Any:
        """
        Plot feature importance scores.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            top_features = importance.nlargest(top_n, 'importance')
            
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance')
            ax.invert_yaxis()
            
            return fig
        except ImportError:
            return None
    
    def plot_sensitivity_analysis(self, sensitivity: pd.DataFrame) -> Any:
        """
        Plot sensitivity analysis results.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(sensitivity['feature'], sensitivity['impact'])
            ax.set_xlabel('Feature')
            ax.set_ylabel('Prediction Impact (%)')
            ax.set_title('Sensitivity Analysis')
            ax.tick_params(axis='x', rotation=45)
            
            return fig
        except ImportError:
            return None
    
    def save_figure(self, fig: Any, filename: str, dpi: int = 300) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution (default 300)
            
        Returns:
            Path to saved file
        """
        if fig is None:
            return ""
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        return filepath
    
    def get_figure_dpi(self, filepath: str) -> int:
        """
        Get DPI of saved figure.
        """
        try:
            from PIL import Image
            img = Image.open(filepath)
            dpi = img.info.get('dpi', (72, 72))
            return dpi[0] if isinstance(dpi, tuple) else dpi
        except:
            return 0
