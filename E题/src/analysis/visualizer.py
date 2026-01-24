"""
Visualization system for E-Problem evaluation results

Generates O-Award quality figures:
1. Weight comparison chart (AHP vs EWM vs Combined)
2. TOPSIS ranking bar chart with scores
3. Sensitivity heatmap (weight perturbation impact)

Aligned with E题可视化知识库.csv and config.output settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import config
from .evaluation_pipeline import EvaluationOutputs

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = config.output.font_family
plt.rcParams['font.size'] = config.output.font_size
plt.rcParams['figure.dpi'] = config.output.figure_dpi


class EvaluationVisualizer:
    """
    Visualization system for E-problem evaluation results.
    
    Usage:
        >>> viz = EvaluationVisualizer(output_dir='E题/output')
        >>> viz.plot_weight_comparison(outputs)
        >>> viz.plot_topsis_ranking(outputs)
        >>> viz.plot_sensitivity_heatmap(sensitivity_df)
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize visualizer.
        
        Parameters:
            output_dir: Directory to save figures (default: config.OUTPUT_DIR)
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes (academic + colorblind-friendly)
        self.colors = {
            'ahp': '#1f77b4',      # Blue
            'ewm': '#ff7f0e',      # Orange
            'combined': '#2ca02c', # Green
            'benefit': '#2ca02c',  # Green (benefit indicators)
            'cost': '#d62728',     # Red (cost indicators)
        }
    
    def plot_weight_comparison(
        self,
        outputs: EvaluationOutputs,
        save: bool = True,
        filename: str = 'weight_comparison.png',
    ) -> plt.Figure:
        """Plot AHP vs EWM vs Combined weights comparison.
        
        Parameters:
            outputs: Evaluation outputs containing weight results
            save: Whether to save figure to disk
            filename: Output filename
        
        Returns:
            matplotlib Figure object
        """
        criteria_names = outputs.ewm_result.criteria_names
        
        # Prepare data
        if outputs.ahp_result is not None:
            ahp_weights = outputs.ahp_result.weights
        else:
            ahp_weights = np.zeros(len(criteria_names))
        
        ewm_weights = outputs.ewm_result.weights
        combined_weights = outputs.combined_weights
        
        # Create figure
        fig, ax = plt.subplots(figsize=config.output.figure_size)
        
        x = np.arange(len(criteria_names))
        width = 0.25
        
        # Plot bars
        bars1 = ax.bar(x - width, ahp_weights, width, 
                      label='AHP (Subjective)', color=self.colors['ahp'], alpha=0.8)
        bars2 = ax.bar(x, ewm_weights, width,
                      label='EWM (Objective)', color=self.colors['ewm'], alpha=0.8)
        bars3 = ax.bar(x + width, combined_weights, width,
                      label='Combined (α=0.5)', color=self.colors['combined'], alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Indicators', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
        ax.set_title('Weight Comparison: AHP vs EWM vs Combined', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(criteria_names, rotation=45, ha='right')
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(max(ahp_weights), max(ewm_weights), max(combined_weights)) * 1.1)
        
        # Add value labels on top of bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Only label if significant
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=config.output.figure_dpi, 
                       bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_topsis_ranking(
        self,
        outputs: EvaluationOutputs,
        save: bool = True,
        filename: str = 'topsis_ranking.png',
    ) -> plt.Figure:
        """Plot TOPSIS ranking results with scores and distances.
        
        Parameters:
            outputs: Evaluation outputs
            save: Whether to save
            filename: Output filename
        
        Returns:
            matplotlib Figure
        """
        df = outputs.topsis_result.to_dataframe()
        df = df.sort_values('Rank')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Left panel: TOPSIS scores ---
        alternatives = df['Alternative'].values
        scores = df['Score'].values
        colors = plt.cm.RdYlGn(scores)  # Red to green colormap
        
        bars = ax1.barh(alternatives, scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('TOPSIS Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Location', fontsize=12, fontweight='bold')
        ax1.set_title('TOPSIS Ranking Results', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (alt, score) in enumerate(zip(alternatives, scores)):
            ax1.text(score + 0.02, i, f'{score:.4f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        # Add rank badges
        for i, rank in enumerate(df['Rank'].values):
            ax1.text(-0.05, i, f'#{int(rank)}', 
                    va='center', ha='right', fontsize=12, 
                    fontweight='bold', color='darkblue')
        
        # --- Right panel: D+ and D- distances ---
        d_pos = df['D+'].values
        d_neg = df['D-'].values
        
        x = np.arange(len(alternatives))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, d_pos, width, label='D⁺ (to positive ideal)',
                       color='#d62728', alpha=0.7)
        bars2 = ax2.bar(x + width/2, d_neg, width, label='D⁻ (to negative ideal)',
                       color='#2ca02c', alpha=0.7)
        
        ax2.set_xlabel('Location', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Distance', fontsize=12, fontweight='bold')
        ax2.set_title('Distance to Ideal Solutions', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(alternatives, rotation=45, ha='right')
        ax2.legend(loc='upper right', frameon=True, shadow=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=config.output.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_sensitivity_heatmap(
        self,
        sensitivity_df: pd.DataFrame,
        criteria_names: List[str],
        save: bool = True,
        filename: str = 'sensitivity_heatmap.png',
    ) -> plt.Figure:
        """Plot sensitivity analysis heatmap.
        
        Parameters:
            sensitivity_df: Output from run_weight_sensitivity()
            criteria_names: List of indicator names
            save: Whether to save
            filename: Output filename
        
        Returns:
            matplotlib Figure
        """
        # Prepare data matrix for heatmap
        n_criteria = len(criteria_names)
        sensitivity_matrix = np.zeros((n_criteria, 2))
        
        for _, row in sensitivity_df.iterrows():
            idx = int(row['Criterion_Index'])
            col = 0 if row['Direction'] == 'increase' else 1
            sensitivity_matrix[idx, col] = row['Max_Score_Change']
        
        fig, ax = plt.subplots(figsize=(8, max(6, n_criteria * 0.5)))
        
        # Create heatmap
        im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        
        # Set ticks and labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Weight +20%', 'Weight -20%'])
        ax.set_yticks(np.arange(n_criteria))
        ax.set_yticklabels(criteria_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Max Score Change', rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations
        for i in range(n_criteria):
            for j in range(2):
                value = sensitivity_matrix[i, j]
                color = 'white' if value > sensitivity_matrix.max() * 0.5 else 'black'
                text = ax.text(j, i, f'{value:.4f}',
                             ha='center', va='center', color=color, fontsize=9)
        
        ax.set_title('Weight Sensitivity Analysis (±20% Perturbation)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Perturbation Direction', fontsize=12, fontweight='bold')
        ax.set_ylabel('Indicators', fontsize=12, fontweight='bold')
        
        # Add stability threshold line
        threshold = config.sensitivity.high_sensitivity_threshold
        ax.text(1.02, 0.5, f'High sensitivity\nthreshold:\n{threshold:.2f}',
               transform=ax.transAxes, va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=config.output.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    def plot_indicator_radar(
        self,
        outputs: EvaluationOutputs,
        decision_matrix: np.ndarray,
        save: bool = True,
        filename: str = 'indicator_radar.png',
    ) -> plt.Figure:
        """Plot radar chart comparing locations across indicators.
        
        Parameters:
            outputs: Evaluation outputs
            decision_matrix: Original decision matrix
            save: Whether to save
            filename: Output filename
        
        Returns:
            matplotlib Figure
        """
        criteria_names = outputs.ewm_result.criteria_names
        alternative_names = outputs.topsis_result.alternative_names
        
        # Normalize data to [0, 1] for visualization
        X_norm = decision_matrix.copy()
        for j in range(X_norm.shape[1]):
            x_min, x_max = X_norm[:, j].min(), X_norm[:, j].max()
            if x_max > x_min:
                X_norm[:, j] = (X_norm[:, j] - x_min) / (x_max - x_min)
        
        # Setup radar chart
        num_vars = len(criteria_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each location
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (name, color) in enumerate(zip(alternative_names, colors_list)):
            values = X_norm[i, :].tolist()
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria_names, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('Multi-Indicator Comparison (Normalized)',
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / filename
            fig.savefig(output_path, dpi=config.output.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    def generate_all_figures(
        self,
        outputs: EvaluationOutputs,
        sensitivity_df: pd.DataFrame,
        decision_matrix: np.ndarray,
    ) -> Dict[str, plt.Figure]:
        """Generate all visualization figures at once.
        
        Parameters:
            outputs: Evaluation outputs
            sensitivity_df: Sensitivity analysis results
            decision_matrix: Original decision matrix
        
        Returns:
            Dictionary of figure names to Figure objects
        """
        figures = {}
        
        print("\n" + "=" * 60)
        print("Generating O-Award Quality Figures")
        print("=" * 60)
        
        figures['weight_comparison'] = self.plot_weight_comparison(outputs)
        figures['topsis_ranking'] = self.plot_topsis_ranking(outputs)
        figures['sensitivity_heatmap'] = self.plot_sensitivity_heatmap(
            sensitivity_df, outputs.ewm_result.criteria_names
        )
        figures['indicator_radar'] = self.plot_indicator_radar(
            outputs, decision_matrix
        )
        
        print("=" * 60)
        print(f"✓ All figures saved to: {self.output_dir}")
        print("=" * 60)
        
        return figures


__all__ = [
    'EvaluationVisualizer',
]
