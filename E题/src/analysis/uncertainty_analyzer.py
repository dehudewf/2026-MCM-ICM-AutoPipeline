"""
Uncertainty Analysis Module

Quantifies uncertainty in evaluation results using:
- Bootstrap confidence intervals for TOPSIS scores
- Monte Carlo simulation for ranking stability
- Bayesian credible intervals for AHP weights (future)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models import TOPSISModel


@dataclass
class UncertaintyResult:
    """Container for uncertainty analysis results."""
    alternative_names: List[str]
    point_estimates: np.ndarray  # TOPSIS scores
    confidence_intervals: np.ndarray  # shape (n_alternatives, 2) for (lower, upper)
    confidence_level: float  # e.g., 0.95
    n_bootstrap: int
    ranking_stability: float  # Fraction of bootstrap samples with same top-1
    method: str  # 'bootstrap' or 'monte_carlo'


class UncertaintyAnalyzer:
    """
    Quantify uncertainty in multi-criteria evaluation results.
    
    Usage:
        >>> analyzer = UncertaintyAnalyzer()
        >>> result = analyzer.bootstrap_topsis_scores(
        >>>     decision_matrix=X,
        >>>     weights=w,
        >>>     indicator_types=types,
        >>>     n_bootstrap=1000,
        >>> )
        >>> print(f"95% CI for location 1: [{result.confidence_intervals[0, 0]:.3f}, {result.confidence_intervals[0, 1]:.3f}]")
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize analyzer.
        
        Parameters:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.topsis = TOPSISModel()
    
    def bootstrap_topsis_scores(
        self,
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        indicator_types: List[bool],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        alternative_names: Optional[List[str]] = None,
        method: str = 'case_resampling',
    ) -> UncertaintyResult:
        """
        Bootstrap confidence intervals for TOPSIS scores.
        
        Parameters:
            decision_matrix: (m x n) evaluation matrix
            weights: (n,) indicator weights
            indicator_types: List of n booleans (True=benefit, False=cost)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            alternative_names: Names of alternatives
            method: 'case_resampling' (resample alternatives) or 'weight_perturbation'
        
        Returns:
            UncertaintyResult with confidence intervals
        """
        m, n = decision_matrix.shape
        
        if alternative_names is None:
            alternative_names = [f"A{i+1}" for i in range(m)]
        
        # Storage for bootstrap samples
        bootstrap_scores = np.zeros((n_bootstrap, m))
        bootstrap_rankings = np.zeros((n_bootstrap, m), dtype=int)
        
        for b in range(n_bootstrap):
            if method == 'case_resampling':
                # Resample alternatives (rows) with replacement
                indices = np.random.choice(m, size=m, replace=True)
                X_boot = decision_matrix[indices, :]
            
            elif method == 'weight_perturbation':
                # Perturb weights within reasonable bounds
                # Add Gaussian noise: N(0, 0.05 * weight)
                noise = np.random.normal(0, 0.05, size=n)
                w_boot = weights * (1 + noise)
                w_boot = np.maximum(w_boot, 0.01)  # Ensure positive
                w_boot = w_boot / w_boot.sum()  # Renormalize
                X_boot = decision_matrix
                weights_for_topsis = w_boot
            
            else:
                X_boot = decision_matrix
                weights_for_topsis = weights
            
            if method != 'weight_perturbation':
                weights_for_topsis = weights
            
            # Evaluate with TOPSIS
            result = self.topsis.evaluate(
                decision_matrix=X_boot,
                weights=weights_for_topsis,
                indicator_types=indicator_types,
            )
            
            if method == 'case_resampling':
                # Map bootstrap scores back to original indices
                scores_mapped = np.zeros(m)
                for i, orig_idx in enumerate(indices):
                    scores_mapped[orig_idx] = max(scores_mapped[orig_idx], result.scores[i])
                bootstrap_scores[b, :] = scores_mapped
            else:
                bootstrap_scores[b, :] = result.scores
            
            bootstrap_rankings[b, :] = result.ranking
        
        # Compute point estimates (mean of bootstrap samples)
        point_estimates = np.mean(bootstrap_scores, axis=0)
        
        # Compute confidence intervals (percentile method)
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = np.zeros((m, 2))
        for i in range(m):
            confidence_intervals[i, 0] = np.percentile(bootstrap_scores[:, i], lower_percentile)
            confidence_intervals[i, 1] = np.percentile(bootstrap_scores[:, i], upper_percentile)
        
        # Ranking stability: how often is top-1 the same?
        base_top1 = np.argmin(bootstrap_rankings, axis=1)  # Rank 1 = index with min rank
        most_common_top1 = np.bincount(base_top1).argmax()
        ranking_stability = np.mean(base_top1 == most_common_top1)
        
        return UncertaintyResult(
            alternative_names=alternative_names,
            point_estimates=point_estimates,
            confidence_intervals=confidence_intervals,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            ranking_stability=ranking_stability,
            method=method,
        )
    
    def monte_carlo_ranking_stability(
        self,
        decision_matrix: np.ndarray,
        weights: np.ndarray,
        indicator_types: List[bool],
        n_simulations: int = 1000,
        noise_level: float = 0.1,
    ) -> Tuple[float, np.ndarray]:
        """
        Monte Carlo simulation to test ranking stability under data noise.
        
        Parameters:
            decision_matrix: (m x n) evaluation matrix
            weights: (n,) indicator weights
            indicator_types: List of n booleans
            n_simulations: Number of Monte Carlo runs
            noise_level: Standard deviation of multiplicative noise (default: 10%)
        
        Returns:
            (stability_rate, ranking_frequency_matrix)
            - stability_rate: Fraction of simulations with same top-1 as base case
            - ranking_frequency_matrix: (m x m) matrix where [i, j] = freq of alt i at rank j
        """
        m, n = decision_matrix.shape
        
        # Base case
        base_result = self.topsis.evaluate(decision_matrix, weights, indicator_types)
        base_ranking = base_result.ranking
        base_top1 = np.argmin(base_ranking)
        
        # Storage
        ranking_frequency = np.zeros((m, m), dtype=int)
        stable_count = 0
        
        for _ in range(n_simulations):
            # Add multiplicative noise to decision matrix
            noise = 1 + np.random.normal(0, noise_level, size=(m, n))
            X_noisy = decision_matrix * noise
            
            # Evaluate
            result = self.topsis.evaluate(X_noisy, weights, indicator_types)
            
            # Check stability
            sim_top1 = np.argmin(result.ranking)
            if sim_top1 == base_top1:
                stable_count += 1
            
            # Record ranking frequencies
            for alt_idx in range(m):
                rank = result.ranking[alt_idx] - 1  # Convert to 0-indexed
                ranking_frequency[alt_idx, rank] += 1
        
        stability_rate = stable_count / n_simulations
        
        return stability_rate, ranking_frequency
    
    def to_dataframe(self, result: UncertaintyResult) -> pd.DataFrame:
        """Convert UncertaintyResult to DataFrame for display."""
        df = pd.DataFrame({
            'Alternative': result.alternative_names,
            'Score': result.point_estimates,
            'CI_Lower': result.confidence_intervals[:, 0],
            'CI_Upper': result.confidence_intervals[:, 1],
            'CI_Width': result.confidence_intervals[:, 1] - result.confidence_intervals[:, 0],
        })
        
        df = df.sort_values('Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df[['Rank', 'Alternative', 'Score', 'CI_Lower', 'CI_Upper', 'CI_Width']]


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Uncertainty Analyzer - Bootstrap Confidence Intervals")
    print("=" * 80)
    
    # Example data
    X = np.array([
        [15.2, 2.1, 8.2, 5.3],
        [28.4, 12.5, 22.6, 18.2],
        [65.8, 45.8, 58.4, 52.1],
        [142.5, 88.3, 91.7, 85.6],
    ])
    
    weights = np.array([0.3, 0.3, 0.2, 0.2])
    indicator_types = [False, False, False, False]  # All cost indicators
    alternative_names = ['Protected', 'Rural', 'Suburban', 'Urban']
    
    # Run bootstrap
    analyzer = UncertaintyAnalyzer(random_seed=42)
    
    print("\n[1] Bootstrap Confidence Intervals (case resampling)...")
    result = analyzer.bootstrap_topsis_scores(
        decision_matrix=X,
        weights=weights,
        indicator_types=indicator_types,
        n_bootstrap=1000,
        confidence_level=0.95,
        alternative_names=alternative_names,
        method='case_resampling',
    )
    
    df = analyzer.to_dataframe(result)
    print(df.to_string(index=False))
    print(f"\nRanking stability (top-1): {result.ranking_stability:.1%}")
    
    print("\n[2] Monte Carlo Ranking Stability (data noise)...")
    stability, freq_matrix = analyzer.monte_carlo_ranking_stability(
        decision_matrix=X,
        weights=weights,
        indicator_types=indicator_types,
        n_simulations=1000,
        noise_level=0.10,
    )
    
    print(f"Top-1 stability under 10% data noise: {stability:.1%}")
    
    print("\n✓ Uncertainty analysis complete")
