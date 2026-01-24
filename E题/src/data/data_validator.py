"""
Data Validation Module

Validates input data quality for evaluation models:
- Outlier detection (IQR, Z-score methods)
- Missing value checks
- Range validation
- Data type validation
- Correlation analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    severity: str  # "INFO", "WARNING", "ERROR"
    message: str
    affected_indices: Optional[List[int]] = None
    recommended_action: Optional[str] = None


class DataValidator:
    """
    Validates data quality for multi-criteria evaluation models.
    
    Usage:
        >>> validator = DataValidator()
        >>> results = validator.validate_decision_matrix(X, criteria_names)
        >>> if not validator.has_fatal_errors(results):
        >>>     # Proceed with evaluation
    """
    
    def __init__(
        self,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 3.0,
        missing_value_threshold: float = 0.1,
    ):
        """
        Initialize validator.
        
        Parameters:
            outlier_method: 'iqr' or 'zscore'
            outlier_threshold: IQR multiplier or Z-score threshold
            missing_value_threshold: Max fraction of missing values allowed per indicator
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_value_threshold = missing_value_threshold
    
    def validate_decision_matrix(
        self,
        decision_matrix: np.ndarray,
        criteria_names: List[str],
        indicator_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[ValidationResult]:
        """
        Comprehensive validation of decision matrix.
        
        Parameters:
            decision_matrix: (m x n) matrix of indicator values
            criteria_names: List of n indicator names
            indicator_ranges: Optional dict mapping indicator name to (min, max) valid range
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        # Check 1: Missing values
        results.extend(self._check_missing_values(decision_matrix, criteria_names))
        
        # Check 2: Data types and finite values
        results.extend(self._check_data_types(decision_matrix, criteria_names))
        
        # Check 3: Outliers
        results.extend(self._check_outliers(decision_matrix, criteria_names))
        
        # Check 4: Range validation
        if indicator_ranges:
            results.extend(self._check_ranges(decision_matrix, criteria_names, indicator_ranges))
        
        # Check 5: Zero variance
        results.extend(self._check_variance(decision_matrix, criteria_names))
        
        # Check 6: High correlation (multicollinearity)
        results.extend(self._check_correlation(decision_matrix, criteria_names))
        
        return results
    
    def _check_missing_values(
        self,
        X: np.ndarray,
        criteria_names: List[str],
    ) -> List[ValidationResult]:
        """Check for missing (NaN/None/inf) values."""
        results = []
        m, n = X.shape
        
        for j, name in enumerate(criteria_names):
            col = X[:, j]
            n_missing = np.sum(np.isnan(col)) + np.sum(np.isinf(col))
            
            if n_missing > 0:
                missing_frac = n_missing / m
                
                if missing_frac > self.missing_value_threshold:
                    results.append(ValidationResult(
                        passed=False,
                        severity="ERROR",
                        message=f"Indicator '{name}' has {missing_frac:.1%} missing values (threshold: {self.missing_value_threshold:.1%})",
                        affected_indices=np.where(np.isnan(col) | np.isinf(col))[0].tolist(),
                        recommended_action="Remove indicator or impute missing values",
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        severity="WARNING",
                        message=f"Indicator '{name}' has {n_missing} missing values ({missing_frac:.1%})",
                        affected_indices=np.where(np.isnan(col) | np.isinf(col))[0].tolist(),
                        recommended_action="Consider imputation (mean/median/interpolation)",
                    ))
        
        return results
    
    def _check_data_types(
        self,
        X: np.ndarray,
        criteria_names: List[str],
    ) -> List[ValidationResult]:
        """Check data types and finite values."""
        results = []
        
        if not np.issubdtype(X.dtype, np.number):
            results.append(ValidationResult(
                passed=False,
                severity="ERROR",
                message=f"Decision matrix contains non-numeric data (dtype: {X.dtype})",
                recommended_action="Convert to numeric or remove non-numeric columns",
            ))
        
        return results
    
    def _check_outliers(
        self,
        X: np.ndarray,
        criteria_names: List[str],
    ) -> List[ValidationResult]:
        """Detect outliers using IQR or Z-score method."""
        results = []
        m, n = X.shape
        
        for j, name in enumerate(criteria_names):
            col = X[:, j]
            col_clean = col[~np.isnan(col)]  # Remove NaN for outlier detection
            
            if len(col_clean) == 0:
                continue
            
            if self.outlier_method == 'iqr':
                Q1 = np.percentile(col_clean, 25)
                Q3 = np.percentile(col_clean, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                outliers = (col < lower_bound) | (col > upper_bound)
            
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(col_clean, nan_policy='omit'))
                # Map back to original indices
                outliers = np.zeros(m, dtype=bool)
                outliers[~np.isnan(col)] = z_scores > self.outlier_threshold
            
            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")
            
            n_outliers = np.sum(outliers)
            
            if n_outliers > 0:
                outlier_frac = n_outliers / m
                severity = "ERROR" if outlier_frac > 0.2 else "WARNING"
                
                results.append(ValidationResult(
                    passed=(severity != "ERROR"),
                    severity=severity,
                    message=f"Indicator '{name}' has {n_outliers} outliers ({outlier_frac:.1%}) using {self.outlier_method} method",
                    affected_indices=np.where(outliers)[0].tolist(),
                    recommended_action="Investigate outliers: data error or genuine extreme value?",
                ))
        
        return results
    
    def _check_ranges(
        self,
        X: np.ndarray,
        criteria_names: List[str],
        indicator_ranges: Dict[str, Tuple[float, float]],
    ) -> List[ValidationResult]:
        """Check if values are within expected ranges."""
        results = []
        
        for j, name in enumerate(criteria_names):
            if name not in indicator_ranges:
                continue
            
            col = X[:, j]
            min_val, max_val = indicator_ranges[name]
            
            out_of_range = (col < min_val) | (col > max_val)
            n_out = np.sum(out_of_range)
            
            if n_out > 0:
                results.append(ValidationResult(
                    passed=False,
                    severity="ERROR",
                    message=f"Indicator '{name}' has {n_out} values outside valid range [{min_val}, {max_val}]",
                    affected_indices=np.where(out_of_range)[0].tolist(),
                    recommended_action=f"Expected range: [{min_val}, {max_val}]. Check data source.",
                ))
        
        return results
    
    def _check_variance(
        self,
        X: np.ndarray,
        criteria_names: List[str],
    ) -> List[ValidationResult]:
        """Check for zero or near-zero variance indicators."""
        results = []
        
        for j, name in enumerate(criteria_names):
            col = X[:, j]
            col_clean = col[~np.isnan(col)]
            
            if len(col_clean) == 0:
                continue
            
            variance = np.var(col_clean)
            
            if variance < 1e-6:
                results.append(ValidationResult(
                    passed=False,
                    severity="WARNING",
                    message=f"Indicator '{name}' has near-zero variance (σ² = {variance:.2e})",
                    recommended_action="Remove indicator: cannot distinguish between alternatives",
                ))
        
        return results
    
    def _check_correlation(
        self,
        X: np.ndarray,
        criteria_names: List[str],
        high_corr_threshold: float = 0.95,
    ) -> List[ValidationResult]:
        """Check for high correlation (multicollinearity)."""
        results = []
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        
        # Find high correlations
        high_corr_pairs = np.where(np.abs(corr_matrix) > high_corr_threshold)
        
        reported_pairs = set()
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i < j:  # Avoid duplicates
                pair_key = (i, j)
                if pair_key not in reported_pairs:
                    reported_pairs.add(pair_key)
                    
                    results.append(ValidationResult(
                        passed=True,  # Not a hard error, but should be noted
                        severity="WARNING",
                        message=f"High correlation between '{criteria_names[i]}' and '{criteria_names[j]}' (r = {corr_matrix[i, j]:.3f})",
                        recommended_action="Consider removing one or combining into composite indicator",
                    ))
        
        return results
    
    def has_fatal_errors(self, results: List[ValidationResult]) -> bool:
        """Check if any validation result is a fatal ERROR."""
        return any(r.severity == "ERROR" and not r.passed for r in results)
    
    def print_report(self, results: List[ValidationResult]) -> None:
        """Print formatted validation report."""
        print("=" * 80)
        print("DATA VALIDATION REPORT")
        print("=" * 80)
        
        errors = [r for r in results if r.severity == "ERROR"]
        warnings = [r for r in results if r.severity == "WARNING"]
        infos = [r for r in results if r.severity == "INFO"]
        
        print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info")
        
        if errors:
            print("\n❌ ERRORS:")
            for r in errors:
                print(f"  • {r.message}")
                if r.recommended_action:
                    print(f"    → {r.recommended_action}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for r in warnings:
                print(f"  • {r.message}")
                if r.recommended_action:
                    print(f"    → {r.recommended_action}")
        
        if not errors and not warnings:
            print("\n✓ All validation checks passed")
        
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    print("Testing Data Validator...\n")
    
    # Create test data with issues
    X_test = np.array([
        [15.2, 2.1, 8.2, 5.3],
        [28.4, 12.5, 22.6, 18.2],
        [65.8, 45.8, 58.4, np.nan],  # Missing value
        [999.0, 88.3, 91.7, 85.6],   # Outlier
    ])
    
    criteria = ['SkyBrightness', 'OverIllumination', 'EcoDisruption', 'CircadianImpact']
    
    # Validate
    validator = DataValidator(outlier_method='iqr', outlier_threshold=1.5)
    results = validator.validate_decision_matrix(X_test, criteria)
    
    # Print report
    validator.print_report(results)
    
    # Check for fatal errors
    if validator.has_fatal_errors(results):
        print("\n⚠️  Fatal errors detected - cannot proceed with evaluation")
    else:
        print("\n✓ No fatal errors - safe to proceed")
