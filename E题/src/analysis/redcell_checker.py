"""
Red Cell Attack System for E-Problem Evaluation Models

This module implements @redcell systematic attack framework:
1. Indicator system completeness check
2. AHP judgment matrix rationality check  
3. Extreme perturbation robustness test
4. Ranking stability verification

Aligned with E题-modeling-prompts-final.md 提示词9 (Red Cell终极攻击)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..models import AHPModel
from .evaluation_pipeline import EvaluationInputs, EvaluationPipeline, EvaluationOutputs


@dataclass
class AttackResult:
    """Container for red cell attack findings."""
    severity: str  # "FATAL", "CRITICAL", "MAJOR", "MINOR"
    dimension: str  # Attack dimension
    issue: str  # Issue description
    evidence: str  # Evidence/quantification
    impact: str  # Potential impact
    recommendation: str  # Fix suggestion
    
    def to_dict(self) -> Dict:
        return {
            'Severity': self.severity,
            'Dimension': self.dimension,
            'Issue': self.issue,
            'Evidence': self.evidence,
            'Impact': self.impact,
            'Recommendation': self.recommendation,
        }


class RedCellChecker:
    """
    Red Cell attack system for E-problem models.
    
    Usage:
        >>> checker = RedCellChecker()
        >>> inputs = EvaluationInputs(...)
        >>> outputs = pipeline.run(inputs)
        >>> attack_report = checker.attack(inputs, outputs)
        >>> df_issues = attack_report.to_dataframe()
    """
    
    def __init__(self):
        self.attacks: List[AttackResult] = []
        self.ahp_model = AHPModel()
    
    def attack(
        self, 
        inputs: EvaluationInputs,
        outputs: EvaluationOutputs,
        run_extreme_tests: bool = True,
    ) -> 'AttackReport':
        """Run all red cell attacks.
        
        Parameters:
            inputs: Original evaluation inputs
            outputs: Evaluation outputs to attack
            run_extreme_tests: Whether to run expensive extreme perturbation tests
        
        Returns:
            AttackReport with all findings
        """
        self.attacks = []
        
        # Attack 1: Indicator system completeness
        self._attack_indicator_system(inputs)
        
        # Attack 2: AHP judgment matrix rationality
        if outputs.ahp_result is not None:
            self._attack_ahp_matrix(inputs, outputs.ahp_result)
        
        # Attack 3: Weight reasonableness
        self._attack_weights(outputs)
        
        # Attack 4: TOPSIS ranking logic
        self._attack_ranking(inputs, outputs)
        
        # Attack 5: Extreme perturbation robustness (optional - expensive)
        if run_extreme_tests:
            self._attack_extreme_perturbations(inputs, outputs)
        
        return AttackReport(attacks=self.attacks)
    
    # ========================================================================
    # Attack dimension 1: Indicator system
    # ========================================================================
    def _attack_indicator_system(self, inputs: EvaluationInputs):
        """Check indicator completeness, independence, measurability."""
        n_indicators = inputs.decision_matrix.shape[1]
        
        # Check 1: Minimum indicator count
        if n_indicators < 5:
            self.attacks.append(AttackResult(
                severity="CRITICAL",
                dimension="Indicator System",
                issue="Insufficient number of indicators",
                evidence=f"Only {n_indicators} indicators (recommend ≥6 for multi-dimensional evaluation)",
                impact="May miss critical dimensions of light pollution risk",
                recommendation="Add indicators for missing dimensions (e.g., wildlife impact, public health metrics)",
            ))
        
        # Check 2: Indicator correlation (independence check)
        X = inputs.decision_matrix
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.85)
        
        if len(high_corr_pairs[0]) > 0:
            pairs = []
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i < j:  # Avoid duplicates
                    pairs.append((inputs.criteria_names[i], inputs.criteria_names[j], corr_matrix[i, j]))
            
            if pairs:
                evidence = "; ".join([f"{a} ↔ {b} (r={c:.3f})" for a, b, c in pairs[:3]])
                self.attacks.append(AttackResult(
                    severity="MAJOR",
                    dimension="Indicator System",
                    issue="High correlation between indicators (lack of independence)",
                    evidence=evidence,
                    impact="Redundant information → biased weights → distorted ranking",
                    recommendation="Consider removing or combining highly correlated indicators",
                ))
        
        # Check 3: Data quality (missing values, outliers)
        if np.any(np.isnan(X)):
            self.attacks.append(AttackResult(
                severity="FATAL",
                dimension="Indicator System",
                issue="Missing values in decision matrix",
                evidence=f"{np.sum(np.isnan(X))} NaN values detected",
                impact="Cannot proceed with evaluation",
                recommendation="Impute missing values or remove incomplete indicators",
            ))
        
        # Check 4: Zero variance indicators
        for j, name in enumerate(inputs.criteria_names):
            if np.std(X[:, j]) < 1e-6:
                self.attacks.append(AttackResult(
                    severity="CRITICAL",
                    dimension="Indicator System",
                    issue=f"Indicator '{name}' has zero variance",
                    evidence=f"std={np.std(X[:, j]):.6f}",
                    impact="Cannot distinguish between alternatives → zero weight in EWM",
                    recommendation=f"Remove '{name}' or provide more diverse data",
                ))
    
    # ========================================================================
    # Attack dimension 2: AHP judgment matrix
    # ========================================================================
    def _attack_ahp_matrix(self, inputs: EvaluationInputs, ahp_result):
        """Attack AHP judgment matrix rationality."""
        J = inputs.ahp_judgment_matrix
        
        # Check 1: Consistency ratio (CR)
        if ahp_result.cr >= 0.1:
            self.attacks.append(AttackResult(
                severity="CRITICAL",
                dimension="AHP Matrix",
                issue="Consistency ratio exceeds threshold",
                evidence=f"CR = {ahp_result.cr:.4f} ≥ 0.10",
                impact="Judgment matrix is inconsistent → unreliable weights",
                recommendation="Revise pairwise comparisons or use auto-repair function",
            ))
        elif ahp_result.cr >= 0.08:
            self.attacks.append(AttackResult(
                severity="MAJOR",
                dimension="AHP Matrix",
                issue="Consistency ratio approaching threshold",
                evidence=f"CR = {ahp_result.cr:.4f} (threshold=0.10)",
                impact="Marginal consistency → weights may be unstable",
                recommendation="Consider revising most inconsistent comparisons",
            ))
        
        # Check 2: Extreme judgments (>7 in Saaty scale)
        extreme_count = np.sum(np.abs(J) > 7)
        if extreme_count > 0:
            self.attacks.append(AttackResult(
                severity="MINOR",
                dimension="AHP Matrix",
                issue="Presence of extreme pairwise judgments",
                evidence=f"{extreme_count} judgments with scale >7 (out of {J.size})",
                impact="May indicate over-confident expert judgment",
                recommendation="Verify extreme judgments (scale 8-9) are justified",
            ))
        
        # Check 3: Reciprocity check
        n = J.shape[0]
        reciprocity_errors = []
        for i in range(n):
            for j in range(i+1, n):
                expected = 1.0 / J[i, j]
                actual = J[j, i]
                if not np.isclose(expected, actual, rtol=1e-3):
                    reciprocity_errors.append((i, j))
        
        if reciprocity_errors:
            self.attacks.append(AttackResult(
                severity="FATAL",
                dimension="AHP Matrix",
                issue="Reciprocity violations in judgment matrix",
                evidence=f"{len(reciprocity_errors)} pairs violate a_ij × a_ji = 1",
                impact="Invalid AHP matrix → cannot compute meaningful weights",
                recommendation="Fix reciprocity: ensure J[j,i] = 1/J[i,j] for all i,j",
            ))
    
    # ========================================================================
    # Attack dimension 3: Weights
    # ========================================================================
    def _attack_weights(self, outputs: EvaluationOutputs):
        """Attack weight reasonableness."""
        combined_weights = outputs.combined_weights
        
        # Check 1: Weight concentration (Gini coefficient)
        sorted_w = np.sort(combined_weights)
        n = len(sorted_w)
        index = np.arange(1, n+1)
        gini = (2 * np.sum(index * sorted_w)) / (n * np.sum(sorted_w)) - (n + 1) / n
        
        if gini > 0.6:
            self.attacks.append(AttackResult(
                severity="MAJOR",
                dimension="Weights",
                issue="High weight concentration (few indicators dominate)",
                evidence=f"Gini coefficient = {gini:.3f} (>0.6 indicates high inequality)",
                impact="Model may be overly sensitive to 1-2 indicators",
                recommendation="Consider more balanced weighting or verify indicator importance",
            ))
        
        # Check 2: Near-zero weights
        min_weight = np.min(combined_weights)
        if min_weight < 0.01:
            idx = np.argmin(combined_weights)
            name = outputs.ewm_result.criteria_names[idx]
            self.attacks.append(AttackResult(
                severity="MINOR",
                dimension="Weights",
                issue=f"Indicator '{name}' has negligible weight",
                evidence=f"w = {min_weight:.4f} (<1%)",
                impact="Indicator contributes almost nothing → consider removing",
                recommendation=f"Remove '{name}' or investigate why it's unimportant",
            ))
        
        # Check 3: AHP vs EWM divergence
        if outputs.ahp_result is not None:
            ahp_w = outputs.ahp_result.weights
            ewm_w = outputs.ewm_result.weights
            
            # Normalize both to compare
            ahp_w = ahp_w / ahp_w.sum()
            ewm_w = ewm_w / ewm_w.sum()
            
            # Compute L1 distance
            l1_dist = np.sum(np.abs(ahp_w - ewm_w))
            
            if l1_dist > 0.8:
                self.attacks.append(AttackResult(
                    severity="MAJOR",
                    dimension="Weights",
                    issue="Large divergence between AHP (subjective) and EWM (objective) weights",
                    evidence=f"L1 distance = {l1_dist:.3f} (>0.8 indicates strong disagreement)",
                    impact="Expert judgment and data patterns conflict → α parameter choice is critical",
                    recommendation="Investigate cause of divergence; consider adjusting α or revising AHP matrix",
                ))
    
    # ========================================================================
    # Attack dimension 4: Ranking
    # ========================================================================
    def _attack_ranking(self, inputs: EvaluationInputs, outputs: EvaluationOutputs):
        """Attack ranking logic and interpretability."""
        scores = outputs.topsis_result.scores
        ranking = outputs.topsis_result.ranking
        
        # Check 1: Score gaps
        sorted_scores = np.sort(scores)[::-1]
        score_gaps = np.diff(sorted_scores)
        min_gap = np.min(np.abs(score_gaps))
        
        if min_gap < 0.05:
            idx1, idx2 = np.argsort(scores)[::-1][:2]
            name1 = outputs.topsis_result.alternative_names[idx1]
            name2 = outputs.topsis_result.alternative_names[idx2]
            self.attacks.append(AttackResult(
                severity="MAJOR",
                dimension="Ranking",
                issue="Very close scores between top alternatives",
                evidence=f"{name1} vs {name2}: Δscore = {abs(scores[idx1] - scores[idx2]):.4f}",
                impact="Ranking is fragile → small data/weight changes may reverse order",
                recommendation="Report top-2 as statistically tied; conduct deeper sensitivity analysis",
            ))
        
        # Check 2: Counter-intuitive ranking
        # For light pollution: expect Protected < Rural < Suburban < Urban in risk
        # (i.e., Protected should have HIGHEST score = lowest risk)
        expected_order = [0, 1, 2, 3]  # Protected, Rural, Suburban, Urban
        actual_order = np.argsort(ranking)
        
        if not np.array_equal(actual_order, expected_order):
            self.attacks.append(AttackResult(
                severity="MINOR",
                dimension="Ranking",
                issue="Ranking deviates from intuitive expectation",
                evidence=f"Expected: Protected>Rural>Suburban>Urban; Got: {' > '.join(outputs.topsis_result.alternative_names[i] for i in np.argsort(-scores))}",
                impact="May indicate data quality issues or indicator direction errors",
                recommendation="Verify indicator types (benefit/cost) and data accuracy",
            ))
    
    # ========================================================================
    # Attack dimension 5: Extreme perturbations
    # ========================================================================
    def _attack_extreme_perturbations(self, inputs: EvaluationInputs, outputs: EvaluationOutputs):
        """Test robustness under extreme perturbations."""
        base_ranking = outputs.topsis_result.ranking.copy()
        pipeline = EvaluationPipeline()
        
        # Test 1: Extreme weight perturbations (±50%)
        perturb_levels = [0.5, 2.0]  # ×0.5 and ×2.0
        instability_count = 0
        
        for level in perturb_levels:
            for i in range(len(outputs.combined_weights)):
                perturbed_weights = outputs.combined_weights.copy()
                perturbed_weights[i] *= level
                perturbed_weights /= perturbed_weights.sum()
                
                # Re-evaluate with perturbed weights
                test_pipeline = EvaluationPipeline()
                test_outputs = test_pipeline.topsis.evaluate(
                    decision_matrix=inputs.decision_matrix,
                    weights=perturbed_weights,
                    indicator_types=inputs.indicator_is_benefit,
                    alternative_names=inputs.alternative_names,
                    criteria_names=inputs.criteria_names,
                )
                
                # Check if top-1 changed
                if test_outputs.ranking[0] != base_ranking[0]:
                    instability_count += 1
        
        total_tests = len(perturb_levels) * len(outputs.combined_weights)
        instability_rate = instability_count / total_tests
        
        if instability_rate > 0.3:
            self.attacks.append(AttackResult(
                severity="CRITICAL",
                dimension="Robustness",
                issue="High sensitivity to extreme weight perturbations",
                evidence=f"Top-1 ranking changed in {instability_rate:.1%} of extreme tests (±50% weight)",
                impact="Model lacks robustness → unreliable for real-world decision making",
                recommendation="Investigate weight stability; consider ensemble methods or uncertainty quantification",
            ))
        elif instability_rate > 0.1:
            self.attacks.append(AttackResult(
                severity="MAJOR",
                dimension="Robustness",
                issue="Moderate sensitivity to extreme perturbations",
                evidence=f"Top-1 changed in {instability_rate:.1%} of extreme tests",
                impact="Model is moderately fragile under large uncertainties",
                recommendation="Report confidence intervals for rankings",
            ))


@dataclass
class AttackReport:
    """Container for full red cell attack report."""
    attacks: List[AttackResult]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert attacks to DataFrame."""
        return pd.DataFrame([a.to_dict() for a in self.attacks])
    
    def get_by_severity(self, severity: str) -> List[AttackResult]:
        """Filter attacks by severity level."""
        return [a for a in self.attacks if a.severity == severity]
    
    def has_fatal_issues(self) -> bool:
        """Check if any fatal issues exist."""
        return any(a.severity == "FATAL" for a in self.attacks)
    
    def summary(self) -> str:
        """Generate text summary of attack report."""
        counts = {
            'FATAL': len(self.get_by_severity('FATAL')),
            'CRITICAL': len(self.get_by_severity('CRITICAL')),
            'MAJOR': len(self.get_by_severity('MAJOR')),
            'MINOR': len(self.get_by_severity('MINOR')),
        }
        
        lines = [
            "=" * 80,
            "RED CELL ATTACK REPORT SUMMARY",
            "=" * 80,
            f"Total issues found: {len(self.attacks)}",
            f"  🚨 FATAL:    {counts['FATAL']}",
            f"  ⚠️  CRITICAL: {counts['CRITICAL']}",
            f"  📝 MAJOR:    {counts['MAJOR']}",
            f"  💡 MINOR:    {counts['MINOR']}",
            "",
        ]
        
        if self.has_fatal_issues():
            lines.append("❌ FATAL ISSUES DETECTED - Model cannot proceed to deployment")
        elif counts['CRITICAL'] > 0:
            lines.append("⚠️  CRITICAL ISSUES - Requires immediate attention before O-Award submission")
        elif counts['MAJOR'] > 0:
            lines.append("📝 MAJOR ISSUES - Recommended fixes for stronger submission")
        else:
            lines.append("✅ No critical issues - Model is ready for checker approval")
        
        lines.append("=" * 80)
        return "\n".join(lines)


__all__ = [
    'RedCellChecker',
    'AttackResult',
    'AttackReport',
]
