"""
Coach Effect Analysis Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class TTestResult:
    """Result of t-test"""
    t_statistic: float
    p_value: float
    mean_before: float
    mean_after: float
    is_significant: bool


@dataclass
class EffectSizeResult:
    """Result of effect size calculation"""
    cohens_d: float
    interpretation: str  # 'small', 'medium', 'large'


@dataclass
class CoachEffectCase:
    """Case study result for a country"""
    country: str
    sport: str
    coach_arrival_year: int
    t_test: TTestResult
    effect_size: EffectSizeResult
    narrative: str


class CoachEffectAnalyzer:
    """
    Analyzes the impact of elite coaches on medal performance.
    
    Case studies:
    - Kenya (long-distance running)
    - Jamaica (sprinting)
    - Singapore (table tennis)
    """
    
    # Predefined case study countries
    CASE_STUDIES = {
        'KEN': {'sport': 'Athletics', 'coach_year': 2008},
        'JAM': {'sport': 'Athletics', 'coach_year': 2004},
        'SGP': {'sport': 'Table Tennis', 'coach_year': 2006}
    }
    
    def __init__(self):
        self.case_studies = self.CASE_STUDIES
    
    def perform_t_test(self, before: np.ndarray,
                       after: np.ndarray,
                       significance: float = 0.05) -> TTestResult:
        """
        Perform independent samples t-test.
        
        Args:
            before: Medal counts before coach arrival
            after: Medal counts after coach arrival
            significance: Significance level
        """
        if len(before) < 2 or len(after) < 2:
            return TTestResult(0, 1, 0, 0, False)
        
        t_stat, p_value = stats.ttest_ind(after, before, alternative='greater')
        
        return TTestResult(
            t_statistic=t_stat,
            p_value=p_value,
            mean_before=np.mean(before),
            mean_after=np.mean(after),
            is_significant=p_value < significance
        )

    def compute_cohens_d(self, before: np.ndarray,
                         after: np.ndarray) -> EffectSizeResult:
        """
        Compute Cohen's d effect size.
        
        Cohen's d = (mean_after - mean_before) / pooled_std
        """
        if len(before) < 2 or len(after) < 2:
            return EffectSizeResult(0, 'none')
        
        mean_diff = np.mean(after) - np.mean(before)
        
        # Pooled standard deviation
        n1, n2 = len(before), len(after)
        var1, var2 = np.var(before, ddof=1), np.var(after, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return EffectSizeResult(0, 'none')
        
        d = mean_diff / pooled_std
        
        # Interpret effect size
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return EffectSizeResult(cohens_d=d, interpretation=interpretation)
    
    def analyze_country(self, df: pd.DataFrame,
                        country: str,
                        coach_year: int,
                        medal_col: str = 'total',
                        year_col: str = 'year',
                        country_col: str = 'country') -> Tuple[TTestResult, EffectSizeResult]:
        """
        Analyze coach effect for a specific country.
        """
        country_data = df[df[country_col] == country].sort_values(year_col)
        
        before = country_data[country_data[year_col] < coach_year][medal_col].values
        after = country_data[country_data[year_col] >= coach_year][medal_col].values
        
        t_test = self.perform_t_test(before, after)
        effect_size = self.compute_cohens_d(before, after)
        
        return t_test, effect_size
    
    def generate_case_narrative(self, country: str,
                                t_test: TTestResult,
                                effect_size: EffectSizeResult) -> str:
        """
        Generate narrative for case study.
        """
        if t_test.is_significant:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        narrative = (
            f"Analysis of {country}: "
            f"Mean medals increased from {t_test.mean_before:.1f} to {t_test.mean_after:.1f}. "
            f"The difference is {significance} (p={t_test.p_value:.3f}). "
            f"Effect size (Cohen's d) = {effect_size.cohens_d:.2f} ({effect_size.interpretation})."
        )
        
        return narrative
    
    def run_all_case_studies(self, df: pd.DataFrame) -> List[CoachEffectCase]:
        """
        Run analysis for all predefined case study countries.
        """
        results = []
        
        for country, info in self.case_studies.items():
            t_test, effect_size = self.analyze_country(
                df, country, info['coach_year']
            )
            narrative = self.generate_case_narrative(country, t_test, effect_size)
            
            results.append(CoachEffectCase(
                country=country,
                sport=info['sport'],
                coach_arrival_year=info['coach_year'],
                t_test=t_test,
                effect_size=effect_size,
                narrative=narrative
            ))
        
        return results
