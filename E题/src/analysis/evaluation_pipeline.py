"""
Evaluation pipeline for E-problem style multi-criteria decision tasks

This module wires together AHP (subjective weights), EWM (objective weights)
and TOPSIS (comprehensive evaluation) into a reusable pipeline.

It is designed to support 2023 ICM Problem E (Light Pollution) and other
E-type evaluation problems with different indicator systems and locations.

Agent alignment:
- Thinker (@strategist): designs indicator system + weight logic
- Executor (@executor): calls this pipeline with concrete data
- Checker (@redcell): attacks indicator set, weights, and ranking stability
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..models import AHPModel, AHPResult, EWMModel, EWMResult, TOPSISModel, TOPSISResult, combine_ahp_ewm
from ..config import config

# Reproducibility
SEED = config.random_seed
np.random.seed(SEED)


@dataclass
class EvaluationInputs:
    """Container for evaluation inputs.

    This stays generic: you can use it for light pollution (2023 E题) or
    any other E-problem style evaluation.
    """

    decision_matrix: np.ndarray  # shape (m_locations, n_indicators)
    criteria_names: List[str]
    indicator_types_ewm: List[str]
    # True = benefit (larger is better), False = cost (smaller is better)
    indicator_is_benefit: List[bool]
    alternative_names: Optional[List[str]] = None
    # Optional AHP judgement matrix for criteria layer (n×n)
    ahp_judgment_matrix: Optional[np.ndarray] = None
    # Weight combination parameter
    alpha: float = 0.5  # 0 = pure EWM, 1 = pure AHP


@dataclass
class EvaluationOutputs:
    """Container for evaluation outputs (for executor + checker)."""

    topsis_result: TOPSISResult
    combined_weights: np.ndarray
    ahp_result: Optional[AHPResult]
    ewm_result: EWMResult

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Return a compact table of scores, ranks, and weights."""
        df_scores = self.topsis_result.to_dataframe()
        df_weights = self.ewm_result.to_dataframe()[["Criterion", "Weight"]]
        df_weights = df_weights.rename(columns={"Weight": "EWM_Weight"})

        # Combine by criterion name if possible
        return df_scores, df_weights


class EvaluationPipeline:
    """High-level AHP + EWM + TOPSIS evaluation pipeline.

    Typical usage (executor phase):

    >>> pipeline = EvaluationPipeline()
    >>> inputs = EvaluationInputs(
    ...     decision_matrix=X,
    ...     criteria_names=[...],
    ...     indicator_types_ewm=["positive", "negative", ...],
    ...     indicator_is_benefit=[True, False, ...],
    ...     alternative_names=["Protected", "Rural", "Suburban", "Urban"],
    ...     ahp_judgment_matrix=J,  # or None to use EWM only
    ...     alpha=0.5,
    ... )
    >>> outputs = pipeline.run(inputs)
    >>> df_scores = outputs.topsis_result.to_dataframe()
    """

    def __init__(self) -> None:
        self.ahp = AHPModel(consistency_threshold=config.ahp.consistency_threshold)
        self.ewm = EWMModel()
        self.topsis = TOPSISModel()

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------
    def run(self, inputs: EvaluationInputs) -> EvaluationOutputs:
        X = np.asarray(inputs.decision_matrix, dtype=float)
        m, n = X.shape

        if len(inputs.criteria_names) != n:
            raise ValueError("Number of criteria names must match number of columns in decision_matrix")
        if len(inputs.indicator_types_ewm) != n:
            raise ValueError("indicator_types_ewm length must match number of criteria")
        if len(inputs.indicator_is_benefit) != n:
            raise ValueError("indicator_is_benefit length must match number of criteria")

        # ---------- Step 1: EWM objective weights ----------
        ewm_result = self.ewm.calculate_weights(
            data=X,
            criteria_names=inputs.criteria_names,
            indicator_types=inputs.indicator_types_ewm,
        )

        # ---------- Step 2: AHP subjective weights (optional) ----------
        ahp_result: Optional[AHPResult] = None
        if inputs.ahp_judgment_matrix is not None:
            ahp_result = self.ahp.calculate_weights(
                judgment_matrix=np.asarray(inputs.ahp_judgment_matrix, dtype=float),
                criteria_names=inputs.criteria_names,
            )
            base_weights = combine_ahp_ewm(
                ahp_weights=ahp_result.weights,
                ewm_weights=ewm_result.weights,
                alpha=inputs.alpha,
            )
        else:
            # If no AHP, fall back to pure EWM
            base_weights = ewm_result.weights.copy()

        # Ensure weights sum to 1
        combined_weights = base_weights / base_weights.sum()

        # ---------- Step 3: TOPSIS evaluation ----------
        alternative_names = inputs.alternative_names or [f"A{i+1}" for i in range(m)]

        topsis_result = self.topsis.evaluate(
            decision_matrix=X,
            weights=combined_weights,
            indicator_types=inputs.indicator_is_benefit,
            alternative_names=alternative_names,
            criteria_names=inputs.criteria_names,
            normalization_method=config.topsis.normalization_method,
            distance_method=config.topsis.distance_method,
        )

        return EvaluationOutputs(
            topsis_result=topsis_result,
            combined_weights=combined_weights,
            ahp_result=ahp_result,
            ewm_result=ewm_result,
        )


def run_weight_sensitivity(
    pipeline: EvaluationPipeline,
    inputs: EvaluationInputs,
    perturbation_range: Optional[float] = None,
) -> pd.DataFrame:
    """Run weight perturbation sensitivity analysis on top of the pipeline.

    This wraps TOPSISModel.sensitivity_analysis and uses the configured default
    range if none is provided. Intended for @executor to satisfy E题要求 of
    权重扰动分析 (typically ±20%).
    """
    X = np.asarray(inputs.decision_matrix, dtype=float)
    if perturbation_range is None:
        perturbation_range = config.sensitivity.weight_variation

    # Reuse the combined weights from a full run
    outputs = pipeline.run(inputs)
    base_weights = outputs.combined_weights
    indicator_types = inputs.indicator_is_benefit

    df = pipeline.topsis.sensitivity_analysis(
        decision_matrix=X,
        weights=base_weights,
        indicator_types=indicator_types,
        perturbation_range=perturbation_range,
    )
    return df


__all__ = [
    "EvaluationInputs",
    "EvaluationOutputs",
    "EvaluationPipeline",
    "run_weight_sensitivity",
]
