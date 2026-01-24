"""
Complete end-to-end example: Light Pollution Risk Assessment using AHP+EWM+TOPSIS

This script demonstrates the full thinker-executor-checker workflow:
1. Load synthetic data (or replace with real data from NASA, World Bank, etc.)
2. Run evaluation pipeline (AHP + EWM + TOPSIS)
3. Run weight sensitivity analysis (±20%)
4. Output results ready for @redcell attack

Usage:
    cd E题
    python src/main_light_pollution.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.analysis.evaluation_pipeline import (
    EvaluationInputs,
    EvaluationPipeline,
    run_weight_sensitivity,
)
from src.data.light_pollution_data import (
    generate_light_pollution_indicators,
    get_indicator_metadata,
    generate_ahp_judgment_matrix,
)
from src.config import config

# Reproducibility
np.random.seed(config.random_seed)


def main():
    print("=" * 80)
    print("2023 ICM Problem E: Light Pollution Risk Assessment")
    print("AHP + EWM + TOPSIS Evaluation Pipeline")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Step 1: Load data
    # ========================================================================
    print("[1/5] Loading indicator data...")
    df = generate_light_pollution_indicators()
    print(df.to_string(index=False))
    print()
    
    # Extract decision matrix (drop location names column)
    X = df.drop(columns=['Location']).values
    location_names = df['Location'].tolist()
    
    criteria_names, indicator_types_ewm, indicator_is_benefit = get_indicator_metadata()
    
    print(f"Decision matrix shape: {X.shape}")
    print(f"  {X.shape[0]} locations × {X.shape[1]} indicators")
    print()
    
    # ========================================================================
    # Step 2: Prepare inputs
    # ========================================================================
    print("[2/5] Preparing evaluation inputs...")
    J = generate_ahp_judgment_matrix()
    
    inputs = EvaluationInputs(
        decision_matrix=X,
        criteria_names=criteria_names,
        indicator_types_ewm=indicator_types_ewm,
        indicator_is_benefit=indicator_is_benefit,
        alternative_names=location_names,
        ahp_judgment_matrix=J,
        alpha=0.5,  # 50% AHP + 50% EWM
    )
    print(f"  AHP judgment matrix: {J.shape}")
    print(f"  Weight combination: α = {inputs.alpha} (AHP/EWM balance)")
    print()
    
    # ========================================================================
    # Step 3: Run evaluation pipeline
    # ========================================================================
    print("[3/5] Running AHP + EWM + TOPSIS evaluation...")
    pipeline = EvaluationPipeline()
    outputs = pipeline.run(inputs)
    
    # Display AHP results
    if outputs.ahp_result is not None:
        print("\nAHP Results:")
        print("-" * 60)
        print(f"  Consistency Ratio (CR): {outputs.ahp_result.cr:.4f}")
        if outputs.ahp_result.is_consistent:
            print(f"  ✓ CR < {config.ahp.consistency_threshold} (consistent)")
        else:
            print(f"  ✗ CR ≥ {config.ahp.consistency_threshold} (inconsistent)")
        print()
        df_ahp = outputs.ahp_result.to_dataframe()
        print(df_ahp.to_string(index=False))
    
    # Display EWM results
    print("\nEWM Results:")
    print("-" * 60)
    df_ewm = outputs.ewm_result.to_dataframe()
    print(df_ewm.to_string(index=False))
    
    # Display combined weights
    print("\nCombined Weights (α=0.5):")
    print("-" * 60)
    df_weights = pd.DataFrame({
        'Criterion': criteria_names,
        'AHP': outputs.ahp_result.weights if outputs.ahp_result else [np.nan] * len(criteria_names),
        'EWM': outputs.ewm_result.weights,
        'Combined': outputs.combined_weights,
    })
    print(df_weights.to_string(index=False))
    print()
    
    # Display TOPSIS ranking
    print("\nTOPSIS Evaluation Results:")
    print("-" * 60)
    df_scores = outputs.topsis_result.to_dataframe()
    print(df_scores.to_string(index=False))
    print()
    
    # Interpretation
    top_location = df_scores.iloc[0]['Alternative']
    top_score = df_scores.iloc[0]['Score']
    print(f"📊 Interpretation:")
    print(f"  → {top_location} has the LOWEST light pollution risk (score={top_score:.4f})")
    print(f"  → Risk ranking: {' > '.join(df_scores['Alternative'].tolist())}")
    print()
    
    # ========================================================================
    # Step 4: Sensitivity analysis
    # ========================================================================
    print("[4/5] Running weight sensitivity analysis (±20%)...")
    df_sens = run_weight_sensitivity(pipeline, inputs, perturbation_range=0.20)
    
    print("\nWeight Sensitivity Summary:")
    print("-" * 60)
    print(df_sens[['Criterion_Index', 'Direction', 'Rank_Changes', 'Max_Score_Change', 'Ranking_Stable']].to_string(index=False))
    
    # Count stable scenarios
    n_stable = df_sens['Ranking_Stable'].sum()
    n_total = len(df_sens)
    stability_rate = n_stable / n_total
    print()
    print(f"Stability rate: {n_stable}/{n_total} scenarios ({stability_rate:.1%})")
    if stability_rate > 0.8:
        print("✓ Model is STABLE to weight perturbations")
    else:
        print("⚠️  Model is SENSITIVE to weight changes")
    print()
    
    # ========================================================================
    # Step 5: Output for @redcell checker
    # ========================================================================
    print("[5/5] Outputs ready for @redcell attack:")
    print("-" * 60)
    print("  1. Indicator system: 8 indicators across 4 dimensions")
    print("  2. Weight scheme: AHP (expert) + EWM (data-driven)")
    print(f"  3. AHP CR = {outputs.ahp_result.cr:.4f} {'✓' if outputs.ahp_result.is_consistent else '✗'}")
    print(f"  4. TOPSIS ranking: {' > '.join(df_scores['Alternative'].tolist())}")
    print(f"  5. Sensitivity: {stability_rate:.1%} stable scenarios")
    print()
    print("Next: @redcell should attack:")
    print("  → Indicator completeness (missing dimensions?)")
    print("  → AHP judgment matrix rationality")
    print("  → Ranking robustness under extreme perturbations")
    print()
    print("=" * 80)
    print("✓ Evaluation complete. Ready for checker phase.")
    print("=" * 80)


if __name__ == "__main__":
    main()
