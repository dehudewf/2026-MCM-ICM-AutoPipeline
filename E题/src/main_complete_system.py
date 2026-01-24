"""
Complete E-Problem Evaluation System
=====================================

Integrates:
1. Real data integration (NASA, World Bank sources)
2. @redcell attack phase
3. Visualization generation

Usage:
    python main_complete_system.py [--use-real-data] [--skip-extreme-tests]
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd

from src.data.light_pollution_data import (
    generate_light_pollution_indicators,
    get_indicator_metadata,
    generate_ahp_judgment_matrix,
)
from src.data.data_validator import DataValidator
from src.analysis.evaluation_pipeline import (
    EvaluationInputs,
    EvaluationPipeline,
    run_weight_sensitivity,
)
from src.analysis.redcell_checker import RedCellChecker
from src.analysis.visualizer import EvaluationVisualizer
from src.analysis.uncertainty_analyzer import UncertaintyAnalyzer
from src.optimization.intervention_optimizer import (
    InterventionOptimizer,
    create_default_interventions,
)
from src.config import config


def main(use_real_data: bool = False, skip_extreme_tests: bool = False):
    """Run complete E-problem evaluation system."""
    
    print("=" * 80)
    print(" MCM 2023 Problem E: Light Pollution Risk Assessment")
    print(" Complete Evaluation System with Enhanced Modules")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Data Loading
    # ========================================================================
    print("\n[Step 1] Loading data...")
    
    df = generate_light_pollution_indicators(use_real_data=use_real_data)
    print(f"✓ Loaded {len(df)} locations")
    print(df)
    
    # Extract decision matrix
    X = df.drop(columns=['Location']).values
    alternative_names = df['Location'].tolist()
    
    # Get metadata
    criteria_names, indicator_types_ewm, indicator_is_benefit = get_indicator_metadata()
    
    # Generate AHP judgment matrix
    J = generate_ahp_judgment_matrix()
    
    # ========================================================================
    # Step 1.5: Data Validation
    # ========================================================================
    print("\n[Step 1.5] Validating data quality...")
    
    validator = DataValidator(
        outlier_method='iqr',
        outlier_threshold=1.5,
        missing_value_threshold=0.10,
    )
    
    validation_results = validator.validate_decision_matrix(
        decision_matrix=X,
        criteria_names=criteria_names,
    )
    
    validator.print_report(validation_results)
    
    if validator.has_fatal_errors(validation_results):
        print("\n❌ FATAL DATA ERRORS - Cannot proceed")
        return
    
    # ========================================================================
    # Step 2: Prepare Evaluation Inputs
    # ========================================================================
    print("\n[Step 2] Preparing evaluation inputs...")
    
    inputs = EvaluationInputs(
        decision_matrix=X,
        criteria_names=criteria_names,
        indicator_types_ewm=indicator_types_ewm,
        indicator_is_benefit=indicator_is_benefit,
        alternative_names=alternative_names,
        ahp_judgment_matrix=J,
        alpha=config.alpha_combined_weight,
    )
    
    print(f"✓ Decision matrix: {X.shape}")
    print(f"✓ Indicators: {len(criteria_names)}")
    print(f"✓ Alternatives: {len(alternative_names)}")
    
    # ========================================================================
    # Step 3: Run Evaluation Pipeline
    # ========================================================================
    print("\n[Step 3] Running AHP + EWM + TOPSIS evaluation...")
    
    pipeline = EvaluationPipeline()
    outputs = pipeline.run(inputs)
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # AHP results
    if outputs.ahp_result is not None:
        print(f"\nAHP Consistency Ratio: {outputs.ahp_result.cr:.4f}")
        if outputs.ahp_result.cr < 0.1:
            print("✓ Passed (CR < 0.1)")
        else:
            print("✗ Failed (CR ≥ 0.1) - Matrix is inconsistent")
        
        print("\nAHP Weights (Subjective):")
        for name, weight in zip(criteria_names, outputs.ahp_result.weights):
            print(f"  {name:25s}: {weight:.4f}")
    
    # EWM results
    print("\nEWM Weights (Objective):")
    for name, weight in zip(criteria_names, outputs.ewm_result.weights):
        print(f"  {name:25s}: {weight:.4f}")
    
    # Combined weights
    print("\nCombined Weights (α=0.5):")
    for name, weight in zip(criteria_names, outputs.combined_weights):
        print(f"  {name:25s}: {weight:.4f}")
    
    # TOPSIS ranking
    print("\nTOPSIS Ranking:")
    df_result = outputs.topsis_result.to_dataframe()
    print(df_result.to_string(index=False))
    
    # ========================================================================
    # Step 4: Sensitivity Analysis
    # ========================================================================
    print("\n[Step 4] Running sensitivity analysis (±20% weight perturbation)...")
    
    perturbation_range = config.sensitivity.weight_variation  # 0.20
    df_sensitivity = run_weight_sensitivity(pipeline, inputs, perturbation_range)
    
    print(f"\n✓ Tested {len(df_sensitivity)} scenarios")
    print("\nTop 5 Most Sensitive Indicators:")
    top_sensitive = df_sensitivity.nlargest(5, 'Max_Score_Change')
    
    # Add indicator names
    top_sensitive_display = top_sensitive.copy()
    top_sensitive_display['Indicator'] = top_sensitive_display['Criterion_Index'].apply(
        lambda i: criteria_names[i]
    )
    print(top_sensitive_display[['Indicator', 'Direction', 'Max_Score_Change', 'Ranking_Stable']].to_string(index=False))
    
    # Stability rate
    n_stable = len(df_sensitivity[df_sensitivity['Ranking_Stable'] == True])
    n_total = len(df_sensitivity)
    stability_rate = n_stable / n_total
    print(f"\nStability rate: {n_stable}/{n_total} scenarios ({stability_rate:.1%})")
    
    if stability_rate >= 0.8:
        print("✓ Model is STABLE to weight perturbations")
    else:
        print("⚠ Model shows INSTABILITY - review indicator weights")
    
    # ========================================================================
    # Step 5: @redcell Attack Phase
    # ========================================================================
    print("\n[Step 5] Launching @redcell attack phase...")
    print("=" * 60)
    
    checker = RedCellChecker()
    attack_report = checker.attack(
        inputs=inputs,
        outputs=outputs,
        run_extreme_tests=(not skip_extreme_tests),
    )
    
    # Display summary
    print(attack_report.summary())
    
    # Display detailed findings
    if len(attack_report.attacks) > 0:
        print("\nDETAILED FINDINGS:")
        print("-" * 80)
        df_attacks = attack_report.to_dataframe()
        
        # Group by severity
        for severity in ['FATAL', 'CRITICAL', 'MAJOR', 'MINOR']:
            issues = attack_report.get_by_severity(severity)
            if issues:
                print(f"\n{severity} Issues ({len(issues)}):")
                for i, issue in enumerate(issues, 1):
                    print(f"\n  [{i}] {issue.dimension}")
                    print(f"      Issue: {issue.issue}")
                    print(f"      Evidence: {issue.evidence}")
                    print(f"      Impact: {issue.impact}")
                    print(f"      Recommendation: {issue.recommendation}")
    else:
        print("\n✓ No issues found - Model passed all red cell attacks!")
    
    # Save attack report
    from pathlib import Path
    output_dir = Path(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else 'output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    attack_csv_path = output_dir / 'redcell_attack_report.csv'
    df_attacks = attack_report.to_dataframe()
    df_attacks.to_csv(attack_csv_path, index=False)
    print(f"\n✓ Attack report saved to: {attack_csv_path}")
    
    # ========================================================================
    # Step 6: Uncertainty Quantification (Bootstrap)
    # ========================================================================
    print("\n[Step 6] Quantifying uncertainty (Bootstrap confidence intervals)...")
    print("=" * 60)
    
    uncertainty_analyzer = UncertaintyAnalyzer(random_seed=config.random_seed)
    
    uncertainty_result = uncertainty_analyzer.bootstrap_topsis_scores(
        decision_matrix=X,
        weights=outputs.combined_weights,
        indicator_types=indicator_is_benefit,
        n_bootstrap=config.uncertainty.n_bootstrap_samples,
        confidence_level=config.uncertainty.bootstrap_confidence_level,
        alternative_names=alternative_names,
        method=config.uncertainty.bootstrap_method,
    )
    
    df_uncertainty = uncertainty_analyzer.to_dataframe(uncertainty_result)
    print("\nBootstrap Confidence Intervals (95%):")
    print(df_uncertainty.to_string(index=False))
    print(f"\nRanking stability (top-1): {uncertainty_result.ranking_stability:.1%}")
    
    # Save uncertainty results
    uncertainty_csv = output_dir / 'uncertainty_analysis.csv'
    df_uncertainty.to_csv(uncertainty_csv, index=False)
    print(f"\u2713 Uncertainty analysis saved to: {uncertainty_csv}")
    
    # ========================================================================
    # Step 7: Intervention Strategy Optimization
    # ========================================================================
    print("\n[Step 7] Optimizing intervention strategies (mitigating pollution)...")
    print("=" * 60)
    
    optimizer = InterventionOptimizer(random_seed=config.random_seed)
    interventions = create_default_interventions()
    
    intervention_plan = optimizer.optimize_budget_allocation(
        current_pollution_scores=outputs.topsis_result.scores,
        location_names=alternative_names,
        interventions=interventions,
        budget=config.intervention.default_budget,
        objective=config.intervention.optimization_objective,
    )
    
    print(f"\nOptimization Status: {intervention_plan.status}")
    print(f"Budget Used: ${intervention_plan.budget_used:,.2f} / ${intervention_plan.budget_limit:,.0f}")
    print(f"Objective Value: {intervention_plan.objective_value:.3f}")
    
    df_interventions = optimizer.generate_intervention_recommendations(
        plan=intervention_plan,
        current_scores=outputs.topsis_result.scores,
    )
    
    print("\nIntervention Recommendations:")
    print(df_interventions.to_string(index=False))
    
    # Save intervention plan
    intervention_csv = output_dir / 'intervention_recommendations.csv'
    df_interventions.to_csv(intervention_csv, index=False)
    print(f"\n✓ Intervention plan saved to: {intervention_csv}")
    
    # ========================================================================
    # Step 8: Visualization Generation
    # ========================================================================
    print("\n[Step 8] Generating O-Award quality figures...")
    print("=" * 60)
    
    output_dir_str = str(output_dir)
    visualizer = EvaluationVisualizer(output_dir=output_dir_str)
    
    # Generate all figures
    figures = visualizer.generate_all_figures(
        outputs=outputs,
        sensitivity_df=df_sensitivity,
        decision_matrix=X,
    )
    
    # ========================================================================
    # Step 9: Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print(" EXECUTION SUMMARY")
    print("=" * 80)
    
    print("\n1. Data Source:")
    if use_real_data:
        print("   ✓ Using REAL data from NASA/World Bank")
    else:
        print("   ⚠ Using SYNTHETIC data (replace with real data for submission)")
    
    print("\n2. Model Results:")
    print(f"   - AHP CR: {outputs.ahp_result.cr:.4f} {'✓' if outputs.ahp_result.cr < 0.1 else '✗'}")
    print(f"   - Top ranked location: {outputs.topsis_result.alternative_names[np.argmax(outputs.topsis_result.scores)]}")
    print(f"   - Sensitivity stability: {stability_rate:.1%}")
    print(f"   - Ranking stability (Bootstrap): {uncertainty_result.ranking_stability:.1%}")
    
    print("\n3. Red Cell Attack:")
    counts = {
        'FATAL': len(attack_report.get_by_severity('FATAL')),
        'CRITICAL': len(attack_report.get_by_severity('CRITICAL')),
        'MAJOR': len(attack_report.get_by_severity('MAJOR')),
        'MINOR': len(attack_report.get_by_severity('MINOR')),
    }
    print(f"   - FATAL issues: {counts['FATAL']}")
    print(f"   - CRITICAL issues: {counts['CRITICAL']}")
    print(f"   - MAJOR issues: {counts['MAJOR']}")
    print(f"   - MINOR issues: {counts['MINOR']}")
    
    if attack_report.has_fatal_issues():
        print("\n   ❌ FATAL ISSUES - Must fix before submission")
    elif counts['CRITICAL'] > 0:
        print("\n   ⚠️ CRITICAL ISSUES - Highly recommended to fix")
    else:
        print("\n   ✓ No critical issues - Model ready for paper writing")
    
    print("\n4. Intervention Optimization:")
    print(f"   - Status: {intervention_plan.status}")
    print(f"   - Budget utilization: {intervention_plan.budget_used/intervention_plan.budget_limit:.1%}")
    expected_reduction_avg = np.mean(intervention_plan.expected_pollution_reduction) * 100
    print(f"   - Expected avg pollution reduction: {expected_reduction_avg:.1f}%")
    
    print("\n5. Output Files:")
    print(f"   - Attack report: {attack_csv_path}")
    print(f"   - Uncertainty analysis: {uncertainty_csv}")
    print(f"   - Intervention plan: {intervention_csv}")
    print(f"   - Figures (4 total): {output_dir}/")
    print(f"     • weight_comparison.png")
    print(f"     • topsis_ranking.png")
    print(f"     • sensitivity_heatmap.png")
    print(f"     • indicator_radar.png")
    
    print("\n" + "=" * 80)
    print(" NEXT STEPS")
    print("=" * 80)
    
    if use_real_data:
        print("\n✓ Real data integrated - Ready for paper writing")
    else:
        print("\n⚠ ACTION REQUIRED: Replace synthetic data with real sources:")
        print("  1. NASA Earthdata (earthdata.nasa.gov) - VIIRS night-time lights")
        print("  2. World Bank (data.worldbank.org) - GDP, population")
        print("  3. IUCN (iucnredlist.org) - Biodiversity data")
        print("  4. Run: python main_complete_system.py --use-real-data")
    
    if counts['FATAL'] > 0 or counts['CRITICAL'] > 0:
        print("\n⚠ FIX CRITICAL ISSUES:")
        for issue in attack_report.get_by_severity('FATAL') + attack_report.get_by_severity('CRITICAL'):
            print(f"  • {issue.issue}")
            print(f"    → {issue.recommendation}")
    
    print("\n✓ READY: Uncertainty quantification complete (Bootstrap CI)")
    print("✓ READY: Intervention strategies generated (mitigation module)")
    print("✓ READY: Use generated figures in paper")
    print("✓ READY: Reference attack report for Limitations section")
    print("✓ READY: Use sensitivity analysis for Sensitivity Analysis section")
    print("✓ READY: Use intervention plan for mitigation recommendations")
    
    print("\n" + "=" * 80)
    print(" System execution complete. All modules integrated.")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete E-Problem Evaluation System'
    )
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real data from NASA/World Bank (default: synthetic data)'
    )
    parser.add_argument(
        '--skip-extreme-tests',
        action='store_true',
        help='Skip expensive extreme perturbation tests (faster execution)'
    )
    
    args = parser.parse_args()
    
    main(
        use_real_data=args.use_real_data,
        skip_extreme_tests=args.skip_extreme_tests,
    )
