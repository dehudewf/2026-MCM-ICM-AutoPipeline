#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MCM 2026 Problem A: Enhancement Integration Module
================================================================================

This module integrates the O-Award enhancements into the existing pipeline:

1. Task 1 Enhancement: Enhanced Physics Model (3-equation coupling)
2. Task 2 Enhancement: OU→TTE Monte Carlo uncertainty propagation  
3. Task 3 Enhancement: Power decomposition Sobol sensitivity
4. Task 4 Enhancement: O-Award visualizations

Usage:
    # Import and use the enhanced pipeline
    from src.pipeline_enhanced import EnhancedMCMBatteryPipeline
    
    pipeline = EnhancedMCMBatteryPipeline()
    results = pipeline.run_full_pipeline_enhanced()

Author: MCM Team 2026
================================================================================
"""

import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import existing pipeline
from .pipeline import MCMBatteryPipeline
from .config import SEED, DEFAULT_OUTPUT_DIR, SCENARIOS, SOC_LEVELS

# Import enhancement modules
from .enhanced_physics_model import (
    EnhancedPhysicsBatteryModel,
    EnhancedBatteryParams,
    EnhancedBatterySimulator
)
from .ou_tte_propagation import (
    MonteCarloTTEPropagator,
    OU_SCENARIO_PARAMS,
    plot_tte_uncertainty_heatmap,
    plot_deterministic_vs_stochastic_comparison
)
from .power_decomposition import (
    LinearPowerModel,
    NonlinearPowerCorrector,
    SobolPowerSensitivity,
    SCENARIO_PROFILES,
    plot_power_decomposition_stacked_bar,
    plot_sobol_heatmap,
    plot_nonlinear_correction_impact
)
from .oaward_visualizations import (
    plot_tte_matrix_heatmap,
    plot_soc_trajectories,
    plot_robustness_contour,
    plot_sobol_sensitivity_heatmap,
    plot_uncertainty_comparison,
    plot_composite_figure,
    plot_temperature_efficiency_curve,
    apply_nature_style
)

logger = logging.getLogger(__name__)


class EnhancedMCMBatteryPipeline(MCMBatteryPipeline):
    """
    Enhanced Pipeline with O-Award physics deepening and uncertainty quantification.
    
    Extends MCMBatteryPipeline with:
    1. Enhanced 5-state ODE with electrochemical-thermal-aging coupling
    2. OU→TTE Monte Carlo uncertainty propagation
    3. Sobol power component sensitivity analysis
    4. Publication-quality composite figures
    
    Compatible with existing run_pipeline.py workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced pipeline."""
        super().__init__(config)
        
        # Enhancement-specific configuration
        self.mc_samples = self.config.get('mc_samples', 100)  # Monte Carlo samples
        self.enhanced_output_dir = self.output_dir / 'enhanced_results'
        self.enhanced_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.enhanced_data_dir = self.enhanced_output_dir / 'data'
        self.enhanced_figures_dir = self.enhanced_output_dir / 'figures'
        self.enhanced_data_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize enhanced components
        self.enhanced_params = EnhancedBatteryParams()
        self.enhanced_simulator = EnhancedBatterySimulator(self.enhanced_params)
        self.mc_propagator = MonteCarloTTEPropagator(N_samples=self.mc_samples, seed=SEED)
        self.linear_power = LinearPowerModel()
        self.sobol_analyzer = SobolPowerSensitivity(N_samples=500, seed=SEED)
        
        # Enhanced results storage
        self.enhanced_results: Dict[str, Any] = {
            'metadata': {
                'enhancements': [
                    'enhanced_physics_model',
                    'ou_tte_propagation', 
                    'power_sobol_sensitivity',
                    'oaward_visualizations'
                ],
                'mc_samples': self.mc_samples
            }
        }
        
        logger.info("=" * 60)
        logger.info("Enhanced MCMBatteryPipeline Initialized")
        logger.info("=" * 60)
        logger.info(f"Monte Carlo samples: {self.mc_samples}")
        logger.info(f"Enhanced output: {self.enhanced_output_dir}")
    
    # =========================================================================
    # Task 1 Enhancement: Enhanced Physics Model
    # =========================================================================
    
    def run_task1_enhanced_physics(self) -> Dict[str, Any]:
        """
        Task 1 Enhancement: Run enhanced physics model with 3-equation coupling.
        
        Implements:
        - 5-state ODE: [SOC, V_RC, Θ, F, Ah_throughput]
        - Electrochemical-thermal-aging coupling
        - SEI capacity fade model
        
        Returns
        -------
        Dict[str, Any]
            Enhanced Task 1 results
        """
        logger.info("\n" + "=" * 60)
        logger.info("TASK 1 ENHANCEMENT: Enhanced Physics Model")
        logger.info("=" * 60)
        
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
        soc_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        # Run 25-scenario matrix with enhanced model
        logger.info("\nRunning 25-scenario matrix with enhanced 5-state ODE...")
        matrix_results = self.enhanced_simulator.run_scenario_matrix(
            scenarios=scenarios,
            soc_levels=soc_levels
        )
        
        tte_matrix = matrix_results['matrix']
        
        # Print matrix
        logger.info("\nEnhanced TTE Matrix (hours):")
        logger.info("-" * 60)
        for i, soc in enumerate(soc_levels):
            row = tte_matrix[i]
            row_str = f"SOC₀={int(soc*100):3d}%  " + "  ".join([f"{v:7.2f}" for v in row])
            logger.info(row_str)
        
        # Run full trajectories for SOC=100%
        trajectory_results = {}
        for scenario in scenarios:
            model = EnhancedPhysicsBatteryModel(self.enhanced_params, scenario)
            result = model.simulate(SOC0=1.0)
            trajectory_results[scenario] = result
            logger.info(f"  {scenario}: TTE={result['TTE_hours']:.2f}h, "
                       f"T_final={result['Theta_C'][-1]:.1f}°C, "
                       f"F_final={result['F'][-1]*100:.1f}%")
        
        # Save TTE matrix
        df_tte = pd.DataFrame(
            tte_matrix,
            index=[f"SOC_{int(s*100)}" for s in soc_levels],
            columns=scenarios
        )
        df_tte.to_csv(self.enhanced_data_dir / 'tte_matrix_enhanced.csv')
        logger.info(f"✓ Saved: tte_matrix_enhanced.csv")
        
        # Generate visualizations
        plot_tte_matrix_heatmap(
            tte_matrix, scenarios, soc_levels,
            output_path=str(self.enhanced_figures_dir / 'tte_matrix_heatmap.png')
        )
        
        plot_soc_trajectories(
            trajectory_results, scenarios,
            output_path=str(self.enhanced_figures_dir / 'soc_trajectories.png')
        )
        
        plot_temperature_efficiency_curve(
            output_path=str(self.enhanced_figures_dir / 'temperature_efficiency.png')
        )
        
        return {
            'tte_matrix': tte_matrix.tolist(),
            'trajectory_results': {k: {'TTE_hours': v['TTE_hours']} for k, v in trajectory_results.items()},
            'scenarios': scenarios,
            'soc_levels': soc_levels
        }
    
    # =========================================================================
    # Task 2/3 Enhancement: OU→TTE Uncertainty Propagation
    # =========================================================================
    
    def run_task23_ou_uncertainty(self) -> Dict[str, Any]:
        """
        Task 2/3 Enhancement: OU→TTE Monte Carlo uncertainty propagation.
        
        Implements:
        - Deterministic vs Stochastic TTE comparison
        - Full grid uncertainty analysis
        - 95% CI quantification
        
        Returns
        -------
        Dict[str, Any]
            Uncertainty analysis results
        """
        logger.info("\n" + "=" * 60)
        logger.info("TASK 2/3 ENHANCEMENT: OU→TTE Uncertainty Propagation")
        logger.info("=" * 60)
        logger.info(f"Monte Carlo samples per cell: {self.mc_samples}")
        
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
        soc_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        # Deterministic vs Stochastic comparison
        logger.info("\nRunning deterministic vs stochastic comparison...")
        comparison_results = []
        for scenario in scenarios:
            comp = self.mc_propagator.compare_deterministic_vs_stochastic(scenario, SOC0=1.0)
            comparison_results.append(comp)
            logger.info(f"  {scenario:12s}: Det={comp['deterministic']['TTE_hours']:.2f}h, "
                       f"Stoch={comp['stochastic']['TTE_mean']:.2f}h ± {comp['stochastic']['CI_width']/2:.2f}h")
        
        # Full grid propagation
        logger.info("\nRunning full grid uncertainty analysis...")
        grid_result = self.mc_propagator.propagate_full_grid(
            scenarios=scenarios,
            soc_levels=soc_levels,
            verbose=True
        )
        
        # Save uncertainty data
        uncertainty_data = []
        for scenario in scenarios:
            for soc in soc_levels:
                key = f"{scenario}_{int(soc*100)}"
                if key in grid_result['all_results']:
                    r = grid_result['all_results'][key]
                    uncertainty_data.append({
                        'Scenario': scenario,
                        'SOC_initial': soc,
                        'TTE_median': r['median'],
                        'TTE_mean': r['mean'],
                        'TTE_std': r['std'],
                        'CI_95_lower': r['ci_95_lower'],
                        'CI_95_upper': r['ci_95_upper'],
                        'CI_95_width': r['ci_95_width'],
                        'Relative_uncertainty_pct': r['ci_95_width'] / r['median'] * 100 if r['median'] > 0 else 0
                    })
        
        df_uncertainty = pd.DataFrame(uncertainty_data)
        df_uncertainty.to_csv(self.enhanced_data_dir / 'uncertainty_analysis.csv', index=False)
        logger.info(f"✓ Saved: uncertainty_analysis.csv")
        
        # Generate visualizations
        plot_tte_uncertainty_heatmap(
            grid_result,
            output_path=str(self.enhanced_figures_dir / 'tte_uncertainty_heatmap.png')
        )
        
        plot_deterministic_vs_stochastic_comparison(
            comparison_results,
            output_path=str(self.enhanced_figures_dir / 'det_vs_stoch_comparison.png')
        )
        
        # Prepare data for main comparison figure
        deterministic_tte = {r['scenario']: r['deterministic']['TTE_hours'] for r in comparison_results}
        stochastic_tte = {
            r['scenario']: {
                'mean': r['stochastic']['TTE_mean'],
                'ci_lower': r['stochastic']['CI_95_lower'],
                'ci_upper': r['stochastic']['CI_95_upper']
            }
            for r in comparison_results
        }
        
        plot_uncertainty_comparison(
            deterministic_tte, stochastic_tte, scenarios,
            output_path=str(self.enhanced_figures_dir / 'uncertainty_comparison.png')
        )
        
        return {
            'comparison_results': [
                {
                    'scenario': r['scenario'],
                    'det_tte': r['deterministic']['TTE_hours'],
                    'stoch_tte_mean': r['stochastic']['TTE_mean'],
                    'ci_width': r['stochastic']['CI_width']
                }
                for r in comparison_results
            ],
            'tte_median_matrix': grid_result['tte_median_matrix'].tolist(),
            'ci_width_matrix': grid_result['ci_width_matrix'].tolist(),
            'deterministic_tte': deterministic_tte,
            'stochastic_tte': stochastic_tte
        }
    
    # =========================================================================
    # Task 3 Enhancement: Power Sobol Sensitivity
    # =========================================================================
    
    def run_task3_power_sensitivity(self) -> Dict[str, Any]:
        """
        Task 3 Enhancement: Power decomposition Sobol sensitivity analysis.
        
        Implements:
        - Linear power model analysis
        - Nonlinear correction impact assessment
        - Sobol first-order sensitivity indices
        
        Returns
        -------
        Dict[str, Any]
            Power sensitivity analysis results
        """
        logger.info("\n" + "=" * 60)
        logger.info("TASK 3 ENHANCEMENT: Power Sobol Sensitivity")
        logger.info("=" * 60)
        
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
        
        # Linear model analysis
        logger.info("\n3.1 Linear Power Model Analysis")
        linear_results = []
        for scenario in scenarios:
            result = self.linear_power.scenario_analysis(scenario)
            linear_results.append(result)
            logger.info(f"  {scenario:12s}: P={result['total_power']:.2f}W ± {result['total_std']:.3f}W")
        
        # Nonlinear correction analysis
        logger.info("\n3.2 Nonlinear Correction Impact Analysis")
        corrector = NonlinearPowerCorrector()
        correction_results = []
        for scenario in scenarios:
            result = corrector.corrected_power(scenario, sigma_ou=0.5, R_int=0.06, T_K=303.15)
            correction_results.append(result)
            logger.info(f"  {scenario:12s}: Linear={result['P_linear']:.2f}W, "
                       f"Corrected={result['P_corrected']:.2f}W (+{result['correction_pct']:.2f}%)")
        
        # Sobol sensitivity
        logger.info("\n3.3 Sobol Sensitivity Analysis")
        sobol_result = self.sobol_analyzer.full_scenario_sobol_analysis(scenarios=scenarios)
        
        # Print Sobol matrix
        components = sobol_result['components']
        logger.info("\nFirst-Order Sobol Indices (S₁):")
        for i, scenario in enumerate(scenarios):
            row = sobol_result['S1_matrix'][i]
            top_idx = np.argmax(row)
            logger.info(f"  {scenario:12s}: Top={components[top_idx].upper()} (S₁={row[top_idx]:.3f})")
        
        # Save data
        power_data = []
        for scenario in scenarios:
            profile = SCENARIO_PROFILES[scenario]
            for comp_name, power in profile.components.items():
                power_data.append({
                    'Scenario': scenario,
                    'Component': comp_name,
                    'Power_W': power,
                    'Fraction_pct': power / profile.total_power * 100 if profile.total_power > 0 else 0
                })
        
        df_power = pd.DataFrame(power_data)
        df_power.to_csv(self.enhanced_data_dir / 'power_decomposition.csv', index=False)
        
        df_sobol = pd.DataFrame(
            sobol_result['S1_matrix'],
            index=scenarios,
            columns=components
        )
        df_sobol.to_csv(self.enhanced_data_dir / 'sobol_indices.csv')
        logger.info(f"✓ Saved: power_decomposition.csv, sobol_indices.csv")
        
        # Generate visualizations
        plot_power_decomposition_stacked_bar(
            scenarios,
            output_path=str(self.enhanced_figures_dir / 'power_decomposition_stacked.png')
        )
        
        plot_sobol_heatmap(
            sobol_result,
            output_path=str(self.enhanced_figures_dir / 'sobol_power_heatmap.png')
        )
        
        plot_nonlinear_correction_impact(
            scenarios,
            output_path=str(self.enhanced_figures_dir / 'nonlinear_correction_impact.png')
        )
        
        plot_sobol_sensitivity_heatmap(
            sobol_result['S1_matrix'], scenarios, components,
            output_path=str(self.enhanced_figures_dir / 'sobol_sensitivity.png')
        )
        
        return {
            'linear_results': [
                {'scenario': r['scenario'], 'total_power': r['total_power'], 'total_std': r['total_std']}
                for r in linear_results
            ],
            'nonlinear_corrections': [
                {'scenario': r['scenario'], 'correction_pct': r['correction_pct']}
                for r in correction_results
            ],
            'sobol_matrix': sobol_result['S1_matrix'].tolist(),
            'components': components
        }
    
    # =========================================================================
    # Composite Figure Generation
    # =========================================================================
    
    def generate_composite_figure(self, task1_results: Dict, task23_results: Dict, 
                                    task3_results: Dict) -> None:
        """Generate the main composite publication figure."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPOSITE PUBLICATION FIGURE")
        logger.info("=" * 60)
        
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
        soc_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        # Prepare data
        tte_matrix = np.array(task1_results['tte_matrix'])
        sobol_matrix = np.array(task3_results['sobol_matrix'])
        components = task3_results['components']
        
        # Prepare trajectory results (simplified for composite)
        trajectory_results = {}
        for scenario in scenarios:
            model = EnhancedPhysicsBatteryModel(self.enhanced_params, scenario)
            result = model.simulate(SOC0=1.0)
            trajectory_results[scenario] = result
        
        # Prepare comparison data
        comparison_data = {
            'deterministic': [task23_results['deterministic_tte'].get(s, 0) for s in scenarios],
            'stochastic_mean': [task23_results['stochastic_tte'].get(s, {}).get('mean', 0) for s in scenarios],
            'ci_half': [
                (task23_results['stochastic_tte'].get(s, {}).get('ci_upper', 0) - 
                 task23_results['stochastic_tte'].get(s, {}).get('ci_lower', 0)) / 2
                for s in scenarios
            ]
        }
        
        plot_composite_figure(
            tte_matrix=tte_matrix,
            trajectory_results=trajectory_results,
            sobol_matrix=sobol_matrix,
            comparison_data=comparison_data,
            scenarios=scenarios,
            soc_levels=soc_levels,
            parameters=components,
            output_path=str(self.enhanced_figures_dir / 'composite_figure.png')
        )
    
    # =========================================================================
    # Enhanced Full Pipeline
    # =========================================================================
    
    def run_full_pipeline_enhanced(self) -> Dict[str, Any]:
        """
        Run complete enhanced pipeline (original + enhancements).
        
        Executes:
        1. Original pipeline (Tasks 1-4)
        2. Task 1 Enhancement: Enhanced physics model
        3. Task 2/3 Enhancement: OU→TTE uncertainty
        4. Task 3 Enhancement: Power Sobol sensitivity
        5. Composite figure generation
        
        Returns
        -------
        Dict[str, Any]
            Combined results from original and enhanced pipeline
        """
        logger.info("\n" + "=" * 70)
        logger.info("     MCM 2026 Problem A: ENHANCED FULL PIPELINE")
        logger.info("     Original Pipeline + O-Award Enhancements")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # =====================================================================
        # PHASE 1: Run Original Pipeline
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Original Pipeline (Tasks 1-4)")
        logger.info("=" * 60)
        
        original_results = super().run_full_pipeline()
        self.results.update(original_results)
        
        # =====================================================================
        # PHASE 2: Run Enhancements
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: O-Award Enhancements")
        logger.info("=" * 60)
        
        # Task 1 Enhancement
        task1_enhanced = self.run_task1_enhanced_physics()
        self.enhanced_results['task1_enhanced'] = task1_enhanced
        
        # Task 2/3 Enhancement
        task23_enhanced = self.run_task23_ou_uncertainty()
        self.enhanced_results['task23_ou_uncertainty'] = task23_enhanced
        
        # Task 3 Enhancement
        task3_enhanced = self.run_task3_power_sensitivity()
        self.enhanced_results['task3_power_sensitivity'] = task3_enhanced
        
        # Composite Figure
        self.generate_composite_figure(task1_enhanced, task23_enhanced, task3_enhanced)
        
        # =====================================================================
        # PHASE 3: Generate Enhanced Summary Report
        # =====================================================================
        elapsed_time = time.time() - start_time
        self._generate_enhanced_summary_report(elapsed_time)
        
        # Combine results
        combined_results = {
            'original': self.results,
            'enhanced': self.enhanced_results,
            'metadata': {
                'total_runtime_seconds': elapsed_time,
                'enhanced_output_dir': str(self.enhanced_output_dir)
            }
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("  ENHANCED PIPELINE COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"  Total Runtime: {elapsed_time:.1f} seconds")
        logger.info(f"  Original Output: {self.output_dir}")
        logger.info(f"  Enhanced Output: {self.enhanced_output_dir}")
        logger.info("=" * 70)
        
        return combined_results
    
    def _generate_enhanced_summary_report(self, elapsed_time: float) -> None:
        """Generate enhanced summary report."""
        report = f"""# MCM 2026 Problem A: Enhanced Pipeline Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Runtime**: {elapsed_time:.1f} seconds

---

## Enhancement Summary

| Enhancement | Status | Key Output |
|-------------|--------|------------|
| Enhanced Physics Model | ✅ Complete | 5-state ODE with thermal-aging coupling |
| OU→TTE Uncertainty | ✅ Complete | Monte Carlo propagation ({self.mc_samples} samples) |
| Power Sobol Sensitivity | ✅ Complete | First-order indices for 6 components |
| Composite Figure | ✅ Complete | Publication-ready 4-panel figure |

---

## Key Formulas (Paper-Ready)

### Enhanced SOC Dynamics
$$\\frac{{d\\xi}}{{dt}} = -\\frac{{I(t)}}{{Q_{{eff}}(\\Theta, F, Ah) \\cdot \\eta_{{eff}}(I, V_p, \\Theta)}}$$

### OU Power Process  
$$dP_t = \\theta(\\mu - P_t)dt + \\sigma dW_t$$

### Linear Power Decomposition
$$P_{{total}} = P_{{display}} + P_{{cpu}} + P_{{gpu}} + P_{{network}} + P_{{gps}} + P_{{bg}}$$

---

## Output Files

### Enhanced Data Files
- `tte_matrix_enhanced.csv` - 25-scenario TTE with enhanced model
- `uncertainty_analysis.csv` - Full MC results with 95% CI
- `power_decomposition.csv` - Component-level power breakdown
- `sobol_indices.csv` - Sobol first-order indices

### Enhanced Figures
- `composite_figure.png` - **MAIN PAPER FIGURE**
- `tte_matrix_heatmap.png` - TTE prediction matrix
- `soc_trajectories.png` - SOC discharge curves
- `uncertainty_comparison.png` - Det. vs Stoch. TTE
- `sobol_sensitivity.png` - Parameter sensitivity heatmap
- `temperature_efficiency.png` - Temperature efficiency curve

---

*Report generated by Enhanced MCM 2026 Pipeline*
"""
        
        report_path = self.enhanced_output_dir / 'enhanced_summary_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Saved: {report_path}")


def main_enhanced():
    """Main entry point for enhanced pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MCM 2026 Enhanced Battery Pipeline')
    parser.add_argument('--mc-samples', type=int, default=100, 
                        help='Monte Carlo samples per cell (default: 100)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--enhanced-only', action='store_true',
                        help='Run only enhanced components (skip original pipeline)')
    args = parser.parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'mc_samples': args.mc_samples
    }
    
    pipeline = EnhancedMCMBatteryPipeline(config)
    
    if args.enhanced_only:
        # Run only enhancements
        logger.info("Running enhanced components only...")
        task1 = pipeline.run_task1_enhanced_physics()
        task23 = pipeline.run_task23_ou_uncertainty()
        task3 = pipeline.run_task3_power_sensitivity()
        pipeline.generate_composite_figure(task1, task23, task3)
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline_enhanced()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main_enhanced())
