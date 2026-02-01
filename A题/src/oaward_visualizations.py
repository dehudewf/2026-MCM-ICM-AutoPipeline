#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MCM 2026 Problem A: O-Award Visualization Module
================================================================================

Publication-quality visualizations for O-Award paper submission.
All figures designed for:
    - 300+ DPI resolution
    - Grayscale readability
    - Consistent color scheme (Nature/Science style)
    - Self-explanatory captions

Visualization Categories:
    1. Model Results (TTE matrix, SOC trajectories)
    2. Sensitivity Analysis (Sobol heatmap, robustness contour)
    3. Uncertainty Analysis (CI visualization, distribution plots)
    4. Composite Figures (multi-panel publication figures)

Author: MCM Team 2026
Reference: A题可视化知识库.csv, Model_Formulas_Paper_Ready.md
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
import warnings

# ============================================================================
# Style Configuration (O-Award Publication Quality)
# ============================================================================

# Nature/Science color palette (colorblind-friendly)
NATURE_COLORS = {
    'blue': '#4575B4',
    'red': '#D73027',
    'green': '#1A9850',
    'orange': '#FC8D59',
    'purple': '#7570B3',
    'yellow': '#FEE090',
    'gray': '#969696',
    'cyan': '#66C2A5'
}

# Scenario colors (consistent throughout paper)
SCENARIO_COLORS = {
    'idle': NATURE_COLORS['green'],
    'browsing': NATURE_COLORS['blue'],
    'video': NATURE_COLORS['purple'],
    'gaming': NATURE_COLORS['red'],
    'navigation': NATURE_COLORS['orange']
}

# Standard figure settings
FIGURE_SETTINGS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}


def apply_nature_style():
    """Apply Nature/Science publication style."""
    plt.rcParams.update(FIGURE_SETTINGS)


# ============================================================================
# Figure 1: TTE Matrix Heatmap (25-Scenario Grid)
# ============================================================================

def plot_tte_matrix_heatmap(tte_matrix: np.ndarray,
                             scenarios: List[str],
                             soc_levels: List[float],
                             ci_matrix: np.ndarray = None,
                             title: str = None,
                             output_path: str = None) -> plt.Figure:
    """
    Generate publication-quality TTE matrix heatmap.
    
    Parameters
    ----------
    tte_matrix : np.ndarray
        TTE values [n_soc × n_scenarios]
    scenarios : List[str]
        Scenario names
    soc_levels : List[float]
        SOC levels
    ci_matrix : np.ndarray, optional
        CI half-widths for annotation
    title : str, optional
        Custom title
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    apply_nature_style()
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Custom colormap: Red (short) → Yellow → Green (long)
    cmap = LinearSegmentedColormap.from_list(
        'tte_cmap', [NATURE_COLORS['red'], NATURE_COLORS['yellow'], NATURE_COLORS['green']]
    )
    
    im = ax.imshow(tte_matrix, cmap=cmap, aspect='auto')
    
    # Annotate cells
    for i in range(len(soc_levels)):
        for j in range(len(scenarios)):
            tte_val = tte_matrix[i, j]
            
            if ci_matrix is not None:
                ci_val = ci_matrix[i, j]
                text = f'{tte_val:.1f}h\n±{ci_val:.1f}'
            else:
                text = f'{tte_val:.1f}h'
            
            # High contrast text
            text_color = 'white' if tte_val < 5 else 'black'
            
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=10, fontweight='bold', color=text_color)
    
    # Axis labels
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.capitalize() for s in scenarios], fontweight='bold')
    ax.set_yticks(range(len(soc_levels)))
    ax.set_yticklabels([f'{int(s*100)}%' for s in soc_levels], fontweight='bold')
    
    ax.set_xlabel('Usage Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Initial SOC ($\\xi_0$)', fontsize=13, fontweight='bold')
    
    if title is None:
        title = 'Time-to-Empty (TTE) Prediction Matrix\n(Enhanced Physics Model with 95% CI)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TTE (hours)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Figure 2: SOC Trajectory Comparison (5 Scenarios)
# ============================================================================

def plot_soc_trajectories(trajectory_results: Dict,
                           scenarios: List[str] = None,
                           output_path: str = None) -> plt.Figure:
    """
    Plot SOC discharge curves for multiple scenarios.
    
    Parameters
    ----------
    trajectory_results : Dict
        Results from model.simulate() per scenario
    scenarios : List[str], optional
        Scenarios to plot
    output_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
    """
    apply_nature_style()
    
    if scenarios is None:
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for scenario in scenarios:
        if scenario not in trajectory_results:
            continue
        
        result = trajectory_results[scenario]
        t = result['t_hours']
        soc = result['SOC'] * 100  # Convert to %
        
        color = SCENARIO_COLORS.get(scenario, 'gray')
        
        ax.plot(t, soc, linewidth=2.5, color=color, 
                label=f'{scenario.capitalize()} (TTE={result["TTE_hours"]:.1f}h)')
    
    # Threshold line
    ax.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Shutdown Threshold (5%)')
    
    ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('State of Charge (%)', fontsize=13, fontweight='bold')
    ax.set_title('SOC Discharge Trajectories by Usage Scenario\n'
                 '(Initial SOC = 100%, Enhanced Physics Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, None])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Figure 3: Robustness Contour Plot (SOH × Temperature → TTE)
# ============================================================================

def plot_robustness_contour(soh_range: np.ndarray,
                             temp_range: np.ndarray,
                             tte_grid: np.ndarray,
                             scenario: str = 'video',
                             output_path: str = None) -> plt.Figure:
    """
    Contour plot showing TTE variation across SOH and Temperature.
    
    Demonstrates model robustness to parameter variation.
    
    Parameters
    ----------
    soh_range : np.ndarray
        SOH values (x-axis)
    temp_range : np.ndarray
        Temperature values in °C (y-axis)
    tte_grid : np.ndarray
        TTE values [n_temp × n_soh]
    scenario : str
        Scenario name for title
    output_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
    """
    apply_nature_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for contour
    SOH, TEMP = np.meshgrid(soh_range * 100, temp_range)  # SOH in %, Temp in °C
    
    # Filled contours
    levels = np.linspace(tte_grid.min(), tte_grid.max(), 12)
    cf = ax.contourf(SOH, TEMP, tte_grid, levels=levels, cmap='RdYlGn', alpha=0.85)
    
    # Line contours
    cs = ax.contour(SOH, TEMP, tte_grid, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    
    # Optimal region annotation
    opt_soh = soh_range[len(soh_range)//2] * 100
    opt_temp = temp_range[len(temp_range)//2]
    ax.scatter([opt_soh], [opt_temp], marker='*', s=300, color='blue', 
               edgecolor='white', linewidth=2, zorder=5)
    ax.annotate('Optimal Operating Point', xy=(opt_soh, opt_temp),
                xytext=(opt_soh + 10, opt_temp + 8),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    ax.set_xlabel('State of Health (SOH) %', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title(f'TTE Robustness Analysis: {scenario.capitalize()} Scenario\n'
                 f'(SOC₀=100%, Contours = TTE hours)',
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TTE (hours)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Figure 4: Sobol Sensitivity Heatmap
# ============================================================================

def plot_sobol_sensitivity_heatmap(S1_matrix: np.ndarray,
                                    scenarios: List[str],
                                    parameters: List[str],
                                    title: str = None,
                                    output_path: str = None) -> plt.Figure:
    """
    Heatmap of Sobol first-order sensitivity indices.
    
    Parameters
    ----------
    S1_matrix : np.ndarray
        Sobol indices [n_scenarios × n_parameters]
    scenarios : List[str]
        Scenario names (rows)
    parameters : List[str]
        Parameter names (columns)
    title : str, optional
        Custom title
    output_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
    """
    apply_nature_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Heatmap
    im = ax.imshow(S1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    # Annotate
    for i in range(len(scenarios)):
        for j in range(len(parameters)):
            val = S1_matrix[i, j]
            text_color = 'white' if val > 0.25 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=text_color)
    
    ax.set_xticks(range(len(parameters)))
    ax.set_xticklabels(parameters, fontsize=11, rotation=45, ha='right')
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.capitalize() for s in scenarios], fontsize=11)
    
    ax.set_xlabel('Model Parameter', fontsize=13, fontweight='bold')
    ax.set_ylabel('Usage Scenario', fontsize=13, fontweight='bold')
    
    if title is None:
        title = 'Sobol First-Order Sensitivity Indices ($S_1$)\n(Higher = More Influence on TTE Variance)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$S_1$ Index', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Figure 5: Uncertainty Propagation Comparison
# ============================================================================

def plot_uncertainty_comparison(deterministic_tte: Dict,
                                  stochastic_tte: Dict,
                                  scenarios: List[str] = None,
                                  output_path: str = None) -> plt.Figure:
    """
    Side-by-side comparison of deterministic vs stochastic TTE with error bars.
    
    Parameters
    ----------
    deterministic_tte : Dict
        {scenario: TTE_value}
    stochastic_tte : Dict
        {scenario: {'mean': float, 'ci_lower': float, 'ci_upper': float}}
    scenarios : List[str], optional
        Scenario order
    output_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
    """
    apply_nature_style()
    
    if scenarios is None:
        scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Deterministic bars
    det_values = [deterministic_tte.get(s, 0) for s in scenarios]
    bars1 = ax.bar(x - width/2, det_values, width, 
                   label='Deterministic Model', color=NATURE_COLORS['blue'],
                   alpha=0.85, edgecolor='black', linewidth=1)
    
    # Stochastic bars with error bars
    stoch_means = [stochastic_tte.get(s, {}).get('mean', 0) for s in scenarios]
    ci_lower = [stochastic_tte.get(s, {}).get('mean', 0) - stochastic_tte.get(s, {}).get('ci_lower', 0) for s in scenarios]
    ci_upper = [stochastic_tte.get(s, {}).get('ci_upper', 0) - stochastic_tte.get(s, {}).get('mean', 0) for s in scenarios]
    
    bars2 = ax.bar(x + width/2, stoch_means, width,
                   label='Stochastic Model (Mean)', color=NATURE_COLORS['orange'],
                   alpha=0.85, edgecolor='black', linewidth=1)
    
    ax.errorbar(x + width/2, stoch_means, yerr=[ci_lower, ci_upper],
                fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
    
    # Value annotations
    for bar, val in zip(bars1, det_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}h', ha='center', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, stoch_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}h', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], fontsize=12, fontweight='bold')
    ax.set_xlabel('Usage Scenario', fontsize=13, fontweight='bold')
    ax.set_ylabel('Time-to-Empty (hours)', fontsize=13, fontweight='bold')
    ax.set_title('Deterministic vs Stochastic TTE Comparison\n'
                 '(Error bars = 95% CI from OU→TTE Monte Carlo)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Figure 6: Composite Publication Figure (Multi-Panel)
# ============================================================================

def plot_composite_figure(tte_matrix: np.ndarray,
                           trajectory_results: Dict,
                           sobol_matrix: np.ndarray,
                           comparison_data: Dict,
                           scenarios: List[str],
                           soc_levels: List[float],
                           parameters: List[str],
                           output_path: str = None) -> plt.Figure:
    """
    Create a multi-panel composite figure for paper.
    
    Layout:
        [A: TTE Matrix]     [B: SOC Trajectories]
        [C: Sobol Heatmap]  [D: Uncertainty Comparison]
    
    This is the KEY FIGURE for the paper - combines all major results.
    
    Parameters
    ----------
    Various data inputs for each panel.
    output_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
    """
    apply_nature_style()
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)
    
    # ===== Panel A: TTE Matrix =====
    ax_a = fig.add_subplot(gs[0, 0])
    
    cmap = LinearSegmentedColormap.from_list(
        'tte_cmap', [NATURE_COLORS['red'], NATURE_COLORS['yellow'], NATURE_COLORS['green']]
    )
    im_a = ax_a.imshow(tte_matrix, cmap=cmap, aspect='auto')
    
    for i in range(len(soc_levels)):
        for j in range(len(scenarios)):
            val = tte_matrix[i, j]
            color = 'white' if val < 5 else 'black'
            ax_a.text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)
    
    ax_a.set_xticks(range(len(scenarios)))
    ax_a.set_xticklabels([s[:4].capitalize() for s in scenarios])
    ax_a.set_yticks(range(len(soc_levels)))
    ax_a.set_yticklabels([f'{int(s*100)}%' for s in soc_levels])
    ax_a.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Initial SOC', fontsize=11, fontweight='bold')
    ax_a.set_title('(a) TTE Matrix (hours)', fontsize=12, fontweight='bold')
    plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    
    # ===== Panel B: SOC Trajectories =====
    ax_b = fig.add_subplot(gs[0, 1])
    
    for scenario in scenarios:
        if scenario not in trajectory_results:
            continue
        result = trajectory_results[scenario]
        color = SCENARIO_COLORS.get(scenario, 'gray')
        ax_b.plot(result['t_hours'], result['SOC'] * 100, 
                  linewidth=2, color=color, label=scenario.capitalize())
    
    ax_b.axhline(5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_b.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('SOC (%)', fontsize=11, fontweight='bold')
    ax_b.set_title('(b) SOC Discharge Curves', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=9, loc='upper right')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim([0, None])
    ax_b.set_ylim([0, 105])
    
    # ===== Panel C: Sobol Heatmap =====
    ax_c = fig.add_subplot(gs[1, 0])
    
    im_c = ax_c.imshow(sobol_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    for i in range(len(scenarios)):
        for j in range(min(len(parameters), sobol_matrix.shape[1])):
            val = sobol_matrix[i, j]
            color = 'white' if val > 0.25 else 'black'
            ax_c.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=9, fontweight='bold', color=color)
    
    ax_c.set_xticks(range(min(len(parameters), sobol_matrix.shape[1])))
    ax_c.set_xticklabels(parameters[:sobol_matrix.shape[1]], rotation=45, ha='right', fontsize=9)
    ax_c.set_yticks(range(len(scenarios)))
    ax_c.set_yticklabels([s.capitalize() for s in scenarios])
    ax_c.set_xlabel('Parameter', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Scenario', fontsize=11, fontweight='bold')
    ax_c.set_title('(c) Sobol Sensitivity Indices ($S_1$)', fontsize=12, fontweight='bold')
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    
    # ===== Panel D: Uncertainty Comparison =====
    ax_d = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    det_values = comparison_data.get('deterministic', [0] * len(scenarios))
    stoch_values = comparison_data.get('stochastic_mean', [0] * len(scenarios))
    ci_half = comparison_data.get('ci_half', [0] * len(scenarios))
    
    ax_d.bar(x - width/2, det_values, width, label='Deterministic',
             color=NATURE_COLORS['blue'], alpha=0.85, edgecolor='black')
    ax_d.bar(x + width/2, stoch_values, width, label='Stochastic',
             color=NATURE_COLORS['orange'], alpha=0.85, edgecolor='black')
    ax_d.errorbar(x + width/2, stoch_values, yerr=ci_half,
                  fmt='none', color='black', capsize=5, capthick=1.5)
    
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([s[:4].capitalize() for s in scenarios])
    ax_d.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('TTE (hours)', fontsize=11, fontweight='bold')
    ax_d.set_title('(d) Deterministic vs Stochastic TTE', fontsize=12, fontweight='bold')
    ax_d.legend(fontsize=9, loc='upper right')
    ax_d.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('Comprehensive Battery TTE Analysis Results\n'
                 '(Enhanced Physics Model with OU Stochastic Extension)',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Additional O-Award Figures
# ============================================================================

def plot_aging_curve_degradation(cycles: np.ndarray,
                                   capacity_retention: np.ndarray,
                                   fit_params: Dict = None,
                                   output_path: str = None) -> plt.Figure:
    """
    Capacity degradation curve with SEI model fit.
    """
    apply_nature_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.scatter(cycles, capacity_retention * 100, s=50, color=NATURE_COLORS['blue'],
               alpha=0.7, edgecolor='black', label='Measured Data')
    
    if fit_params is not None:
        # SEI model: F(n) = 1 - α√n
        alpha = fit_params.get('alpha', 0.002)
        cycles_fit = np.linspace(0, cycles.max(), 200)
        F_fit = (1 - alpha * np.sqrt(cycles_fit)) * 100
        ax.plot(cycles_fit, F_fit, linewidth=2.5, color=NATURE_COLORS['red'],
                label=f'SEI Model Fit: F = 1 - {alpha:.4f}√n')
    
    ax.set_xlabel('Cycle Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Capacity Retention (%)', fontsize=13, fontweight='bold')
    ax.set_title('Battery Capacity Degradation (SEI Film Growth Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, None])
    ax.set_ylim([60, 105])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


def plot_temperature_efficiency_curve(output_path: str = None) -> plt.Figure:
    """
    Temperature efficiency curve f_T(Θ) showing Arrhenius effect.
    """
    apply_nature_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    T_C = np.linspace(-10, 50, 200)
    T_K = T_C + 273.15
    
    # Arrhenius model
    E_a = 20000  # J/mol
    R = 8.314
    T_ref = 298.15
    
    f_T_arrhenius = np.exp(-(E_a / R) * (1/T_K - 1/T_ref))
    f_T_arrhenius = np.clip(f_T_arrhenius, 0.5, 1.2)
    
    # Piecewise model
    f_T_piecewise = np.ones_like(T_C)
    for i, T in enumerate(T_C):
        if T < 20:
            f_T_piecewise[i] = max(0.7, 1.0 - 0.015 * (20 - T))
        elif T > 30:
            f_T_piecewise[i] = max(0.85, 1.0 - 0.015 * (T - 30))
    
    ax.plot(T_C, f_T_arrhenius, linewidth=2.5, color=NATURE_COLORS['blue'],
            label='Arrhenius Model')
    ax.plot(T_C, f_T_piecewise, linewidth=2.5, color=NATURE_COLORS['orange'],
            linestyle='--', label='Piecewise Model')
    
    # Optimal zone
    ax.axvspan(20, 30, alpha=0.2, color='green', label='Optimal Zone (20-30°C)')
    
    ax.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature Factor $f_T(\\Theta)$', fontsize=13, fontweight='bold')
    ax.set_title('Temperature Efficiency Factor\n'
                 '$f_T(\\Theta) = \\exp(-E_a/R \\cdot (1/\\Theta - 1/\\Theta_{ref}))$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10, 50])
    ax.set_ylim([0.5, 1.3])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


def plot_power_component_radar(component_powers: Dict[str, Dict],
                                output_path: str = None) -> plt.Figure:
    """
    Radar chart comparing power component breakdown across scenarios.
    """
    apply_nature_style()
    
    scenarios = list(component_powers.keys())
    components = ['display', 'cpu', 'gpu', 'network', 'gps', 'bg']
    
    angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for scenario in scenarios:
        if scenario not in component_powers:
            continue
        
        values = [component_powers[scenario].get(c, 0) for c in components]
        values += values[:1]  # Close the plot
        
        color = SCENARIO_COLORS.get(scenario, 'gray')
        ax.plot(angles, values, linewidth=2, color=color, label=scenario.capitalize())
        ax.fill(angles, values, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.upper() for c in components], fontsize=11, fontweight='bold')
    ax.set_title('Power Component Distribution by Scenario\n',
                 fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
    
    return fig


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == '__main__':
    import os
    
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'oaward_figures')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Generating O-Award Publication Figures")
    print("=" * 70)
    
    # Demo data
    scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
    soc_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    # TTE Matrix (demo values)
    tte_matrix = np.array([
        [45.0, 7.8, 4.4, 2.5, 4.3],
        [36.0, 6.2, 3.5, 2.0, 3.4],
        [27.0, 4.7, 2.6, 1.5, 2.6],
        [18.0, 3.1, 1.8, 1.0, 1.7],
        [9.0, 1.6, 0.9, 0.5, 0.9]
    ])
    
    ci_matrix = tte_matrix * 0.1  # ±10% CI for demo
    
    # Generate figures
    print("\n1. TTE Matrix Heatmap...")
    plot_tte_matrix_heatmap(tte_matrix, scenarios, soc_levels, ci_matrix,
                             output_path=os.path.join(OUTPUT_DIR, 'tte_matrix_heatmap.png'))
    
    print("\n2. Temperature Efficiency Curve...")
    plot_temperature_efficiency_curve(
        output_path=os.path.join(OUTPUT_DIR, 'temperature_efficiency.png'))
    
    print("\n3. Sobol Sensitivity Heatmap (demo)...")
    sobol_matrix = np.random.rand(5, 6) * 0.4
    sobol_matrix[3, 1] = 0.45  # CPU high for gaming
    sobol_matrix[3, 2] = 0.35  # GPU high for gaming
    parameters = ['Display', 'CPU', 'GPU', 'Network', 'GPS', 'BG']
    plot_sobol_sensitivity_heatmap(sobol_matrix, scenarios, parameters,
                                    output_path=os.path.join(OUTPUT_DIR, 'sobol_sensitivity.png'))
    
    print("\n4. Power Component Radar (demo)...")
    component_powers = {
        'idle': {'display': 0.0, 'cpu': 0.05, 'gpu': 0.0, 'network': 0.02, 'gps': 0.0, 'bg': 0.08},
        'browsing': {'display': 0.36, 'cpu': 0.23, 'gpu': 0.05, 'network': 0.15, 'gps': 0.0, 'bg': 0.10},
        'video': {'display': 0.64, 'cpu': 0.50, 'gpu': 0.15, 'network': 0.25, 'gps': 0.0, 'bg': 0.10},
        'gaming': {'display': 1.0, 'cpu': 0.70, 'gpu': 0.65, 'network': 0.15, 'gps': 0.0, 'bg': 0.10},
        'navigation': {'display': 0.49, 'cpu': 0.30, 'gpu': 0.10, 'network': 0.35, 'gps': 0.30, 'bg': 0.10}
    }
    plot_power_component_radar(component_powers,
                                output_path=os.path.join(OUTPUT_DIR, 'power_radar.png'))
    
    print("\n" + "=" * 70)
    print("All O-Award Figures Generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
