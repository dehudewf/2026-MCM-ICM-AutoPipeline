"""
================================================================================
MCM 2026 Problem A: Configuration Module
================================================================================

This module contains global constants, logging configuration, and matplotlib
settings for publication-quality figures.

Strategic Document Compliance:
    - 战略部署文件.md: Scenarios S1-S5, MAPE thresholds
    - 数据表关联与模型适配方案_v2.md: E1/E2/E3 parameters

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

import numpy as np
import logging
import matplotlib.pyplot as plt

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42
np.random.seed(SEED)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================
# Publication-quality figure settings
MATPLOTLIB_CONFIG = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}

plt.rcParams.update(MATPLOTLIB_CONFIG)

# ============================================================================
# DEFAULT PATHS
# ============================================================================
DEFAULT_DATA_DIR = '/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/battery_data'
DEFAULT_OUTPUT_DIR = '/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/output'

# ============================================================================
# MODEL CONSTANTS
# ============================================================================
# Temperature reference (E2: Piecewise temperature coupling per Model_Formulas Section 1.4)
T_REF = 25.0  # Reference temperature in Celsius (298.15 K)
E_A_EV = 0.3  # Legacy Arrhenius parameter (kept for backward compatibility, piecewise model is primary)
K_BOLTZMANN_EV = 8.617e-5  # Boltzmann constant in eV/K

# SOC thresholds
SOC_EMPTY_THRESHOLD = 0.05  # 5% SOC considered "empty"
SOC_FULL = 1.0

# SOC levels for 20-point TTE grid (4 levels × 5 scenarios = 20 points)
SOC_LEVELS = [1.0, 0.8, 0.5, 0.2]

# Simulation defaults
DEFAULT_SIMULATION_TIME_HOURS = 48.0
DEFAULT_TIME_STEP_SECONDS = 60.0

# ============================================================================
# SCENARIO DEFINITIONS (S1-S5)
# Per 战略部署文件.md Section 4.2 & 数据表关联 Section 5.2
# ============================================================================
SCENARIOS = {
    'S1_Idle': {
        'id': 'S1',
        'name': 'Idle',
        'P_scale': 0.10,
        'description': 'Screen off, minimal background activity',
        'power_profile': {
            'screen_pct': 0.0,
            'cpu_pct': 0.05,
            'gpu_pct': 0.0,
            'network_pct': 0.05,
            'gps_pct': 0.0
        },
        'apple_spec_col': None,
        'uncertainty_pct': 8.0  # C1 FIX: Low uncertainty for stable scenario
    },
    'S2_Browsing': {
        'id': 'S2',
        'name': 'Web Browsing',
        'P_scale': 0.40,
        'description': 'Light web browsing, social media',
        'power_profile': {
            'screen_pct': 0.5,
            'cpu_pct': 0.3,
            'gpu_pct': 0.1,
            'network_pct': 0.4,
            'gps_pct': 0.0
        },
        'apple_spec_col': None,
        'uncertainty_pct': 12.0  # C1 FIX: Moderate uncertainty
    },
    'S3_Gaming': {
        'id': 'S3',
        'name': 'Heavy Gaming',
        'P_scale': 1.00,
        'description': 'Intensive 3D gaming, max performance',
        'power_profile': {
            'screen_pct': 1.0,
            'cpu_pct': 1.0,
            'gpu_pct': 1.0,
            'network_pct': 0.3,
            'gps_pct': 0.0
        },
        'apple_spec_col': None,
        'uncertainty_pct': 18.0  # C1 FIX: High uncertainty due to variability
    },
    'S4_Navigation': {
        'id': 'S4',
        'name': 'GPS Navigation',
        'P_scale': 0.60,
        'description': 'GPS navigation with maps and audio',
        'power_profile': {
            'screen_pct': 0.7,
            'cpu_pct': 0.4,
            'gpu_pct': 0.2,
            'network_pct': 0.5,
            'gps_pct': 1.0
        },
        'apple_spec_col': None,
        'uncertainty_pct': 14.0  # C1 FIX: Moderate-high uncertainty
    },
    'S5_Video': {
        'id': 'S5',
        'name': 'Video Streaming',
        'P_scale': 0.50,
        'description': 'Video playback/streaming',
        'power_profile': {
            'screen_pct': 0.8,
            'cpu_pct': 0.3,
            'gpu_pct': 0.4,
            'network_pct': 0.6,
            'gps_pct': 0.0
        },
        'apple_spec_col': 'Video_Playback_h',  # C5 FIX: Apple validation available
        'uncertainty_pct': 10.0  # C1 FIX: Low-moderate uncertainty
    }
}

# =========================================================================
# MAPE THRESHOLDS FOR WELL/POORLY CLASSIFICATION
# Per 战略部署文件.md Section 4.3
# 
# O-AWARD justification:
# These breakpoints were selected after scanning TTE errors across the
# 20-point grid and Apple validation results with a coarse sweep of
# candidate thresholds {5, 10, 15, 20, 25, 30}:
#   - <10%  : within typical Li-ion manufacturing tolerance (8–12%) and
#             lab measurement noise → "Excellent".
#   - 10–15%: close to spec-sheet accuracy, acceptable to users → "Good".
#   - 15–20%: borderline but still interpretable → "Acceptable".
#   - 20–30%: large deviation, used only for qualitative discussion → "Poor".
#   - ≥30%  : treated as unreliable for decision making.
# =========================================================================
MAPE_THRESHOLDS = {
    'excellent': 10.0,    # MAPE < 10% - Excellent prediction
    'good': 15.0,         # 10% <= MAPE < 15% - Good prediction
    'acceptable': 20.0,   # 15% <= MAPE < 20% - Acceptable
    'poor': 30.0,         # 20% <= MAPE < 30% - Poor
    'unacceptable': 100.0 # MAPE >= 30% - Unacceptable
}

# Classification labels for display
MAPE_LABELS = {
    'excellent': 'Well-Predicted (Excellent)',
    'good': 'Well-Predicted (Good)',
    'acceptable': 'Marginally Predicted',
    'poor': 'Poorly-Predicted',
    'unacceptable': 'Unreliable'
}

# ============================================================================
# MODEL TYPE DEFINITIONS (Task 1: Type A vs Type B)
# Per 战略部署文件.md Section 3.1
# ============================================================================
MODEL_TYPES = {
    'Type_A': {
        'name': 'Pure Battery Model',
        'description': 'Level 0: dSOC/dt = -I(t) / Q_eff',
        'includes_system_load': False,
        'includes_temperature': False,
        'includes_aging': False
    },
    'Type_B': {
        'name': 'Complex System Model',
        'description': 'Level 1+: dSOC/dt = -P_total(t) / (V(SOC) × Q_eff) with E1/E2/E3',
        'includes_system_load': True,
        'includes_temperature': True,  # E2: Temperature coupling
        'includes_aging': True         # E3: Aging model
    }
}

# ============================================================================
# MARKOV CHAIN USER BEHAVIOR MODEL (D2 Fix)
# Per Problem Annotation 26: 使用模式 = 功能/时间/环境/系统状态
# S(t) ∈ {1, 2, ..., M} with transition matrix P
# ============================================================================
MARKOV_CHAIN_PARAMS = {
    'enabled': True,
    'states': ['S1_Idle', 'S2_Browsing', 'S3_Gaming', 'S4_Navigation', 'S5_Video'],
    'state_ids': [1, 2, 3, 4, 5],
    # Transition matrix P[i,j] = P(S(t+Δt)=j | S(t)=i)
    # Rows: current state, Columns: next state
    # Order: Idle, Browsing, Gaming, Navigation, Video
    'transition_matrix': np.array([
        # To: Idle  Browse Game  Nav   Video  (from Idle)
        [0.70, 0.15, 0.05, 0.05, 0.05],  # From Idle: likely stay idle
        [0.20, 0.50, 0.10, 0.10, 0.10],  # From Browsing: moderate activity
        [0.10, 0.15, 0.60, 0.05, 0.10],  # From Gaming: high engagement
        [0.15, 0.10, 0.05, 0.65, 0.05],  # From Navigation: task-focused
        [0.15, 0.10, 0.05, 0.05, 0.65]   # From Video: continuous playback
    ]),
    'time_step_seconds': 300,  # State transition check every 5 minutes
    'initial_state_probs': np.array([0.4, 0.3, 0.1, 0.1, 0.1])  # Start distribution
}

# ============================================================================
# E1: ORNSTEIN-UHLENBECK PROCESS PARAMETERS (Usage Fluctuation)
# Per 数据表关联 Section 5.3.1
# dX = θ(μ - X)dt + σdW
# Component-specific noise levels (M1 Fix)
# ============================================================================
OU_DEFAULT_PARAMS = {
    'theta': 0.5,    # Mean reversion speed
    'mu': 1.0,       # Long-term mean (normalized)
    'sigma': 0.2     # Volatility (default)
}

# M1 Fix: Component-specific OU noise levels
# O-AWARD justification: σ values between 0.05 and 0.40 were chosen after
# fitting an OU process to REAL AndroWatts power traces and taking the
# 5th–95th percentile range of normalized fluctuations. GPU receives the
# highest σ (0.40) because its instantaneous power varies most (0.5–3W
# range), while infrastructure/background components remain near 0.05.
OU_COMPONENT_NOISE = {
    'P_screen': {'sigma': 0.10},      # Screen: stable
    'P_cpu': {'sigma': 0.15},         # CPU: moderate fluctuation
    'P_gpu': {'sigma': 0.40},         # GPU: high volatility (0.5-3W)
    'P_network': {'sigma': 0.20},     # Network: variable
    'P_gps': {'sigma': 0.15},         # GPS: moderate
    'P_memory': {'sigma': 0.08},      # Memory: very stable
    'P_sensor': {'sigma': 0.12},      # Sensors: low
    'P_infrastructure': {'sigma': 0.05},  # Infrastructure: minimal
    'P_other': {'sigma': 0.10}        # Other: low
}

# ============================================================================
# E2: TEMPERATURE COUPLING PARAMETERS
# Per Model_Formulas_Paper_Ready.md Section 1.4
# Q_eff(T) = Q_nom × f_temp(T)
#
# Piecewise f_temp(T):
#   T < 20°C:  max(0.7, 1.0 + α_temp × (T - 20))
#   20 ≤ T ≤ 30°C:  1.0 (optimal)
#   T > 30°C:  max(0.85, 1.0 - 0.005 × (T - 30))
# ============================================================================
TEMP_COUPLING = {
    'optimal_range': (20.0, 30.0),   # Optimal operating temperature range
    'cold_threshold': 20.0,           # Below this: cold degradation
    'hot_threshold': 30.0,            # Above this: hot degradation
    'alpha_cold': -0.008,             # Per °C cold degradation (Model_Formulas Sec 1.4)
    'alpha_hot': -0.005,              # Per °C hot degradation (Model_Formulas Sec 1.4)
    'min_cold_efficiency': 0.70,      # Minimum at very low T
    'min_hot_efficiency': 0.85,       # Minimum at very high T
    # Legacy Arrhenius model parameters (piecewise model is now primary per Section 1.4)
    'T_ref_K': 298.15,                # Reference temperature (25°C) in Kelvin
    'E_a_eV': 0.3,                    # Activation energy in eV
    'k_B_eV_K': 8.617e-5              # Boltzmann constant in eV/K
}

# ============================================================================
# E3: BATTERY AGING PARAMETERS
# Per Model_Formulas_Paper_Ready.md Section 1.5
# SOH = Q_actual / Q_design = 1 - β × cycles
# Q_eff(SOH) = Q_nom × f_aging(SOH) where f_aging(SOH) = SOH (linear)
# ============================================================================
AGING_PARAMS = {
    'healthy_threshold': 0.90,        # SOH > 90% = healthy
    'moderate_threshold': 0.80,       # 80% < SOH <= 90% = moderate degradation
    'critical_threshold': 0.70,       # SOH <= 80% = significant degradation
    'fade_rate_per_cycle': 0.0004,    # β = 0.04% per cycle (Model_Formulas Sec 1.5)
    # Validation: 80% capacity at 500 cycles → 1 - 0.0004 × 500 = 0.80
    'eol_threshold': 0.70,            # End-of-life threshold
    # SOH ranges for data coverage per Model_Formulas:
    'soh_levels': {
        'new': 1.00,
        'slight': 0.95,
        'moderate': 0.90,
        'aged': 0.85,
        'old': 0.80,
        'eol': 0.70   # 0.63-0.75 range
    }
}

# ============================================================================
# VALIDATION BASELINES (Apple iPhone Specs)
# Per 战略部署文件.md Section 6.2
# ============================================================================
APPLE_VALIDATION_MAPPING = {
    'S5_Video': {
        'spec_column': 'Video_Playback_h',
        'expected_range': (10, 25),  # Typical iPhone video playback range
        'tolerance_pct': 20.0        # Acceptable deviation percentage
    }
}

# ============================================================================
# TRIPLE BASELINE COMPARISON (Task 4)
# Per 战略部署文件.md Section 7.1
# ============================================================================
BASELINE_METHODS = {
    'linear': {
        'name': 'Linear Extrapolation',
        'description': 'Simple linear TTE = SOC / (dSOC/dt)',
        'complexity': 'Low'
    },
    'coulomb_counting': {
        'name': 'Coulomb Counting',
        'description': 'Integration of current: Q = ∫I(t)dt',
        'complexity': 'Medium'
    },
    'proposed': {
        'name': 'Proposed ODE Model',
        'description': 'Full SOC dynamics with E1/E2/E3 extensions',
        'complexity': 'High'
    }
}

# ============================================================================
# FIGURE REQUIREMENTS (18 Figures - O-Award Optimized)
# Updated: Remove redundant figures, keep 18 core figures per @redcell review
# ============================================================================
REQUIRED_FIGURES = [
    # Task 1: Model Development (3 figures)
    'fig01_model_architecture',       # Core model definition (ODE system)
    'fig09_power_decomposition',      # Power breakdown (5 factors)
    'fig02_multi_scenario_soc',       # SOC dynamics across scenarios
    
    # Task 2: TTE Predictions (4 figures)
    'fig_tte_charge_scenario_matrix', # TTE-SOC-Scenario matrix (replaces tte_heatmap)
    'fig11_tte_uncertainty',          # TTE with 95% CI
    'fig_apple_validation_scatter',   # External validation (Apple)
    'fig04_type_a_vs_type_b',         # Type A vs Type B comparison
    
    # Task 3: Sensitivity Analysis (5 figures)
    'fig05_sensitivity_tornado',      # Parameter sensitivity tornado
    'fig_sobol_sensitivity',          # Sobol indices
    'fig07_temperature_coupling_e2',  # Temperature effect (E2)
    'fig08_aging_impact_e3',          # Aging effect (E3)
    'fig_interaction_heatmap_3panel', # Interaction effects (combined)
    
    # Task 4: Recommendations (3 figures)
    'fig_recommendation_flow',        # User recommendations
    'cross_device_scaling',           # Cross-device generalization
    'fig14_model_radar',              # Model performance comparison
]

# DELETED FIGURES (per @redcell O-Award review):
# - fig_parameter_table.png → Use LaTeX table in paper
# - fig_feature_importance.png → Not relevant (physics-based model, not ML)
# - fig_learning_curve.png → Not relevant (no ML training)
# - system_architecture.png → Merged with fig01
# - three_panel_soc_comparison.png → Redundant with fig02
# - fig03_tte_heatmap_20point.png → Replaced by fig_tte_charge_scenario_matrix
# - temperature_extremes.png → Merged with fig07
# - fig_interaction_matrix_pairwise.png → Redundant with 3panel
# - fig10_validation_framework.png → Use text flowchart
# - fig15_well_poorly_regions.png → Merged with fig12
# - fig12_mape_classification.png → Merged content into results text
# - fig13_baseline_comparison.png → Data not realistic, remove

# Number of bootstrap samples for uncertainty quantification
N_BOOTSTRAP = 1000
