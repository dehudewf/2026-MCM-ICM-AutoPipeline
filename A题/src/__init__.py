"""
================================================================================
MCM 2026 Problem A: Smartphone Battery Modeling - O-Award Framework
================================================================================

This package implements a complete end-to-end modeling pipeline for smartphone
battery state-of-charge (SOC) dynamics and time-to-empty (TTE) prediction.

Modules:
    - config: Constants, logging, and matplotlib configuration
    - decorators: Self-healing error handler decorator
    - data_classes: Core data structures (PowerComponents, BatteryState, etc.)
    - data_loader: Data loading and preprocessing (DataLoader)
    - soc_model: SOC dynamics model (Task 1)
    - tte_predictor: TTE prediction with uncertainty (Task 2)
    - sensitivity: Sensitivity analysis (Task 3)
    - recommendations: Model-based recommendations (Task 4)
    - visualizer: Publication-quality visualizations
    - pipeline: End-to-end pipeline orchestrator

Key Features:
    - Continuous-time ODE model for SOC dynamics (Task 1)
    - Multi-scenario TTE predictions with uncertainty quantification (Task 2)
    - Comprehensive sensitivity analysis with Sobol indices (Task 3)
    - Model-based recommendations with traceability (Task 4)

O-Award Compliance:
    - Self-healing error handling ✓
    - Reproducibility: SEED=42 ✓
    - SHAP-based explainability ✓
    - Bootstrap confidence intervals ✓

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

# Configuration and constants
from .config import (
    SEED,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    T_REF,
    SOC_EMPTY_THRESHOLD,
    SOC_FULL,
    DEFAULT_SIMULATION_TIME_HOURS,
    DEFAULT_TIME_STEP_SECONDS,
    logger
)

# Decorators
from .decorators import self_healing

# Data classes
from .data_classes import (
    PowerComponents,
    BatteryState,
    TTEResult,
    SensitivityResult
)

# Core modules
from .data_loader import DataLoader
from .soc_model import SOCDynamicsModel
from .tte_predictor import TTEPredictor
from .sensitivity import SensitivityAnalyzer
from .recommendations import RecommendationEngine
from .visualizer import Visualizer
from .pipeline import MCMBatteryPipeline, main

# Version
__version__ = '1.0.0'

# Public API
__all__ = [
    # Config
    'SEED',
    'DEFAULT_DATA_DIR',
    'DEFAULT_OUTPUT_DIR',
    'T_REF',
    'SOC_EMPTY_THRESHOLD',
    'SOC_FULL',
    'DEFAULT_SIMULATION_TIME_HOURS',
    'DEFAULT_TIME_STEP_SECONDS',
    'logger',
    # Decorators
    'self_healing',
    # Data classes
    'PowerComponents',
    'BatteryState',
    'TTEResult',
    'SensitivityResult',
    # Core modules
    'DataLoader',
    'SOCDynamicsModel',
    'TTEPredictor',
    'SensitivityAnalyzer',
    'RecommendationEngine',
    'Visualizer',
    'MCMBatteryPipeline',
    'main',
]
