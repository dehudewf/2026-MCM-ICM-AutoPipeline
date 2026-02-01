"""
================================================================================
MCM 2026 Problem A: Data Classes Module
================================================================================

This module defines the core data structures used throughout the battery
modeling framework.

Data Classes:
    - PowerComponents: Decomposed power consumption by subsystem
    - BatteryState: Battery state parameters
    - TTEResult: Time-to-Empty prediction result
    - SensitivityResult: Sensitivity analysis results

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerComponents:
    """
    Decomposed power consumption by subsystem (µW).
    
    This class represents the breakdown of total power consumption into
    individual hardware subsystems for fine-grained modeling.
    
    Attributes
    ----------
    P_screen : float
        Display power consumption in µW
    P_cpu : float
        CPU power consumption (all clusters) in µW
    P_gpu : float
        GPU power consumption (including 3D) in µW
    P_network : float
        Network power (WiFi + Cellular + Bluetooth) in µW
    P_gps : float
        GPS power consumption in µW
    P_memory : float
        Memory/Storage power consumption in µW
    P_sensor : float
        Sensor power consumption in µW
    P_infrastructure : float
        Infrastructure overhead power in µW
    P_other : float
        Other components power in µW
        
    Properties
    ----------
    P_total : float
        Total power consumption in µW
    P_total_W : float
        Total power consumption in Watts
    """
    P_screen: float = 0.0      # Display power
    P_cpu: float = 0.0         # CPU power (all clusters)
    P_gpu: float = 0.0         # GPU power (including 3D)
    P_network: float = 0.0     # Network (WiFi + Cellular + BT)
    P_gps: float = 0.0         # GPS power
    P_memory: float = 0.0      # Memory/Storage power
    P_sensor: float = 0.0      # Sensor power
    P_infrastructure: float = 0.0  # Infrastructure overhead
    P_other: float = 0.0       # Other components
    
    @property
    def P_total(self) -> float:
        """Total power consumption in µW."""
        return (self.P_screen + self.P_cpu + self.P_gpu + self.P_network + 
                self.P_gps + self.P_memory + self.P_sensor + 
                self.P_infrastructure + self.P_other)
    
    @property
    def P_total_W(self) -> float:
        """Total power consumption in Watts."""
        return self.P_total * 1e-6


@dataclass
class BatteryState:
    """
    Battery state parameters.
    
    This class encapsulates the physical state and characteristics of a battery,
    including capacity, health, and voltage-SOC relationship.
    
    Per Model_Formulas_Paper_Ready.md:
        - Section 1.3: OCV-SOC polynomial relationship
        - Section 1.4: Temperature coupling f_temp(T)
        - Section 1.5: Aging factor f_aging(SOH)
    
    Attributes
    ----------
    battery_state_id : str
        Unique identifier for the battery state
    Q_full_Ah : float
        Full charge capacity in Amp-hours (Ah)
    SOH : float
        State of Health as a fraction [0, 1]
    Q_eff_C : float
        Effective charge capacity in Coulombs
    ocv_coefficients : np.ndarray
        OCV polynomial coefficients [c0, c1, ..., c5]
        
    Methods
    -------
    OCV(soc)
        Compute Open Circuit Voltage from SOC using polynomial (Section 1.3)
    f_temp(T)
        Piecewise temperature efficiency factor (Section 1.4)
    f_aging()
        Aging efficiency factor (Section 1.5)
    Q_effective(T)
        Combined effective capacity Q_eff(T, SOH) (Section 1.5)
        
    Properties
    ----------
    Q_eff_Ah : float
        Effective capacity in Ah considering SOH
    """
    battery_state_id: str
    Q_full_Ah: float          # Full charge capacity (Ah)
    SOH: float                # State of Health [0, 1]
    Q_eff_C: float            # Effective charge (Coulombs)
    ocv_coefficients: np.ndarray  # OCV polynomial coefficients [c0, c1, ..., c5]
    
    def OCV(self, soc: float) -> float:
        """
        Compute Open Circuit Voltage from SOC using polynomial.
        
        Per Model_Formulas_Paper_Ready.md Section 1.3:
        V_OCV(SOC) = Σ_{k=0}^{5} c_k × SOC^k
        
        Parameters
        ----------
        soc : float
            State of charge as a fraction [0, 1]
            
        Returns
        -------
        float
            Open circuit voltage in Volts
        """
        return np.polyval(self.ocv_coefficients[::-1], soc)
    
    def f_temp(self, T: float) -> float:
        """
        E2: Piecewise temperature efficiency factor.
        
        Per Model_Formulas_Paper_Ready.md Section 1.4:
        f_temp(T) = 
            max(0.7, 1.0 + α_temp × (T - 20))   if T < 20°C
            1.0                                  if 20 ≤ T ≤ 30°C
            max(0.85, 1.0 - 0.005 × (T - 30))   if T > 30°C
        
        where α_temp = -0.008 per °C (cold degradation coefficient)
        
        Parameters
        ----------
        T : float
            Temperature in Celsius
            
        Returns
        -------
        float
            Temperature efficiency factor [0.7, 1.0]
        """
        # Constants per Model_Formulas Section 1.4
        OPTIMAL_LOW = 20.0
        OPTIMAL_HIGH = 30.0
        ALPHA_COLD = -0.008  # per °C
        ALPHA_HOT = -0.005   # per °C
        MIN_COLD_EFF = 0.70
        MIN_HOT_EFF = 0.85
        
        # Optimal range: 20-30°C (no degradation)
        if OPTIMAL_LOW <= T <= OPTIMAL_HIGH:
            return 1.0
        
        # Cold temperature: T < 20°C
        if T < OPTIMAL_LOW:
            f_cold = 1.0 + ALPHA_COLD * (T - 20.0)
            return max(MIN_COLD_EFF, f_cold)
        
        # Hot temperature: T > 30°C
        if T > OPTIMAL_HIGH:
            f_hot = 1.0 + ALPHA_HOT * (T - 30.0)
            return max(MIN_HOT_EFF, f_hot)
        
        return 1.0
    
    def f_aging(self) -> float:
        """
        E3: Aging efficiency factor.
        
        Per Model_Formulas_Paper_Ready.md Section 1.5:
        f_aging(SOH) = SOH (linear assumption)
        
        Physical meaning: SOH directly scales effective capacity.
        - SOH = 1.0 (new): 100% capacity
        - SOH = 0.8 (aged): 80% capacity
        
        Returns
        -------
        float
            Aging factor [0, 1] equal to SOH
        """
        return self.SOH
    
    def Q_effective(self, T: float = 25.0) -> float:
        """
        Compute effective capacity considering temperature and aging.
        
        Per Model_Formulas_Paper_Ready.md Section 1.5 (Combined Effect):
        Q_eff(T, SOH) = Q_nom × f_temp(T) × f_aging(SOH)
        
        Physical Interpretation:
        - New battery (SOH=1.0) at optimal temp (T=25°C): Q_eff = Q_nom
        - Aged battery (SOH=0.8) in cold (T=5°C, f_temp=0.88): 
          Q_eff = Q_nom × 0.88 × 0.8 = 0.704 × Q_nom (30% capacity loss!)
        
        Parameters
        ----------
        T : float
            Temperature in Celsius (default 25°C)
            
        Returns
        -------
        float
            Effective capacity in Coulombs
        """
        Q_nominal = self.Q_full_Ah * 3600  # Ah to Coulombs
        f_t = self.f_temp(T)
        f_a = self.f_aging()
        return Q_nominal * f_t * f_a
    
    @property
    def Q_eff_Ah(self) -> float:
        """Effective capacity in Ah considering SOH."""
        return self.Q_full_Ah * self.SOH


@dataclass
class TTEResult:
    """
    Time-to-Empty prediction result.
    
    This class stores the results of a TTE prediction, including point estimate,
    confidence interval, and the trajectories used in the calculation.
    
    Attributes
    ----------
    tte_hours : float
        Point estimate of time-to-empty in hours
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    soc_trajectory : np.ndarray
        SOC values over time
    time_trajectory : np.ndarray
        Time values in hours
    scenario_id : str
        Identifier for the usage scenario
    initial_soc : float
        Initial state of charge [0, 1]
        
    Properties
    ----------
    ci_width : float
        Width of the confidence interval
    uncertainty_pct : float
        Relative uncertainty as percentage
    """
    tte_hours: float          # Point estimate
    ci_lower: float           # 95% CI lower bound
    ci_upper: float           # 95% CI upper bound
    soc_trajectory: np.ndarray
    time_trajectory: np.ndarray
    scenario_id: str = ""
    initial_soc: float = 1.0
    
    @property
    def ci_width(self) -> float:
        """Width of the confidence interval in hours."""
        return self.ci_upper - self.ci_lower
    
    @property
    def uncertainty_pct(self) -> float:
        """Relative uncertainty as percentage."""
        return (self.ci_width / self.tte_hours) * 100 if self.tte_hours > 0 else np.inf


@dataclass
class SensitivityResult:
    """
    Sensitivity analysis results.
    
    This class stores the results of sensitivity analysis for a single parameter,
    including both local sensitivity and Sobol indices.
    
    Attributes
    ----------
    parameter_name : str
        Name of the analyzed parameter
    sensitivity_index : float
        Local sensitivity ∂TTE/∂param
    sobol_first_order : float
        Sobol first-order index S_i
    sobol_total : float
        Sobol total-order index S_Ti
    baseline_value : float
        Baseline value of the parameter
    impact_category : str
        Impact classification: "high", "medium", or "low"
    """
    parameter_name: str
    sensitivity_index: float  # ∂TTE/∂param
    sobol_first_order: float  # Sobol S_i
    sobol_total: float        # Sobol S_Ti
    baseline_value: float
    impact_category: str      # "high", "medium", "low"


@dataclass
class Scenario:
    """
    Usage scenario definition (S1-S5).
    
    Per 战略部署文件.md Section 4.2:
    5 scenarios × 4 SOC levels = 20-point TTE grid
    
    Attributes
    ----------
    scenario_id : str
        Unique identifier (S1, S2, S3, S4, S5)
    name : str
        Human-readable name
    P_scale : float
        Power scaling factor [0, 1] relative to max power
    description : str
        Detailed description of usage pattern
    power_profile : dict
        Component-wise power percentages
    """
    scenario_id: str
    name: str
    P_scale: float
    description: str
    power_profile: dict = field(default_factory=dict)
    
    @property
    def is_high_power(self) -> bool:
        """Whether this is a high-power scenario (P_scale >= 0.7)."""
        return self.P_scale >= 0.7


@dataclass
class OUParameters:
    """
    Ornstein-Uhlenbeck process parameters (E1: Usage Fluctuation).
    
    Per 数据表关联 Section 5.3.1:
    dX = θ(μ - X)dt + σdW
    
    Attributes
    ----------
    theta : float
        Mean reversion speed (how fast it returns to μ)
    mu : float
        Long-term mean (equilibrium level)
    sigma : float
        Volatility (noise amplitude)
    """
    theta: float  # Mean reversion speed
    mu: float     # Long-term mean
    sigma: float  # Volatility
    
    def half_life(self) -> float:
        """Time for deviation to decay by half: t_1/2 = ln(2) / θ."""
        return np.log(2) / self.theta if self.theta > 0 else np.inf
    
    def stationary_variance(self) -> float:
        """Stationary variance: σ² / (2θ)."""
        return (self.sigma ** 2) / (2 * self.theta) if self.theta > 0 else np.inf


@dataclass
class ValidationResult:
    """
    Model validation result against reference data.
    
    Per 战略部署文件.md Section 6.2 (Apple Specs Validation)
    
    Attributes
    ----------
    scenario_id : str
        Scenario identifier
    device : str
        Reference device name (e.g., "iPhone 15 Pro")
    predicted_tte : float
        Model prediction (hours)
    observed_tte : float
        Reference value (hours)
    mape : float
        Mean Absolute Percentage Error
    classification : str
        Well/Poorly classification
    ci_lower : float
        95% CI lower bound
    ci_upper : float
        95% CI upper bound
    watt_hour : float
        Device battery capacity (Watt-Hours) - for debugging
    battery_mah : float
        Device battery capacity (mAh) - for debugging
    capacity_ratio : float
        Ratio to reference capacity (12 Wh) - for debugging
    """
    scenario_id: str
    device: str
    predicted_tte: float
    observed_tte: float
    mape: float
    classification: str
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    watt_hour: float = None
    battery_mah: float = None
    capacity_ratio: float = None
    
    @property
    def is_well_predicted(self) -> bool:
        """Whether prediction is 'well-predicted' (MAPE < 15%)."""
        return self.mape < 15.0
    
    @property
    def absolute_error(self) -> float:
        """Absolute error in hours."""
        return abs(self.predicted_tte - self.observed_tte)


@dataclass
class ModelComparison:
    """
    Type A vs Type B model comparison result.
    
    Per 战略部署文件.md Section 3.1:
    Type A: Pure Battery Model (Level 0)
    Type B: Complex System Model (Level 1+ with E1/E2/E3)
    
    Attributes
    ----------
    scenario_id : str
        Scenario identifier
    initial_soc : float
        Initial state of charge
    type_a_tte : float
        Type A (Pure Battery) TTE prediction
    type_b_tte : float
        Type B (Complex System) TTE prediction
    delta_hours : float
        Difference (Type B - Type A)
    delta_pct : float
        Percentage difference
    type_a_trajectory : np.ndarray
        SOC trajectory for Type A
    type_b_trajectory : np.ndarray
        SOC trajectory for Type B
    time_trajectory : np.ndarray
        Time array (shared)
    """
    scenario_id: str
    initial_soc: float
    type_a_tte: float
    type_b_tte: float
    delta_hours: float
    delta_pct: float
    type_a_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    type_b_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    time_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def complexity_improvement(self) -> str:
        """Assess if Type B complexity is justified."""
        if abs(self.delta_pct) > 20:
            return "High - Type B significantly different"
        elif abs(self.delta_pct) > 10:
            return "Moderate - Type B provides meaningful refinement"
        else:
            return "Low - Type A may be sufficient"


@dataclass
class TTEGridResult:
    """
    Result for 20-point TTE prediction grid.
    
    Per 战略部署文件.md Section 4.2:
    5 scenarios × 4 SOC levels = 20 grid points
    
    Attributes
    ----------
    scenario_id : str
        Scenario identifier (S1-S5)
    initial_soc : float
        Initial SOC (1.0, 0.8, 0.5, 0.2)
    tte_hours : float
        TTE point estimate
    ci_lower : float
        95% CI lower bound
    ci_upper : float
        95% CI upper bound
    P_total_W : float
        Total power consumption (Watts)
    classification : str
        Well/Poorly classification based on uncertainty
    """
    scenario_id: str
    initial_soc: float
    tte_hours: float
    ci_lower: float
    ci_upper: float
    P_total_W: float
    classification: str = "unknown"
    
    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower
    
    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as percentage."""
        return (self.ci_width / self.tte_hours) * 100 if self.tte_hours > 0 else np.inf


@dataclass
class BaselineComparisonResult:
    """
    Triple baseline comparison result (Task 4).
    
    Per 战略部署文件.md Section 7.1:
    Compare proposed model vs Linear Extrapolation vs Coulomb Counting
    
    Attributes
    ----------
    linear_tte : float
        Linear extrapolation TTE
    coulomb_tte : float
        Coulomb counting TTE
    proposed_tte : float
        Proposed ODE model TTE
    linear_mape : float
        MAPE for linear method
    coulomb_mape : float
        MAPE for coulomb counting
    proposed_mape : float
        MAPE for proposed model
    improvement_vs_linear : float
        Percentage improvement vs linear baseline
    improvement_vs_coulomb : float
        Percentage improvement vs coulomb counting
    scenario_id : str, optional
        Scenario identifier
    initial_soc : float, optional
        Initial SOC
    reference_tte : float, optional
        Reference/ground truth TTE (if available)
    """
    linear_tte: float
    coulomb_tte: float
    proposed_tte: float
    linear_mape: float = 0.0
    coulomb_mape: float = 0.0
    proposed_mape: float = 0.0
    improvement_vs_linear: float = 0.0
    improvement_vs_coulomb: float = 0.0
    scenario_id: str = 'default'
    initial_soc: float = 1.0
    reference_tte: float = 0.0
    
    @property
    def best_method(self) -> str:
        """Return the method with lowest MAPE."""
        mapes = {
            'linear': self.linear_mape,
            'coulomb': self.coulomb_mape,
            'proposed': self.proposed_mape
        }
        return min(mapes, key=mapes.get)
    
    @property
    def proposed_improvement_pct(self) -> float:
        """Improvement of proposed vs best baseline."""
        best_baseline = min(self.linear_mape, self.coulomb_mape)
        if best_baseline > 0:
            return (best_baseline - self.proposed_mape) / best_baseline * 100
        return 0.0
