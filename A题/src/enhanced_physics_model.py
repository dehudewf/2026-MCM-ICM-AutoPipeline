#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MCM 2026 Problem A: Enhanced Physics-Based Battery Model
================================================================================

O-Award Enhancement Module: Integrates three physical sub-models into SOC dynamics:
    - Equation A: Electrochemical Model (voltage, polarization)
    - Equation B: SOC Dynamics Model (coulomb counting with corrections)
    - Equation C: Thermal Model (temperature evolution)

Model Architecture:
    dSOC/dt = -I(t) / (Q(t) × f_T(Θ) × η_eff(I, V_p, Θ))

Where:
    - Q(t): SEI-based capacity fade
    - f_T(Θ): Arrhenius thermal factor
    - η_eff: Polarization-corrected coulombic efficiency

Author: MCM Team 2026
Reference: 战略部署文件.md Section 3.1, Model_Formulas_Paper_Ready.md
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Physical Constants (Paper-Ready)
# ============================================================================
R_GAS = 8.314  # J/(mol·K) - Universal gas constant
E_A_DEFAULT = 20000  # J/mol - Activation energy for Li-ion (typical)
T_REF_K = 298.15  # K - Reference temperature (25°C)

# SEI Growth Parameters (from literature)
ALPHA_SEI = 0.002  # Capacity loss coefficient [Ah^-0.5]
BETA_POLARIZATION = 0.1  # Polarization sensitivity [1/V]


@dataclass
class EnhancedBatteryParams:
    """
    Enhanced battery parameters with three-model coupling.
    
    Attributes per Model_Formulas_Paper_Ready.md Section 1.
    """
    # Nominal parameters
    Q_nom_Ah: float = 2.0       # Nominal capacity [Ah]
    V_nom: float = 3.7          # Nominal voltage [V]
    
    # ECM parameters (Equation A)
    R0: float = 0.05            # Ohmic resistance [Ω]
    R1: float = 0.02            # Polarization resistance [Ω]
    C1: float = 2000            # Polarization capacitance [F]
    
    # Thermal parameters (Equation C)
    m: float = 0.045            # Battery mass [kg]
    cp: float = 1100            # Specific heat [J/(kg·K)]
    h: float = 10               # Convection coefficient [W/(m²·K)]
    A_surf: float = 0.01        # Surface area [m²]
    T_amb_K: float = 298.15     # Ambient temperature [K]
    
    # Aging parameters
    lambda_cal: float = 1e-8    # Calendar aging rate [1/s]
    lambda_cyc: float = 5e-9    # Cycle aging rate [1/(A·s)]
    
    # Efficiency parameters
    eta_0: float = 0.98         # Base coulombic efficiency [-]
    
    # Activation energy for Arrhenius
    E_a: float = E_A_DEFAULT    # Activation energy [J/mol]
    
    @property
    def tau(self) -> float:
        """RC time constant [s]"""
        return self.R1 * self.C1
    
    @property
    def Q_nom_C(self) -> float:
        """Nominal capacity in Coulombs"""
        return self.Q_nom_Ah * 3600


class EnhancedPhysicsBatteryModel:
    """
    Enhanced Physics-Based Battery Model with Three-Model Coupling.
    
    Implements the O-Award enhancement framework per 战略部署文件.md:
    - Equation A: U = V_ocv(SOC,T) - I·R₀(T) - V_p1 - V_p2
    - Equation B: dSOC/dt = -I / Q_max (enhanced with f_T, η_eff)
    - Equation C: dT/dt = (I²R₀(T) - hA(T-T_amb)) / C_th
    
    State Vector: y = [SOC, V_RC, Θ, F, Ah_throughput]
    - SOC: State of Charge [0, 1]
    - V_RC: RC polarization voltage [V]
    - Θ: Temperature [K]
    - F: Capacity retention factor [0, 1]
    - Ah_throughput: Cumulative Ah for SEI model
    """
    
    def __init__(self, params: EnhancedBatteryParams = None, scenario: str = 'video'):
        """
        Initialize enhanced battery model.
        
        Parameters
        ----------
        params : EnhancedBatteryParams
            Battery parameters (uses defaults if None)
        scenario : str
            Usage scenario for power profile
        """
        self.params = params or EnhancedBatteryParams()
        self.scenario = scenario
        
        # Power profiles per scenario [W]
        self.scenario_power = {
            'idle': 0.15,
            'browsing': 0.84,
            'video': 1.49,
            'gaming': 2.60,
            'navigation': 1.54
        }
        self.P_total = self.scenario_power.get(scenario, 1.49)
        
        # Detailed power breakdown for component-level analysis
        self.power_breakdown = {
            'idle': {'display': 0.00, 'cpu': 0.05, 'network': 0.02, 'gps': 0.00, 'bg': 0.08},
            'browsing': {'display': 0.36, 'cpu': 0.23, 'network': 0.15, 'gps': 0.00, 'bg': 0.10},
            'video': {'display': 0.64, 'cpu': 0.50, 'network': 0.25, 'gps': 0.00, 'bg': 0.10},
            'gaming': {'display': 1.00, 'cpu': 1.35, 'network': 0.15, 'gps': 0.00, 'bg': 0.10},
            'navigation': {'display': 0.49, 'cpu': 0.30, 'network': 0.35, 'gps': 0.30, 'bg': 0.10}
        }
        
        logger.debug(f"EnhancedPhysicsBatteryModel initialized: scenario={scenario}, P={self.P_total}W")
    
    # ========================================================================
    # Equation A: Electrochemical Model
    # ========================================================================
    
    def OCV(self, SOC: float, T: float = None) -> float:
        """
        Open Circuit Voltage with optional temperature correction.
        
        OCV(ξ, T) = 3.2 + 0.6ξ + 0.1·exp(10(ξ-0.1)) + α_T·(T - T_ref)
        
        Parameters
        ----------
        SOC : float
            State of charge [0, 1]
        T : float, optional
            Temperature [K]
            
        Returns
        -------
        float
            OCV in Volts
        """
        # Base OCV curve (polynomial + exponential)
        OCV_base = 3.2 + 0.6 * SOC + 0.1 * np.exp(10 * (SOC - 0.1))
        
        # Temperature correction (if provided)
        if T is not None:
            alpha_T = 0.001  # Temperature coefficient [V/K]
            T_ref = T_REF_K
            OCV_base += alpha_T * (T - T_ref)
        
        return OCV_base
    
    def R0_temperature_dependent(self, T: float) -> float:
        """
        Temperature-dependent internal resistance.
        
        R₀(T) = R₀_ref × exp(E_R/R × (1/T - 1/T_ref))
        
        Physical basis: Ion mobility decreases at low temperature.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Internal resistance [Ω]
        """
        E_R = 10000  # Resistance activation energy [J/mol] (lower than E_a)
        R0_ref = self.params.R0
        
        factor = np.exp((E_R / R_GAS) * (1/T - 1/T_REF_K))
        
        # Limit resistance increase to 3x at extreme cold
        return R0_ref * np.clip(factor, 0.8, 3.0)
    
    def compute_current(self, SOC: float, V_RC: float, Theta: float,
                        P_total: float = None) -> float:
        """
        Compute discharge current from power demand.
        
        From P = I × V_terminal, solve for I iteratively:
        I = P / (OCV(SOC) - I×R₀(T) - V_RC)
        
        Fixed-point iteration with damping.
        
        Parameters
        ----------
        SOC : float
            State of charge [0, 1]
        V_RC : float
            RC polarization voltage [V]
        Theta : float
            Temperature [K]
        P_total : float, optional
            Override power demand [W]
            
        Returns
        -------
        float
            Discharge current [A]
        """
        if P_total is None:
            P_total = self.P_total
        
        R0_T = self.R0_temperature_dependent(Theta)
        
        # Fixed-point iteration
        I = 0.5  # Initial guess
        for _ in range(10):
            V_term = self.OCV(SOC, Theta) - I * R0_T - V_RC
            V_term = max(V_term, 2.5)  # Minimum terminal voltage
            
            I_new = P_total / V_term
            
            if abs(I_new - I) < 1e-6:
                break
            
            # Damped update
            I = 0.7 * I + 0.3 * I_new
        
        return np.clip(I, 0, 5)  # Limit to 5A max
    
    # ========================================================================
    # Equation B: Enhanced SOC Dynamics
    # ========================================================================
    
    def f_temp_arrhenius(self, T: float) -> float:
        """
        Arrhenius temperature factor for capacity.
        
        f_T(Θ) = exp(-E_a/R × (1/Θ - 1/Θ_ref))
        
        Physical interpretation:
        - T > T_ref: f_T > 1 (faster kinetics, but may accelerate aging)
        - T < T_ref: f_T < 1 (slower kinetics, reduced effective capacity)
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Temperature factor [0.5, 1.2]
        """
        E_a = self.params.E_a
        
        exponent = -(E_a / R_GAS) * (1/T - 1/T_REF_K)
        f_T = np.exp(exponent)
        
        # Physically bounded
        return np.clip(f_T, 0.5, 1.2)
    
    def f_temp_piecewise(self, T_celsius: float) -> float:
        """
        Piecewise temperature efficiency factor (alternative formulation).
        
        Per Model_Formulas_Paper_Ready.md Section 1.4:
        f_temp(T) = 
            max(0.7, 1.0 + α_temp × (T - 20))   if T < 20°C
            1.0                                  if 20 ≤ T ≤ 30°C
            max(0.85, 1.0 - 0.005 × (T - 30))   if T > 30°C
        
        Parameters
        ----------
        T_celsius : float
            Temperature [°C]
            
        Returns
        -------
        float
            Temperature efficiency factor [0.7, 1.0]
        """
        if 20 <= T_celsius <= 30:
            return 1.0
        elif T_celsius < 20:
            alpha_cold = -0.015  # Enhanced cold penalty (3x original)
            f_cold = 1.0 + alpha_cold * (T_celsius - 20)
            return max(0.7, f_cold)
        else:  # T > 30
            f_hot = 1.0 - 0.015 * (T_celsius - 30)  # Enhanced hot penalty
            return max(0.85, f_hot)
    
    def eta_effective(self, I: float, V_RC: float, Theta: float) -> float:
        """
        Enhanced coulombic efficiency with electrochemical-thermal coupling.
        
        η_eff = η₀ × exp(-β × |V_p| / (T/T_ref))
        
        Physical basis: Polarization voltage (activation overpotential)
        increases irreversible losses. Effect is amplified at low temperature.
        
        Parameters
        ----------
        I : float
            Discharge current [A]
        V_RC : float
            RC polarization voltage [V]
        Theta : float
            Temperature [K]
            
        Returns
        -------
        float
            Effective coulombic efficiency [0.85, 1.0]
        """
        eta_0 = self.params.eta_0
        beta = BETA_POLARIZATION
        
        # Activation overpotential approximated by RC voltage
        eta_act = abs(V_RC)
        
        # Temperature-modulated polarization effect
        T_ratio = Theta / T_REF_K
        correction = np.exp(-beta * eta_act / T_ratio)
        
        eta_eff = eta_0 * correction
        
        return np.clip(eta_eff, 0.85, 1.0)
    
    def Q_effective_sei(self, F: float, Ah_throughput: float, Theta: float) -> float:
        """
        SEI-based effective capacity with temperature and aging coupling.
        
        Q_eff(t) = Q_nom × F × f_T(Θ) - α_SEI × √(Ah_throughput)
        
        Parameters
        ----------
        F : float
            Capacity retention factor [0, 1]
        Ah_throughput : float
            Cumulative Ah throughput [Ah]
        Theta : float
            Temperature [K]
            
        Returns
        -------
        float
            Effective capacity [C]
        """
        Q_nom_C = self.params.Q_nom_C
        f_T = self.f_temp_arrhenius(Theta)
        
        # Base capacity with calendar aging
        Q_base = Q_nom_C * F * f_T
        
        # SEI growth contribution (cycle aging)
        Q_sei_loss = ALPHA_SEI * np.sqrt(max(Ah_throughput, 0)) * 3600  # Convert to C
        
        return max(Q_base - Q_sei_loss, 0.3 * Q_nom_C)  # Floor at 30% capacity
    
    # ========================================================================
    # Equation C: Thermal Model
    # ========================================================================
    
    def dTheta_dt(self, I: float, Theta: float) -> float:
        """
        Temperature dynamics per Equation C.
        
        C_th × dΘ/dt = I²R₀(Θ) - hA(Θ - Θ_amb)
        
        where C_th = m × c_p (thermal capacitance)
        
        Parameters
        ----------
        I : float
            Discharge current [A]
        Theta : float
            Temperature [K]
            
        Returns
        -------
        float
            Temperature rate of change [K/s]
        """
        R0_T = self.R0_temperature_dependent(Theta)
        
        # Joule heating
        Q_heat = I**2 * R0_T
        
        # Convective cooling
        Q_loss = self.params.h * self.params.A_surf * (Theta - self.params.T_amb_K)
        
        # Thermal capacitance
        C_th = self.params.m * self.params.cp
        
        return (Q_heat - Q_loss) / C_th
    
    # ========================================================================
    # Enhanced ODE System (5-state)
    # ========================================================================
    
    def ode_system_enhanced(self, t: float, y: np.ndarray,
                             P_trajectory: np.ndarray = None,
                             t_trajectory: np.ndarray = None) -> list:
        """
        Enhanced 5-state ODE system with three-model coupling.
        
        State vector: y = [SOC, V_RC, Θ, F, Ah_throughput]
        
        Equations:
        1. dSOC/dt = -I / (Q_eff(Θ, F, Ah) × η_eff(I, V_RC, Θ))
        2. dV_RC/dt = (I×R₁ - V_RC) / τ
        3. dΘ/dt = (I²R₀(Θ) - hA(Θ-Θ_amb)) / C_th
        4. dF/dt = -(λ_cal + λ_cyc × √I)
        5. dAh/dt = I / 3600
        
        Parameters
        ----------
        t : float
            Time [s]
        y : np.ndarray
            State vector
        P_trajectory : np.ndarray, optional
            Time-varying power for MC simulation
        t_trajectory : np.ndarray, optional
            Time points for P_trajectory interpolation
            
        Returns
        -------
        list
            State derivatives
        """
        SOC, V_RC, Theta, F, Ah_throughput = y
        
        # Bound states
        SOC = np.clip(SOC, 0.01, 1.0)
        F = np.clip(F, 0.6, 1.0)
        Theta = np.clip(Theta, 253.15, 333.15)  # -20°C to 60°C
        
        # Get power (constant or time-varying)
        if P_trajectory is not None and t_trajectory is not None:
            P_total = np.interp(t, t_trajectory, P_trajectory)
        else:
            P_total = self.P_total
        
        # Compute current
        I = self.compute_current(SOC, V_RC, Theta, P_total)
        
        # Effective capacity with all corrections
        Q_eff = self.Q_effective_sei(F, Ah_throughput, Theta)
        
        # Effective efficiency with polarization correction
        eta_eff = self.eta_effective(I, V_RC, Theta)
        
        # 1. SOC dynamics (enhanced)
        dSOC_dt = -I / (Q_eff * eta_eff) if Q_eff > 0 else 0
        
        # 2. RC polarization dynamics
        dV_RC_dt = (I * self.params.R1 - V_RC) / self.params.tau
        
        # 3. Thermal dynamics
        dTheta_dt_val = self.dTheta_dt(I, Theta)
        
        # 4. Capacity fade (enhanced: √I for SEI kinetics)
        dF_dt = -(self.params.lambda_cal + self.params.lambda_cyc * np.sqrt(abs(I) + 0.01))
        
        # 5. Ah throughput accumulation
        dAh_dt = abs(I) / 3600  # [Ah/s]
        
        return [dSOC_dt, dV_RC_dt, dTheta_dt_val, dF_dt, dAh_dt]
    
    def event_soc_threshold(self, t: float, y: np.ndarray) -> float:
        """Event function: Stop when SOC reaches 5%."""
        return y[0] - 0.05
    
    event_soc_threshold.terminal = True
    event_soc_threshold.direction = -1
    
    # ========================================================================
    # Simulation Interface
    # ========================================================================
    
    def simulate(self, SOC0: float = 1.0, t_max_hours: float = 50,
                 P_trajectory: np.ndarray = None,
                 t_trajectory: np.ndarray = None) -> Dict:
        """
        Run enhanced simulation.
        
        Parameters
        ----------
        SOC0 : float
            Initial SOC [0, 1]
        t_max_hours : float
            Maximum simulation time [hours]
        P_trajectory : np.ndarray, optional
            Time-varying power for MC simulation [W]
        t_trajectory : np.ndarray, optional
            Time points for P_trajectory [s]
            
        Returns
        -------
        Dict
            Simulation results including all states and TTE
        """
        t_max_s = t_max_hours * 3600
        
        # Initial state: [SOC, V_RC, Θ, F, Ah_throughput]
        y0 = [SOC0, 0.0, self.params.T_amb_K, 1.0, 0.0]
        
        # ODE wrapper for time-varying power
        def ode_func(t, y):
            return self.ode_system_enhanced(t, y, P_trajectory, t_trajectory)
        
        # Solve with event detection
        sol = solve_ivp(
            ode_func,
            t_span=(0, t_max_s),
            y0=y0,
            method='LSODA',
            events=self.event_soc_threshold,
            dense_output=True,
            max_step=60
        )
        
        # Compute TTE
        if sol.t_events[0].size > 0:
            tte_hours = sol.t_events[0][0] / 3600
        else:
            tte_hours = sol.t[-1] / 3600
        
        return {
            't_hours': sol.t / 3600,
            't_seconds': sol.t,
            'SOC': sol.y[0],
            'V_RC': sol.y[1],
            'Theta_K': sol.y[2],
            'Theta_C': sol.y[2] - 273.15,
            'F': sol.y[3],
            'Ah_throughput': sol.y[4],
            'TTE_hours': tte_hours,
            'status': sol.status,
            'message': sol.message
        }
    
    def compute_tte(self, SOC0: float = 1.0, t_max_hours: float = 50) -> float:
        """
        Compute Time-to-Empty.
        
        Parameters
        ----------
        SOC0 : float
            Initial SOC
        t_max_hours : float
            Maximum simulation time
            
        Returns
        -------
        float
            TTE in hours
        """
        result = self.simulate(SOC0, t_max_hours)
        return result['TTE_hours']


class EnhancedBatterySimulator:
    """
    High-level simulator wrapper for batch operations.
    """
    
    def __init__(self, params: EnhancedBatteryParams = None):
        self.params = params or EnhancedBatteryParams()
    
    def run_scenario_matrix(self, scenarios: List[str] = None,
                            soc_levels: List[float] = None) -> Dict:
        """
        Run 25-scenario matrix (5 scenarios × 5 SOC levels).
        
        Returns results matrix compatible with existing visualization.
        """
        if scenarios is None:
            scenarios = ['idle', 'browsing', 'video', 'gaming', 'navigation']
        if soc_levels is None:
            soc_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        results_matrix = np.zeros((len(soc_levels), len(scenarios)))
        full_results = {}
        
        for i, soc0 in enumerate(soc_levels):
            for j, scenario in enumerate(scenarios):
                model = EnhancedPhysicsBatteryModel(self.params, scenario)
                result = model.simulate(SOC0=soc0)
                
                results_matrix[i, j] = result['TTE_hours']
                full_results[f'{scenario}_{int(soc0*100)}'] = result
        
        return {
            'matrix': results_matrix,
            'scenarios': scenarios,
            'soc_levels': soc_levels,
            'full_results': full_results
        }


# ============================================================================
# Paper-Ready Formula Documentation
# ============================================================================
PAPER_FORMULAS = """
## Enhanced SOC Model Formulas (Paper Section 2.3)

### Equation A: Electrochemical Model (Voltage)
$$U = V_{ocv}(\\xi, T) - I \\cdot R_0(T) - V_{p1} - V_{p2}$$

where:
- $V_{ocv}(\\xi) = 3.2 + 0.6\\xi + 0.1 \\exp(10(\\xi - 0.1))$
- $R_0(T) = R_{0,ref} \\exp\\left(\\frac{E_R}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)\\right)$

### Equation B: Enhanced SOC Dynamics
$$\\frac{d\\xi}{dt} = -\\frac{I(t)}{Q_{eff}(T, F, Ah) \\cdot \\eta_{eff}(I, V_p, T)}$$

where:
- $Q_{eff} = Q_{nom} \\cdot F \\cdot f_T(\\Theta) - \\alpha_{SEI} \\sqrt{Ah_{throughput}}$
- $f_T(\\Theta) = \\exp\\left(-\\frac{E_a}{R}\\left(\\frac{1}{\\Theta} - \\frac{1}{\\Theta_{ref}}\\right)\\right)$
- $\\eta_{eff} = \\eta_0 \\cdot \\exp\\left(-\\frac{\\beta |V_p|}{T/T_{ref}}\\right)$

### Equation C: Thermal Model
$$C_{th} \\frac{d\\Theta}{dt} = I^2 R_0(\\Theta) - hA(\\Theta - \\Theta_{amb})$$

### Capacity Fade (SEI Model)
$$\\frac{dF}{dt} = -\\lambda_{cal} - \\lambda_{cyc} \\sqrt{I}$$
"""


if __name__ == '__main__':
    # Test the enhanced model
    print("=" * 70)
    print("Testing Enhanced Physics-Based Battery Model")
    print("=" * 70)
    
    model = EnhancedPhysicsBatteryModel(scenario='video')
    result = model.simulate(SOC0=1.0)
    
    print(f"\nScenario: video (P={model.P_total}W)")
    print(f"Initial SOC: 100%")
    print(f"Time-to-Empty: {result['TTE_hours']:.2f} hours")
    print(f"Final Temperature: {result['Theta_C'][-1]:.2f}°C")
    print(f"Final Capacity Retention: {result['F'][-1]*100:.2f}%")
    print(f"Total Ah Throughput: {result['Ah_throughput'][-1]:.3f} Ah")
    
    # Run full matrix
    print("\n" + "=" * 70)
    print("Running 25-Scenario Matrix")
    print("=" * 70)
    
    simulator = EnhancedBatterySimulator()
    matrix_results = simulator.run_scenario_matrix()
    
    print("\nTTE Matrix (hours):")
    print("              Idle    Browsing   Video    Gaming   Navigation")
    for i, soc in enumerate([100, 80, 60, 40, 20]):
        row = matrix_results['matrix'][i]
        print(f"SOC₀={soc}%    {row[0]:7.2f}  {row[1]:7.2f}  {row[2]:7.2f}  {row[3]:7.2f}  {row[4]:7.2f}")
    
    print("\n" + PAPER_FORMULAS)
