"""
================================================================================
MCM 2026 Problem A: Data Loading Module
================================================================================

This module implements data loading and preprocessing functionality,
including master table generation via Cartesian product of datasets.

Key Features:
    - Multi-source data fusion: AndroWatts × Mendeley Cartesian Product
    - 36,000 scenario combinations (1000 power profiles × 36 aging states)
    - Data validation and quality checks

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from .decorators import self_healing
from .data_classes import PowerComponents, BatteryState

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and preprocessing module.
    
    Implements data validation, cleaning, and master table generation
    through Cartesian product of AndroWatts × Mendeley data.
    
    Attributes
    ----------
    data_dir : Path
        Path to battery_data directory
    master_table : pd.DataFrame or None
        Master modeling table (36,000 rows)
    battery_states : pd.DataFrame or None
        Battery states table (36 rows)
    aggregated : pd.DataFrame or None
        AndroWatts aggregated power data
    apple_specs : pd.DataFrame or None
        Apple device specifications for validation
    oxford_summary : pd.DataFrame or None
        Oxford battery aging data
    """
    
    def __init__(self, data_dir: str = "battery_data"):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        data_dir : str
            Path to battery_data directory
        """
        self.data_dir = Path(data_dir)
        self.master_table: Optional[pd.DataFrame] = None
        self.battery_states: Optional[pd.DataFrame] = None
        self.aggregated: Optional[pd.DataFrame] = None
        self.apple_specs: Optional[pd.DataFrame] = None
        self.oxford_summary: Optional[pd.DataFrame] = None
        
    @self_healing(max_retries=3)
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets with validation.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all loaded datasets
        """
        logger.info("Loading all datasets...")
        
        # 1. Battery State Table (36 states)
        battery_state_path = self.data_dir / "MCM2026_battery_state_table.csv"
        self.battery_states = pd.read_csv(battery_state_path)
        actual_battery_states = len(self.battery_states)
        logger.info(f"Battery states: {actual_battery_states} states loaded")
        
        # Verify expected count per 数据表关联 Section 2.1
        expected_battery_states = 36
        if actual_battery_states != expected_battery_states:
            logger.warning(f"⚠️ Battery state count mismatch: {actual_battery_states} != {expected_battery_states}")
        
        # 2. AndroWatts Aggregated Power Data (1000 tests)
        aggregated_path = self.data_dir / "open_data" / "material" / "res_test" / "aggregated.csv"
        self.aggregated = pd.read_csv(aggregated_path)
        logger.info(f"AndroWatts: {len(self.aggregated)} power tests loaded")
        
        # 3. Apple Specs (Validation baseline)
        apple_path = self.data_dir / "apple" / "apple_iphone_battery_specs.csv"
        self.apple_specs = pd.read_csv(apple_path)
        logger.info(f"Apple specs: {len(self.apple_specs)} models loaded")
        
        # F5 FIX: Validate Apple specs loaded successfully
        if len(self.apple_specs) == 0:
            logger.error("❌ Apple validation data missing - required for F-level stability")
            raise ValueError("Apple specs validation data is required")
        
        if 'Video_Playback_h' not in self.apple_specs.columns:
            logger.warning("⚠️ Video_Playback_h column missing from Apple specs")
        else:
            valid_video = self.apple_specs['Video_Playback_h'].notna().sum()
            logger.info(f"✅ Apple validation: {valid_video} models with video playback data")
        
        # 4. Oxford Summary (Aging validation)
        oxford_path = self.data_dir / "kaggle" / "oxford" / "oxford_summary.csv"
        self.oxford_summary = pd.read_csv(oxford_path)
        logger.info(f"Oxford data: {len(self.oxford_summary)} cells loaded")
        
        # 5. Generate Master Table (Cartesian Product)
        self._generate_master_table()
        
        return {
            "battery_states": self.battery_states,
            "aggregated": self.aggregated,
            "apple_specs": self.apple_specs,
            "oxford_summary": self.oxford_summary,
            "master_table": self.master_table
        }
    
    def _generate_master_table(self) -> None:
        """
        Generate master modeling table via Cartesian product.
        
        AndroWatts (1000 rows) × Battery States (36 rows) = 36,000 rows
        """
        logger.info("Generating master table via Cartesian product...")
        
        # Preprocess AndroWatts data
        aggregated_processed = self._preprocess_androwatts(self.aggregated)
        
        # Cartesian product
        aggregated_processed['_key'] = 1
        self.battery_states['_key'] = 1
        
        self.master_table = pd.merge(
            aggregated_processed, 
            self.battery_states, 
            on='_key'
        ).drop('_key', axis=1)
        
        # Add derived fields
        self._add_derived_fields()
        
        logger.info(f"Master table generated: {len(self.master_table)} rows × {len(self.master_table.columns)} columns")
        
        # Validation
        assert len(self.master_table) == 36000, f"Expected 36000 rows, got {len(self.master_table)}"
    
    def _preprocess_androwatts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess AndroWatts aggregated.csv data.
        
        Conversions:
        - ID → phone_test_id
        - AVG_SOC_TEMP / 1000 → temp_c (m°C to °C)
        - BATTERY__PERCENT / 100 → soc0
        - BATTERY_DISCHARGE_RATE_UAS × 1e-6 → I_obs_A
        - Sum of *_ENERGY_UW → P_total_uW
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw AndroWatts data
            
        Returns
        -------
        pd.DataFrame
            Preprocessed AndroWatts data
        """
        df = df.copy()
        
        # Rename ID
        df = df.rename(columns={'ID': 'phone_test_id'})
        
        # Temperature conversion (m°C to °C)
        df['temp_c'] = df['AVG_SOC_TEMP'] / 1000.0
        
        # SOC normalization
        df['soc0'] = df['BATTERY__PERCENT'] / 100.0
        
        # Current conversion (µA·s to A, approximation)
        df['I_obs_A'] = df['BATTERY_DISCHARGE_RATE_UAS'].astype(float) * 1e-6
        
        # Calculate total power from _ENERGY_UW columns
        energy_uw_cols = [c for c in df.columns if c.endswith('_ENERGY_UW') and not c.endswith('.1')]
        df['P_total_uW'] = df[energy_uw_cols].sum(axis=1)
        
        # Handle 'err' values in CPU_MID_FREQ_KHz
        if 'CPU_MID_FREQ_KHz' in df.columns:
            df['CPU_MID_FREQ_KHz'] = pd.to_numeric(df['CPU_MID_FREQ_KHz'], errors='coerce')
        
        return df
    
    def _add_derived_fields(self) -> None:
        """
        Add derived fields to master table per Model_Formulas_Paper_Ready.md.
        
        Implements:
        - Section 1.2: Power Decomposition (5 Factors)
        - Section 1.1: Estimated dSOC/dt and TTE
        """
        # Effective charge in Coulombs
        if 'Q_eff_C' not in self.master_table.columns:
            self.master_table['Q_eff_C'] = self.master_table['Q_full_Ah'] * 3600
        
        # =================================================================
        # POWER DECOMPOSITION per Model_Formulas Section 1.2
        # P_total = P_screen + P_proc + P_net + P_GPS + P_bg
        # =================================================================
        
        # Estimate test duration (AndroWatts tests typically 60-180 seconds)
        if 'duration_s_est' not in self.master_table.columns:
            self.master_table['duration_s_est'] = 120.0  # Default 2-minute test
            self.master_table['duration_h_est'] = self.master_table['duration_s_est'] / 3600.0
        
        duration_s = self.master_table['duration_s_est'].replace(0, 120.0)
        
        # Screen Power: P_screen = P_display + P_disp_driver (per Section 1.2)
        self.master_table['P_screen_uW'] = (
            self.master_table.get('Display_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('L22M_DISP_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # CPU Power: P_CPU = P_CPU,big + P_CPU,mid + P_CPU,little (per Section 1.2)
        self.master_table['P_cpu_uW'] = (
            self.master_table.get('CPU_BIG_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('CPU_MID_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('CPU_LITTLE_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # GPU Power: P_GPU = P_GPU,2D + P_GPU,3D (per Section 1.2)
        self.master_table['P_gpu_uW'] = (
            self.master_table.get('GPU_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('GPU3D_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # Processor Power: P_proc = P_CPU + P_GPU
        self.master_table['P_proc_uW'] = (
            self.master_table['P_cpu_uW'] + self.master_table['P_gpu_uW']
        )
        
        # Network Power: P_net = P_cellular + P_WiFi/BT (per Section 1.2)
        self.master_table['P_net_uW'] = (
            self.master_table.get('CELLULAR_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('WLANBT_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # GPS Power: P_GPS (per Section 1.2)
        self.master_table['P_gps_uW'] = (
            self.master_table.get('GPS_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # Background Power: P_bg = P_mem + P_sensor + P_infra (per Section 1.2)
        self.master_table['P_bg_uW'] = (
            self.master_table.get('Memory_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('Sensor_ENERGY_UW', 0).fillna(0) +
            self.master_table.get('INFRASTRUCTURE_ENERGY_UW', 0).fillna(0)
        ) / duration_s
        
        # 5-Factor Total Power per Section 1.2:
        # P_total = P_screen + P_proc + P_net + P_GPS + P_bg
        self.master_table['P_5factor_uW'] = (
            self.master_table['P_screen_uW'] +
            self.master_table['P_proc_uW'] +
            self.master_table['P_net_uW'] +
            self.master_table['P_gps_uW'] +
            self.master_table['P_bg_uW']
        )
        
        # Convert to Watts for model use
        self.master_table['P_screen_W'] = self.master_table['P_screen_uW'] * 1e-6
        self.master_table['P_proc_W'] = self.master_table['P_proc_uW'] * 1e-6
        self.master_table['P_net_W'] = self.master_table['P_net_uW'] * 1e-6
        self.master_table['P_gps_W'] = self.master_table['P_gps_uW'] * 1e-6
        self.master_table['P_bg_W'] = self.master_table['P_bg_uW'] * 1e-6
        self.master_table['P_5factor_W'] = self.master_table['P_5factor_uW'] * 1e-6
        
        # =================================================================
        # SOC DYNAMICS per Model_Formulas Section 1.1
        # =================================================================
        
        # Estimated dSOC/dt under constant current assumption
        self.master_table['dSOC_dt_est_per_s'] = (
            -self.master_table['I_obs_A'].abs() / self.master_table['Q_eff_C']
        )
        
        # Estimated time to empty (hours) per Section 2.1:
        # TTE ≈ (SOC_0 - SOC_th) × Q_eff × V_avg / (P_total × 3600)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.master_table['t_empty_h_est'] = np.where(
                self.master_table['dSOC_dt_est_per_s'] != 0,
                self.master_table['soc0'] / (-self.master_table['dSOC_dt_est_per_s']) / 3600,
                np.inf
            )
        
        # Clean infinite values
        self.master_table['t_empty_h_est'] = self.master_table['t_empty_h_est'].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"✓ Derived fields added: 5-factor power decomposition per Section 1.2")
        logger.info(f"  P_screen range: {self.master_table['P_screen_W'].min():.4f} - {self.master_table['P_screen_W'].max():.4f} W")
        logger.info(f"  P_proc range: {self.master_table['P_proc_W'].min():.4f} - {self.master_table['P_proc_W'].max():.4f} W")
        logger.info(f"  P_net range: {self.master_table['P_net_W'].min():.4f} - {self.master_table['P_net_W'].max():.4f} W")
        logger.info(f"  P_gps range: {self.master_table['P_gps_W'].min():.4f} - {self.master_table['P_gps_W'].max():.4f} W")
        logger.info(f"  P_bg range: {self.master_table['P_bg_W'].min():.4f} - {self.master_table['P_bg_W'].max():.4f} W")
        logger.info(f"  P_5factor_total range: {self.master_table['P_5factor_W'].min():.4f} - {self.master_table['P_5factor_W'].max():.4f} W")
    
    def validate_data_quality(self) -> Dict[str, any]:
        """
        Comprehensive data quality validation.
        
        O-AWARD: Includes P_total > 0 validation to ensure physical validity.
        
        Returns
        -------
        Dict
            Validation report with pass/fail status
        """
        report = {
            "status": "PASS",
            "checks": []
        }
        
        # Check 1: Row count
        if len(self.master_table) != 36000:
            report["checks"].append(("Row count", "FAIL", f"{len(self.master_table)} != 36000"))
            report["status"] = "FAIL"
        else:
            report["checks"].append(("Row count", "PASS", "36000 rows"))
        
        # Check 2: SOH range
        soh_valid = self.master_table['SOH'].between(0.6, 1.01).all()
        report["checks"].append(("SOH range", "PASS" if soh_valid else "FAIL", 
                                f"[{self.master_table['SOH'].min():.2f}, {self.master_table['SOH'].max():.2f}]"))
        
        # Check 3: Q_full_Ah positive
        q_valid = (self.master_table['Q_full_Ah'] > 0).all()
        report["checks"].append(("Q_full_Ah > 0", "PASS" if q_valid else "FAIL", ""))
        
        # Check 4: Temperature range
        temp_valid = self.master_table['temp_c'].between(0, 60).all()
        report["checks"].append(("temp_c range", "PASS" if temp_valid else "WARN", 
                                f"[{self.master_table['temp_c'].min():.1f}, {self.master_table['temp_c'].max():.1f}] °C"))
        
        # Check 5: soc0 range
        soc_valid = self.master_table['soc0'].between(0, 1.01).all()
        report["checks"].append(("soc0 range", "PASS" if soc_valid else "FAIL", 
                                f"[{self.master_table['soc0'].min():.2f}, {self.master_table['soc0'].max():.2f}]"))
        
        # =================================================================
        # O-AWARD CHECK: P_total > 0 validation
        # =================================================================
        # Physical requirement: Total power must be positive for battery drain
        p_total_col = 'P_5factor_W' if 'P_5factor_W' in self.master_table.columns else 'P_total_uW'
        if p_total_col in self.master_table.columns:
            p_total = self.master_table[p_total_col]
            p_positive_count = (p_total > 0).sum()
            p_zero_count = (p_total <= 0).sum()
            p_valid = p_zero_count == 0
            
            report["checks"].append((
                "P_total > 0", 
                "PASS" if p_valid else "WARN",
                f"{p_positive_count} valid, {p_zero_count} invalid (P≤0)"
            ))
            
            if p_zero_count > 0:
                logger.warning(f"⚠️ O-AWARD: {p_zero_count} rows have P_total ≤ 0 (non-physical)")
                logger.warning(f"  These rows represent {p_zero_count/len(self.master_table)*100:.2f}% of data")
                logger.warning(f"  Recommendation: Filter or impute these values before modeling")
                
                # Log details about invalid rows
                invalid_rows = self.master_table[p_total <= 0]
                if len(invalid_rows) > 0:
                    logger.warning(f"  Invalid P_total range: [{p_total[p_total <= 0].min():.4f}, {p_total[p_total <= 0].max():.4f}]")
        else:
            report["checks"].append(("P_total > 0", "SKIP", "P_total column not found"))
        
        # =================================================================
        # O-AWARD CHECK: Extended temperature range warning
        # =================================================================
        temp_col = 'temp_c'
        if temp_col in self.master_table.columns:
            temp_min = self.master_table[temp_col].min()
            temp_max = self.master_table[temp_col].max()
            
            # Warn if temperature range is narrow (less than 20°C spread)
            if temp_max - temp_min < 20:
                logger.warning(f"⚠️ O-AWARD: Temperature range narrow ({temp_min:.1f}-{temp_max:.1f}°C)")
                logger.warning(f"  Recommend testing model with extended range (0-50°C) for robustness")
                report["checks"].append((
                    "temp_range_diversity", 
                    "WARN",
                    f"Narrow range ({temp_max - temp_min:.1f}°C spread)"
                ))
        
        return report
    
    def get_power_components(self, row: pd.Series) -> PowerComponents:
        """
        Extract decomposed power components per Model_Formulas Section 1.2.
        
        Power Decomposition (5 Factors):
        P_total = P_screen + P_proc + P_net + P_GPS + P_bg
        
        Component Formulas:
        | Component | Formula | Data Columns |
        |-----------|---------|----------------------------------------------|
        | Screen    | P_screen = P_display + P_disp_driver | Display + L22M_DISP |
        | Processor | P_proc = P_CPU + P_GPU | CPU_BIG/MID/LITTLE + GPU/GPU3D |
        | Network   | P_net = P_cellular + P_WiFi/BT | CELLULAR + WLANBT |
        | GPS       | P_GPS | GPS_ENERGY_UW |
        | Background| P_bg = P_mem + P_sensor + P_infra | Memory + Sensor + INFRASTRUCTURE |
        
        CRITICAL: AndroWatts *_ENERGY_UW columns are TOTAL ENERGY (µJ),
        not power. Must divide by duration to get actual power (µW).
        
        Power (µW) = Energy (µJ) / duration (s)
        
        Parameters
        ----------
        row : pd.Series
            A row from master_table
            
        Returns
        -------
        PowerComponents
            Decomposed power consumption structure in µW
        """
        # Get duration for energy-to-power conversion
        # duration_s_est is in seconds; if missing, use 120s (typical test duration)
        duration_s = row.get('duration_s_est', 120.0)
        if duration_s <= 0 or pd.isna(duration_s):
            duration_s = 120.0  # Fallback to 2 minutes
        
        # Map AndroWatts ENERGY columns (µJ) to POWER (µW) by dividing by duration
        # Energy (µJ) / duration (s) = Power (µW)
        E_screen = row.get('Display_ENERGY_UW', 0) + row.get('L22M_DISP_ENERGY_UW', 0)
        E_cpu = (row.get('CPU_BIG_ENERGY_UW', 0) + 
                row.get('CPU_MID_ENERGY_UW', 0) + 
                row.get('CPU_LITTLE_ENERGY_UW', 0) +
                row.get('S9M_VDD_CPUCL0_M_ENERGY_UW', 0))
        E_gpu = row.get('GPU_ENERGY_UW', 0) + row.get('GPU3D_ENERGY_UW', 0)
        E_network = (row.get('WLANBT_ENERGY_UW', 0) + 
                    row.get('CELLULAR_ENERGY_UW', 0))
        E_gps = row.get('GPS_ENERGY_UW', 0)
        E_memory = row.get('Memory_ENERGY_UW', 0) + row.get('UFS(Disk)_ENERGY_UW', 0)
        E_sensor = row.get('Sensor_ENERGY_UW', 0)
        E_infrastructure = row.get('INFRASTRUCTURE_ENERGY_UW', 0)
        E_other = row.get('Camera_ENERGY_UW', 0) + row.get('TPU_ENERGY_UW', 0)
        
        # Convert energy to power: P = E / t
        return PowerComponents(
            P_screen=E_screen / duration_s,           # µW
            P_cpu=E_cpu / duration_s,                 # µW
            P_gpu=E_gpu / duration_s,                 # µW
            P_network=E_network / duration_s,         # µW
            P_gps=E_gps / duration_s,                 # µW
            P_memory=E_memory / duration_s,           # µW
            P_sensor=E_sensor / duration_s,           # µW
            P_infrastructure=E_infrastructure / duration_s,  # µW
            P_other=E_other / duration_s              # µW
        )
    
    def get_battery_state(self, row: pd.Series) -> BatteryState:
        """
        Extract battery state from a master table row.
        
        Parameters
        ----------
        row : pd.Series
            A row from master_table
            
        Returns
        -------
        BatteryState
            Battery state parameters
        """
        ocv_coeffs = np.array([
            row['ocv_c0'], row['ocv_c1'], row['ocv_c2'],
            row['ocv_c3'], row['ocv_c4'], row['ocv_c5']
        ])
        
        return BatteryState(
            battery_state_id=row['battery_state_id'],
            Q_full_Ah=row['Q_full_Ah'],
            SOH=row['SOH'],
            Q_eff_C=row['Q_eff_C'],
            ocv_coefficients=ocv_coeffs
        )
    
    def generate_master_table(self) -> pd.DataFrame:
        """
        Public accessor to get master table (generates if not exists).
        
        Returns
        -------
        pd.DataFrame
            Master modeling table (36,000 rows)
        """
        if self.master_table is None:
            self.load_all_data()
        return self.master_table
    
    def load_battery_states(self) -> pd.DataFrame:
        """
        Public accessor to get battery states table.
        
        Returns
        -------
        pd.DataFrame
            Battery states (36 rows)
        """
        if self.battery_states is None:
            battery_state_path = self.data_dir / "MCM2026_battery_state_table.csv"
            self.battery_states = pd.read_csv(battery_state_path)
        return self.battery_states
