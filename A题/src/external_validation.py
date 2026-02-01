"""
================================================================================
MCM 2026 Problem A: External Validation Module (Multi-Dataset Physics Checks)
================================================================================

This module implements **optional external validations** that use additional
battery datasets to strengthen the physical and mechanistic credibility of the
main SOC/TTE model, without changing the existing core pipeline.

Implemented validation layers (strategist 4-layer plan):
    1. XJTU CC/CC-CV experiment (coulomb counting baseline validation)
    2. Nature capacity-fade dataset (capacity vs cycle aging curve)
    3. Oxford energy-trading dataset (profile-dependent aging validation)
    4. NASA cleaned metadata (impedance–SOH linkage via Re, Rct)

All validations:
    - Read data from `battery_data/` (DEFAULT_DATA_DIR)
    - Write traceable summary tables into `output/` (DEFAULT_OUTPUT_DIR)
    - Optionally generate simple figures under `output/figures_external/`

These functions are **side-effect only** (CSV + PNG) and do not alter
core model parameters. They are intended for paper figures and robustness
checks.

Author: MCM Team 2026
License: Open for academic use
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, SEED

logger = logging.getLogger(__name__)


# Set reproducibility and simple plotting style
np.random.seed(SEED)
plt.style.use("seaborn-v0_8-whitegrid")


def _ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    """Ensure base output directories for external validation exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures_external"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"output": output_dir, "figures": fig_dir}


# ==============================================================================
# 1. XJTU COULOMB COUNTING VALIDATION (CC / CC-CV LAB EXPERIMENT)
# ============================================================================

def run_xjtu_coulomb_validation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Validate coulomb counting on XJTU CC/CC-CV discharge experiment.

    Data source (per dataset description):
        Path: 3西安交通大学锂电池充放数据集/
              Data for A data-driven coulomb counting method for state of charge
              calibration and estimation of lithium-ion battery/Experimental data/
              Cell 1/Discharging stage.xlsx

        Columns (in order, no header in raw files):
            1. Voltage (V)
            2. Current (A)
            3. Capacity (Ah)  - cumulative
            4. Energy (Wh)
            5. Time (s)

    We compute a purely data-driven coulomb-counting SOC curve:
        SOC_coulomb(t) = 1 - Capacity_Ah(t) / Capacity_Ah(end)

    and export a traceable CSV + a simple Time–Voltage and Time–SOC plot.

    Returns
    -------
    Optional[pd.DataFrame]
        Processed DataFrame if successful, else None.
    """
    base_data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    base_output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    dirs = _ensure_dirs(base_output_dir)

    xjtu_path = (
        base_data_dir
        / "3西安交通大学锂电池充放数据集"
        / "Data for A data-driven coulomb counting method for state of charge calibration and estimation of lithium-ion battery"
        / "Experimental data"
        / "Cell 1"
        / "Discharging stage.xlsx"
    )

    if not xjtu_path.exists():
        logger.warning(f"XJTU discharging file not found: {xjtu_path}")
        return None

    try:
        df_raw = pd.read_excel(xjtu_path, header=None)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to read XJTU Excel file: {e}")
        return None

    # Assign column names based on dataset documentation
    if df_raw.shape[1] < 5:
        logger.warning(f"Unexpected XJTU column count: {df_raw.shape[1]} (expected >=5)")
        return None

    df = df_raw.iloc[:, :5].copy()
    df.columns = [
        "Voltage_V",
        "Current_A",
        "Capacity_Ah",
        "Energy_Wh",
        "Time_s",
    ]

    df = df.dropna(subset=["Time_s", "Capacity_Ah"]).reset_index(drop=True)
    if df.empty:
        logger.warning("XJTU dataframe empty after dropping NaNs")
        return None

    cap_end = float(df["Capacity_Ah"].max())
    if cap_end <= 0:
        logger.warning(f"Invalid final capacity in XJTU data: {cap_end}")
        return None

    df["SOC_coulomb"] = 1.0 - df["Capacity_Ah"] / cap_end

    # Downsample for plotting to avoid extremely dense figures
    if len(df) > 5000:
        df_plot = df.iloc[:: int(len(df) / 5000), :].copy()
    else:
        df_plot = df.copy()

    # Export CSV for traceability
    out_csv = dirs["output"] / "xjtu_coulomb_validation_cell1.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"XJTU coulomb validation CSV saved to: {out_csv}")

    # Plot Voltage and SOC vs Time
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_plot["Time_s"] / 3600.0, df_plot["Voltage_V"], label="Voltage (V)", color="#2E86AB")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Voltage (V)", color="#2E86AB")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")

    ax2 = ax1.twinx()
    ax2.plot(df_plot["Time_s"] / 3600.0, df_plot["SOC_coulomb"] * 100.0,
             label="SOC (coulomb)", color="#A23B72", alpha=0.8)
    ax2.set_ylabel("SOC (%, coulomb counting)", color="#A23B72")
    ax2.tick_params(axis="y", labelcolor="#A23B72")

    fig.suptitle("XJTU Discharge: Voltage and Coulomb-Counting SOC vs Time", fontsize=13)

    fig.tight_layout()
    fig_path = dirs["figures"] / "xjtu_coulomb_validation_cell1.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"XJTU coulomb validation figure saved to: {fig_path}")

    return df


# ==============================================================================
# 2. NATURE CAPACITY-FADE CURVE (E3 AGING VALIDATION)
# ============================================================================

def run_nature_aging_validation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Validate capacity–cycle aging behaviour using Nature dataset.

    Data source:
        battery_data/5Nature论文电池数据集/Capacity_data_csv/Capacity_data.csv

        Columns:
            Cycle, Capacity_Ah

    We compute normalized SOH = Capacity / Capacity_cycle0 and fit a simple
    linear model over the mid-life regime to estimate an approximate
    fade rate per cycle. Results are exported to CSV and plotted.
    """
    base_data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    base_output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    dirs = _ensure_dirs(base_output_dir)

    nature_path = (
        base_data_dir
        / "5Nature论文电池数据集"
        / "Capacity_data_csv"
        / "Capacity_data.csv"
    )

    if not nature_path.exists():
        logger.warning(f"Nature capacity data not found: {nature_path}")
        return None

    try:
        df = pd.read_csv(nature_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to read Nature capacity CSV: {e}")
        return None

    if {"Cycle", "Capacity_Ah"} - set(df.columns):
        logger.warning(f"Nature CSV missing required columns. Found: {df.columns}")
        return None

    df = df.dropna(subset=["Cycle", "Capacity_Ah"]).reset_index(drop=True)
    if df.empty:
        logger.warning("Nature dataframe empty after dropna")
        return None

    cap0 = float(df["Capacity_Ah"].iloc[0])
    if cap0 <= 0:
        logger.warning(f"Invalid initial capacity in Nature data: {cap0}")
        return None

    df["SOH"] = df["Capacity_Ah"] / cap0

    # Fit a simple linear model on mid-life region (10%–70% of cycle range)
    cycles = df["Cycle"].values.astype(float)
    soh = df["SOH"].values.astype(float)
    n = len(df)
    start_idx = int(0.1 * n)
    end_idx = int(0.7 * n)
    if end_idx <= start_idx + 5:
        start_idx, end_idx = 0, n

    c_mid = cycles[start_idx:end_idx]
    soh_mid = soh[start_idx:end_idx]

    # Simple least-squares fit
    A = np.vstack([c_mid, np.ones_like(c_mid)]).T
    slope, intercept = np.linalg.lstsq(A, soh_mid, rcond=None)[0]

    df["SOH_linear_fit"] = slope * df["Cycle"] + intercept

    # Export CSV
    out_csv = dirs["output"] / "nature_capacity_aging_curve.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Nature aging validation CSV saved to: {out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Cycle"], df["SOH"] * 100.0, label="Observed SOH", color="#2E86AB")
    ax.plot(df["Cycle"], df["SOH_linear_fit"] * 100.0,
            label="Mid-life linear fit", color="#A23B72", linestyle="--")

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH (%)")
    ax.set_title("Nature Dataset: Capacity Fade vs Cycle")
    ax.legend()

    fig.tight_layout()
    fig_path = dirs["figures"] / "nature_capacity_aging_curve.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Nature aging validation figure saved to: {fig_path}")

    logger.info(f"Estimated mid-life fade rate (dSOH/dCycle) ≈ {slope:.4e}")

    return df


# ==============================================================================
# 3. OXFORD ENERGY-TRADING DATASET (PROFILE-DEPENDENT AGING)
# ============================================================================

def run_oxford_profile_aging_validation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Analyze profile-dependent capacity fade using Oxford energy-trading data.

    Data source:
        battery_data/13牛津大学电池驾驶循环数据集/Oxford_csv_clean/
            Oxford_all_profiles_combined.csv

        Columns:
            Profile_Type (BMP/BMR/SPM), Cell_Number, time_s,
            profile_time_s, capacity_Ah

    We aggregate capacity vs profile_time for each profile type and export
    a summary table plus a comparative plot.
    """
    base_data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    base_output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    dirs = _ensure_dirs(base_output_dir)

    oxford_path = (
        base_data_dir
        / "13牛津大学电池驾驶循环数据集"
        / "Oxford_csv_clean"
        / "Oxford_all_profiles_combined.csv"
    )

    if not oxford_path.exists():
        logger.warning(f"Oxford combined profile data not found: {oxford_path}")
        return None

    try:
        df = pd.read_csv(oxford_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to read Oxford combined CSV: {e}")
        return None

    required_cols = {"Profile_Type", "Cell_Number", "profile_time_s", "capacity_Ah"}
    if required_cols - set(df.columns):
        logger.warning(f"Oxford combined CSV missing columns. Found: {df.columns}")
        return None

    df = df.dropna(subset=["Profile_Type", "profile_time_s", "capacity_Ah"]).reset_index(drop=True)
    if df.empty:
        logger.warning("Oxford combined dataframe empty after dropna")
        return None

    # Aggregate average capacity by profile type and profile_time
    grouped = (
        df.groupby(["Profile_Type", "profile_time_s"], as_index=False)["capacity_Ah"]
        .mean()
        .rename(columns={"capacity_Ah": "capacity_Ah_mean"})
    )

    # Normalize capacity by initial value per profile for SOH-style curves
    grouped["SOH"] = 0.0
    for p_type in grouped["Profile_Type"].unique():
        mask = grouped["Profile_Type"] == p_type
        sub = grouped.loc[mask]
        if sub.empty:
            continue
        cap0 = float(sub["capacity_Ah_mean"].iloc[0])
        if cap0 <= 0:
            continue
        grouped.loc[mask, "SOH"] = sub["capacity_Ah_mean"] / cap0

    out_csv = dirs["output"] / "oxford_profile_aging_summary.csv"
    grouped.to_csv(out_csv, index=False)
    logger.info(f"Oxford profile aging summary CSV saved to: {out_csv}")

    # Plot capacity fade per profile type
    fig, ax = plt.subplots(figsize=(8, 5))
    for p_type, sub in grouped.groupby("Profile_Type"):
        ax.plot(
            sub["profile_time_s"] / 3600.0,
            sub["SOH"] * 100.0,
            label=p_type,
        )

    ax.set_xlabel("Profile time (hours)")
    ax.set_ylabel("SOH (%, normalized capacity)")
    ax.set_title("Oxford Energy-Trading: Profile-Dependent Aging")
    ax.legend(title="Profile Type")

    fig.tight_layout()
    fig_path = dirs["figures"] / "oxford_profile_aging_curves.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Oxford profile aging figure saved to: {fig_path}")

    return grouped


# ==============================================================================
# 4. NASA CLEANED METADATA: IMPEDANCE–SOH LINK (Re, Rct)
# ============================================================================

def run_nasa_impedance_validation(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Validate impedance–SOH linkage using NASA cleaned metadata.

    Data source:
        battery_data/kaggle/nasa/cleaned_dataset/metadata.csv

        Columns (subset):
            type ("charge"/"discharge"/"impedance"), battery_id,
            Capacity, Re, Rct

    We:
        - Compute per-battery initial and final discharge capacity → SOH_final
        - Compute mean Re, Rct per battery from impedance records
        - Join into a summary table: battery_id, SOH_final, Re_mean, Rct_mean

    and export both CSV + a simple scatter plot (Re vs SOH, Rct vs SOH).
    """
    base_data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    base_output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    dirs = _ensure_dirs(base_output_dir)

    nasa_meta_path = (
        base_data_dir
        / "kaggle"
        / "nasa"
        / "cleaned_dataset"
        / "metadata.csv"
    )

    if not nasa_meta_path.exists():
        logger.warning(f"NASA metadata not found: {nasa_meta_path}")
        return None

    try:
        meta = pd.read_csv(nasa_meta_path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to read NASA metadata CSV: {e}")
        return None

    # Coerce key columns to numeric for safe arithmetic
    for col in ["Capacity", "Re", "Rct"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    required_cols = {"type", "battery_id", "Capacity", "Re", "Rct"}
    if required_cols - set(meta.columns):
        logger.warning(f"NASA metadata missing required columns. Found: {meta.columns}")
        return None

    # Discharge capacity per battery: initial and final
    dis = meta[(meta["type"] == "discharge") & meta["Capacity"].notna()].copy()
    if dis.empty:
        logger.warning("No discharge rows with capacity in NASA metadata")
        return None

    cap_stats = (
        dis.sort_values(["battery_id", "start_time" if "start_time" in dis.columns else "test_id"])
        .groupby("battery_id")["Capacity"]
        .agg(["first", "last"])
        .rename(columns={"first": "Capacity_initial", "last": "Capacity_final"})
    )

    cap_stats["SOH_final"] = cap_stats["Capacity_final"] / cap_stats["Capacity_initial"]

    # Impedance statistics per battery
    imp = meta[(meta["type"] == "impedance") & (meta["Re"].notna()) & (meta["Rct"].notna())].copy()
    if imp.empty:
        logger.warning("No impedance rows with Re/Rct in NASA metadata")
        return None

    imp_stats = (
        imp.groupby("battery_id")[["Re", "Rct"]]
        .mean()
        .rename(columns={"Re": "Re_mean", "Rct": "Rct_mean"})
    )

    summary = cap_stats.join(imp_stats, how="inner").reset_index()
    if summary.empty:
        logger.warning("NASA impedance–SOH summary empty after join")
        return None

    out_csv = dirs["output"] / "nasa_impedance_soh_summary.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"NASA impedance–SOH summary CSV saved to: {out_csv}")

    # Prepare filtered data for plotting (remove obvious outliers)
    plot_df = summary.copy()
    plot_df = plot_df[
        (plot_df["SOH_final"].between(0.3, 1.1))
        & (plot_df["Re_mean"].between(0.0, 1.0))
        & (plot_df["Rct_mean"].between(0.0, 5.0))
    ]

    if plot_df.empty:
        logger.warning("NASA impedance–SOH plot skipped: no data after filtering")
        return summary

    soh_pct = plot_df["SOH_final"] * 100.0
    re_mohm = plot_df["Re_mean"] * 1000.0
    rct_mohm = plot_df["Rct_mean"] * 1000.0

    # Scatter plots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(soh_pct, re_mohm, label="Re_mean", alpha=0.8)
    ax.set_xlabel("Final SOH (%)")
    ax.set_ylabel("Electrolyte resistance Re (mΩ)")
    ax.set_title("NASA: Electrolyte Resistance vs SOH")

    fig.tight_layout()
    fig_path = dirs["figures"] / "nasa_Re_vs_SOH.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"NASA Re vs SOH figure saved to: {fig_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(soh_pct, rct_mohm, label="Rct_mean", alpha=0.8, color="#A23B72")
    ax.set_xlabel("Final SOH (%)")
    ax.set_ylabel("Charge transfer resistance Rct (mΩ)")
    ax.set_title("NASA: Charge Transfer Resistance vs SOH")

    fig.tight_layout()
    fig_path = dirs["figures"] / "nasa_Rct_vs_SOH.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"NASA Rct vs SOH figure saved to: {fig_path}")

    return summary


# ==============================================================================
# MASTER ENTRY POINT
# ============================================================================

def run_all_external_validations(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all external validations and return a small summary dict.

    This function is safe to call from notebooks or scripts. Failures in any
    individual validation are logged as warnings and do not raise exceptions.
    """
    base_data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    base_output_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)

    results: Dict[str, Any] = {}

    try:
        results["xjtu_coulomb"] = run_xjtu_coulomb_validation(base_data_dir, base_output_dir)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"XJTU coulomb validation failed: {e}")
        results["xjtu_coulomb"] = None

    try:
        results["nature_aging"] = run_nature_aging_validation(base_data_dir, base_output_dir)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Nature aging validation failed: {e}")
        results["nature_aging"] = None

    try:
        results["oxford_profile_aging"] = run_oxford_profile_aging_validation(base_data_dir, base_output_dir)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Oxford profile aging validation failed: {e}")
        results["oxford_profile_aging"] = None

    try:
        results["nasa_impedance"] = run_nasa_impedance_validation(base_data_dir, base_output_dir)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"NASA impedance validation failed: {e}")
        results["nasa_impedance"] = None

    return results
