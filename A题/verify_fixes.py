#!/usr/bin/env python3
"""
Quick verification script for all fixes.
Tests that all new methods can be imported and called.
"""

import sys
sys.path.insert(0, '/Users/xiaohuiwei/Downloads/肖惠威美赛/A题')

from src.sensitivity import SensitivityAnalyzer
from src.recommendations import RecommendationEngine
from src.data_classes import PowerComponents, BatteryState
import pandas as pd

print("=" * 70)
print("COMPREHENSIVE FIXES VERIFICATION")
print("=" * 70)

# Test F1: OU simulation
print("\n[F1] Testing OU fluctuation simulation...")
try:
    battery = BatteryState(battery_state_id="test", Q_nominal=3200, SOH=1.0)
    power = PowerComponents(
        P_screen=0.157, P_cpu=0.089, P_gpu=0.012, P_network=0.045,
        P_gps=0.089, P_memory=0.025, P_sensor=0.005, 
        P_infrastructure=0.013, P_other=0.019
    )
    from src.soc_model import SOCDynamicsModel
    model = SOCDynamicsModel(battery, power, temperature=25.0)
    analyzer = SensitivityAnalyzer(model)
    result = analyzer.simulate_fluctuations_with_ou(baseline_power=power)
    print(f"  ✓ OU simulation successful: TTE_mean={result['tte_mean_h']:.2f}h, CI_width={result['ci_width_h']:.2f}h")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test F2: Largest ranking
print("\n[F2] Testing 'largest' ranking in output...")
try:
    df = pd.read_csv('/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/output/greatest_reduction_activities.csv')
    if 'Rank' in df.columns and 'Rank_Explanation' in df.columns:
        print(f"  ✓ Rank columns present: {df.columns.tolist()}")
        print(f"  ✓ Top entry: Rank={df.iloc[0]['Rank']}, {df.iloc[0]['Rank_Explanation'][:50]}...")
    else:
        print(f"  ✗ FAILED: Missing Rank columns")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test F3: OS comparison
print("\n[F3] Testing OS power saver comparison...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    user_recs = rec_engine.generate_user_recommendations()
    os_comp = rec_engine.compare_with_os_power_saver(user_recs)
    print(f"  ✓ OS comparison generated: {len(os_comp)} rows")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test F4: 5-question practicality
print("\n[F4] Testing 5-question practicality score...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    rec_dict = {
        'recommendation_id': 'TEST',
        'action': 'Lower brightness to 30%',
        'category': 'display'
    }
    score, details = rec_engine.compute_5q_practicality_score(rec_dict)
    print(f"  ✓ 5Q score computed: {score}/10, Questions: {list(details.keys())}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test C5: Apple Watch example
print("\n[C5] Testing Apple Watch concrete example...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    apple_result = rec_engine.compute_apple_watch_tte_example()
    print(f"  ✓ Apple Watch TTE: {apple_result['TTE_predicted_h']:.2f}h (error: {apple_result['validation']['error_pct']:.1f}%)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test M1: Source traceability
print("\n[M1] Testing source traceability...")
try:
    df = pd.read_csv('/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/output/surprisingly_little_factors.csv')
    if 'Source_Traceability' in df.columns:
        print(f"  ✓ Source column present: {df.columns.tolist()}")
        print(f"  ✓ Sample source: {df.iloc[0]['Source_Traceability'][:60]}...")
    else:
        print(f"  ✗ FAILED: Missing Source column")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test M2: predict_tte_with_policy
print("\n[M2] Testing predict_tte_with_policy...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    result = rec_engine.predict_tte_with_policy(
        current_soc=0.50,
        current_temp=25.0,
        usage_context='normal'
    )
    print(f"  ✓ Policy prediction: {result['estimated_tte_h']:.2f}h with {result['policy']['mode']} mode")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test M3: Combined Top-3
print("\n[M3] Testing combined Top-3 effect...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    user_recs = rec_engine.generate_user_recommendations()
    combined = rec_engine.compute_combined_top3_effect(user_recs)
    print(f"  ✓ Combined effect: {combined['actual_combined_gain_h']:.2f}h (synergy: {combined['synergy_factor']:.2f})")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test M4: SOH-TTE linkage
print("\n[M4] Testing SOH-TTE linkage...")
try:
    rec_engine = RecommendationEngine(baseline_tte=8.0)
    soh_linkage = rec_engine.compute_soh_tte_linkage(soh_levels=[1.0, 0.9, 0.8, 0.7])
    print(f"  ✓ SOH linkage computed: {len(soh_linkage)} aging levels")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
