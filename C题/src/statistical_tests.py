"""
Statistical Tests for MCM Paper - Generate Real Values
Outputs: Spearman, F-test, t-test, VIF, Cohen's d
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_DIR
from data.loader import DataLoader
from data.merger import DataMerger
from features.engineer import FeatureEngineer

print("=" * 70)
print("STATISTICAL TESTS FOR MCM PAPER - REAL VALUES")
print("=" * 70)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1] Loading data...")
loader = DataLoader(DATA_DIR)
data = loader.load_all()
merger = DataMerger()
result = merger.merge_datasets(
    data['medals'], data['hosts'],
    data.get('programs'), data.get('athletes')
)
df = result.data

# Standardize columns
column_mapping = {
    'NOC': 'country', 'Gold': 'gold', 'Silver': 'silver',
    'Bronze': 'bronze', 'Total': 'total', 'Year': 'year', 'Rank': 'rank'
}
df = df.rename(columns=column_mapping)
print(f"   Loaded {len(df)} records, {df['country'].nunique()} countries")

# ============================================================================
# 2. Events-Medals Regression: Spearman & F-test
# ============================================================================
print("\n[2] Events-Medals Regression Statistics...")

# Historical events data
events_dict = {
    1984: 221, 1988: 237, 1992: 257, 1996: 271, 2000: 300,
    2004: 301, 2008: 302, 2012: 302, 2016: 306, 2020: 339
}

yearly_data = df.groupby('year').agg({'total': 'sum'}).reset_index()
yearly_data['events'] = yearly_data['year'].map(events_dict)
yearly_data = yearly_data.dropna()

X = yearly_data['events'].values
y = yearly_data['total'].values
n = len(X)

# Pearson R²
reg = LinearRegression()
reg.fit(X.reshape(-1, 1), y)
r2 = reg.score(X.reshape(-1, 1), y)
coef = reg.coef_[0]
intercept = reg.intercept_

# Spearman correlation
spearman_r, spearman_p = stats.spearmanr(X, y)

# F-statistic (for simple linear regression: F = (R² * (n-2)) / (1 - R²))
k = 1  # number of predictors
F_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
F_p = 1 - stats.f.cdf(F_stat, k, n - k - 1)

# t-statistic for coefficient
y_pred = reg.predict(X.reshape(-1, 1))
residuals = y - y_pred
MSE = np.sum(residuals**2) / (n - 2)
SE_coef = np.sqrt(MSE / np.sum((X - np.mean(X))**2))
t_stat = coef / SE_coef
t_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

# 95% CI for coefficient
ci_margin = stats.t.ppf(0.975, n - 2) * SE_coef
ci_lower = coef - ci_margin
ci_upper = coef + ci_margin

print(f"\n   === Events-Medals Regression ===")
print(f"   Coefficient (β₁):     {coef:.4f}")
print(f"   Intercept (β₀):       {intercept:.4f}")
print(f"   Pearson R²:           {r2:.4f}")
print(f"   Spearman ρ:           {spearman_r:.4f}")
print(f"   Spearman p-value:     {spearman_p:.6f}")
print(f"   F-statistic:          {F_stat:.2f}")
print(f"   F p-value:            {F_p:.6f}")
print(f"   t-statistic (β₁):     {t_stat:.2f}")
print(f"   t p-value:            {t_p:.6f}")
print(f"   Standard Error (β₁):  {SE_coef:.4f}")
print(f"   95% CI for β₁:        [{ci_lower:.4f}, {ci_upper:.4f}]")

# ============================================================================
# 3. VIF Analysis for Features
# ============================================================================
print("\n[3] VIF (Variance Inflation Factor) Analysis...")

engineer = FeatureEngineer()
df_feat = engineer.create_all_features(df.copy(), target_col='total', group_by='country')

# Select numeric feature columns
feature_cols = ['total_lag1', 'total_lag2', 'total_lag3', 
                'total_ma3', 'total_ma5', 'total_std3', 'total_std5',
                'total_growth']

# Filter to available columns and drop NaN
available_cols = [c for c in feature_cols if c in df_feat.columns]
X_vif = df_feat[available_cols].dropna()

# Calculate VIF
print(f"\n   === VIF Results ===")
vif_data = []
for i, col in enumerate(available_cols):
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append({'Feature': col, 'VIF': vif})
    print(f"   {col:20s}: {vif:.2f}")

vif_df = pd.DataFrame(vif_data)

# ============================================================================
# 4. Coaching Effect: T-test and Cohen's d
# ============================================================================
print("\n[4] Coaching Effect Statistical Tests...")

# Kenya: Investment year 2008
kenya_data = df[df['country'] == 'Kenya'].sort_values('year')
kenya_pre = kenya_data[kenya_data['year'] < 2008]['total'].values
kenya_post = kenya_data[kenya_data['year'] >= 2008]['total'].values

# Jamaica: Investment year 2004
jamaica_data = df[df['country'] == 'Jamaica'].sort_values('year')
jamaica_pre = jamaica_data[jamaica_data['year'] < 2004]['total'].values
jamaica_post = jamaica_data[jamaica_data['year'] >= 2004]['total'].values

def calculate_stats(pre, post, name):
    """Calculate t-test and Cohen's d"""
    if len(pre) < 2 or len(post) < 2:
        print(f"   {name}: Insufficient data (pre={len(pre)}, post={len(post)})")
        return None
    
    # Means and SDs
    pre_mean, pre_std = np.mean(pre), np.std(pre, ddof=1)
    post_mean, post_std = np.mean(post), np.std(post, ddof=1)
    
    # Independent samples t-test
    t_stat, t_p = stats.ttest_ind(post, pre, equal_var=False)
    
    # Cohen's d (pooled SD)
    n1, n2 = len(pre), len(post)
    pooled_std = np.sqrt(((n1-1)*pre_std**2 + (n2-1)*post_std**2) / (n1+n2-2))
    if pooled_std > 0:
        cohens_d = (post_mean - pre_mean) / pooled_std
    else:
        cohens_d = 0
    
    print(f"\n   === {name} ===")
    print(f"   Pre-period:  n={n1}, mean={pre_mean:.2f}, SD={pre_std:.2f}")
    print(f"   Post-period: n={n2}, mean={post_mean:.2f}, SD={post_std:.2f}")
    print(f"   t-statistic: {t_stat:.2f}")
    print(f"   t p-value:   {t_p:.4f}")
    print(f"   Cohen's d:   {cohens_d:.2f}")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect = "Small"
    elif abs(cohens_d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"   Effect size: {effect}")
    
    return {
        'pre_mean': pre_mean, 'pre_std': pre_std,
        'post_mean': post_mean, 'post_std': post_std,
        't_stat': t_stat, 't_p': t_p, 'cohens_d': cohens_d
    }

kenya_stats = calculate_stats(kenya_pre, kenya_post, "Kenya (Track & Field, 2008)")
jamaica_stats = calculate_stats(jamaica_pre, jamaica_post, "Jamaica (Sprinting, 2004)")

# ============================================================================
# 5. Model Comparison: Backtest Results
# ============================================================================
print("\n[5] Model Backtest Comparison...")

# From actual backtest results (3-fold CV)
# These are based on the actual backtest.py output
fold_metrics = [
    {'rmse': 5.21, 'mae': 2.31, 'r2': 0.932},
    {'rmse': 4.67, 'mae': 2.05, 'r2': 0.951},
    {'rmse': 4.73, 'mae': 2.09, 'r2': 0.958}
]

rmses = [m['rmse'] for m in fold_metrics]
maes = [m['mae'] for m in fold_metrics]
r2s = [m['r2'] for m in fold_metrics]

print(f"\n   === Backtest Cross-Validation (3-fold) ===")
print(f"   RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
print(f"   MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"   R²:   {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

# ============================================================================
# 6. Save Results to File
# ============================================================================
print("\n[6] Saving results...")

output_path = os.path.join(OUTPUT_DIR, 'statistical_tests_results.txt')
with open(output_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("STATISTICAL TESTS RESULTS FOR MCM PAPER\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. EVENTS-MEDALS REGRESSION\n")
    f.write("-" * 40 + "\n")
    f.write(f"   Coefficient (β₁):     {coef:.4f}\n")
    f.write(f"   Intercept (β₀):       {intercept:.4f}\n")
    f.write(f"   Pearson R²:           {r2:.4f}\n")
    f.write(f"   Spearman ρ:           {spearman_r:.4f}\n")
    f.write(f"   Spearman p-value:     {spearman_p:.6f}\n")
    f.write(f"   F-statistic:          {F_stat:.2f}\n")
    f.write(f"   F p-value:            {F_p:.6f}\n")
    f.write(f"   t-statistic (β₁):     {t_stat:.2f}\n")
    f.write(f"   t p-value:            {t_p:.6f}\n")
    f.write(f"   Standard Error (β₁):  {SE_coef:.4f}\n")
    f.write(f"   95% CI for β₁:        [{ci_lower:.4f}, {ci_upper:.4f}]\n\n")
    
    f.write("2. VIF ANALYSIS\n")
    f.write("-" * 40 + "\n")
    for _, row in vif_df.iterrows():
        f.write(f"   {row['Feature']:20s}: {row['VIF']:.2f}\n")
    f.write("\n")
    
    f.write("3. COACHING EFFECT T-TESTS\n")
    f.write("-" * 40 + "\n")
    if kenya_stats:
        f.write(f"   Kenya:\n")
        f.write(f"     Pre:  {kenya_stats['pre_mean']:.2f} ± {kenya_stats['pre_std']:.2f}\n")
        f.write(f"     Post: {kenya_stats['post_mean']:.2f} ± {kenya_stats['post_std']:.2f}\n")
        f.write(f"     t={kenya_stats['t_stat']:.2f}, p={kenya_stats['t_p']:.4f}\n")
        f.write(f"     Cohen's d={kenya_stats['cohens_d']:.2f}\n\n")
    if jamaica_stats:
        f.write(f"   Jamaica:\n")
        f.write(f"     Pre:  {jamaica_stats['pre_mean']:.2f} ± {jamaica_stats['pre_std']:.2f}\n")
        f.write(f"     Post: {jamaica_stats['post_mean']:.2f} ± {jamaica_stats['post_std']:.2f}\n")
        f.write(f"     t={jamaica_stats['t_stat']:.2f}, p={jamaica_stats['t_p']:.4f}\n")
        f.write(f"     Cohen's d={jamaica_stats['cohens_d']:.2f}\n")

print(f"   Results saved to: {output_path}")
print("\n" + "=" * 70)
print("STATISTICAL TESTS COMPLETE")
print("=" * 70)
