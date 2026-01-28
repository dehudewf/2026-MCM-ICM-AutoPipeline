# E题建模提示词最终版 (Agent-Enhanced Edition)

> 本文档是E题（决策/评估类）专用建模提示词，整合了Multi-Agent系统和O奖评审标准。
> 核心方法：AHP/EWM/TOPSIS/灰色关联/DEA（区别于C题的ARIMA/LSTM/XGBoost）

---

## 使用说明

```yaml
usage:
  # 审题阶段
  phase_1: "@strategist + 提示词1-3"
  
  # 建模阶段
  phase_2: "@executor + 提示词4-8"
  
  # 检验阶段
  phase_3: "@redcell + 提示词9-10"
  
  # 全程可用
  knowledge_base: "@knowledge:method"

e_problem_focus:
  core_task: "综合评价、决策支持、风险评估"
  core_models: "AHP/EWM/TOPSIS/CRITIC/灰色关联/DEA"
  output_type: "Score, Risk Level, Ranking, 决策建议"
```

---

## 提示词1：E题问题拆解（决策/评估类）

```
【E题问题拆解】- 配合@strategist使用

面对E题任务：{题目全文}

请按O奖标准分析：

一、三问定位（快速定性）
1. 评价什么？
   - 评价对象本质（国家/企业/项目/方案）
   - 评价维度（经济/社会/环境/风险）
   - 评价目的（排序/分类/筛选/优化）

2. 用什么评价？
   - 可获得的指标数据
   - 主观判断 vs 客观数据的比例
   - 指标间的关系（相关性/冲突性）

3. 评价给谁看？
   - 受众决策场景（保险承保/投资决策/政策制定）
   - 可解释性要求（必须解释为什么这样排序）
   - 关键决策问题（是否承保？是否投资？）

二、E题特有要求挖掘
- 指标体系如何构建？（层级结构）
- 权重如何确定？（主观AHP vs 客观EWM）
- 结果如何校验？（敏感性分析是必需章节）
- 不确定性如何处理？（模糊/灰色/区间）

三、评审对标
- 评委最关注的E题要素：
  ✓ 指标体系完整性和合理性
  ✓ 权重计算方法的论证
  ✓ 敏感性分析的深度
  ✓ 决策建议的可操作性

四、E题常见风险
- 指标数据缺失或质量差
- 主观判断矩阵一致性不通过
- 权重方法选择无依据
- 结果对权重过于敏感

输出格式：
| 维度 | 分析 | E题专属策略 |
|------|------|-------------|
```

---

## 提示词2：指标体系构建

```
【指标体系构建】- 配合@strategist使用

评价目标：{目标}
评价对象：{对象}

请构建O奖级指标体系：

一、层次结构设计
```
目标层（Goal）
    ├── 准则层（Criteria）
    │   ├── 指标层（Indicators）
    │   │   ├── 子指标1
    │   │   ├── 子指标2
    │   │   └── ...
    │   └── ...
    └── ...
```

二、指标选取原则（SMART-C）
| 原则 | 检验问题 | 本指标是否符合？ |
|------|----------|------------------|
| Specific 明确性 | 指标定义是否清晰？ | |
| Measurable 可测性 | 能否量化或打分？ | |
| Achievable 可获取性 | 数据能否获取？ | |
| Relevant 相关性 | 与目标是否相关？ | |
| Timely 时效性 | 数据是否及时？ | |
| Comprehensive 综合性 | 是否覆盖全面？ | |

三、指标分类
| 指标名称 | 类型 | 方向 | 量纲 | 数据来源 |
|----------|------|------|------|----------|
| | 效益型/成本型 | 越大越好/越小越好 | | |

四、指标相关性分析
- 高度相关的指标：考虑合并或删除
- 冲突指标：保留，这是多准则决策的核心价值
- 独立指标：权重可能较高

五、E题经典指标体系参考
| 评价领域 | 常用准则层 | 常用指标 |
|----------|------------|----------|
| 可持续发展 | 经济/社会/环境 | GDP、HDI、碳排放 |
| 风险评估 | 脆弱性/暴露度/韧性 | 灾害频率、人口密度、恢复能力 |
| 保险承保 | 财务/运营/市场 | 偿付能力、赔付率、市场份额 |

输出：完整指标体系图 + 指标定义表 + 数据来源说明
```

---

## 提示词3：权重计算方法选择

```
【权重计算方法选择】- 配合@strategist使用

指标数量：{n个}
数据类型：{主观/客观/混合}
参考：E题知识库/权重方法对比

请选择最优权重方法：

一、方法选择决策树
```
有专家判断？
├── 是 → 指标间重要性可比较？
│   ├── 是 → AHP层次分析法
│   └── 否 → 模糊综合评价
└── 否 → 有足够样本数据？
    ├── 是 → 指标差异大？
    │   ├── 是 → 熵权法(EWM)
    │   └── 否 → CRITIC法
    └── 否 → 专家评分 + 主观赋权
```

二、方法对比矩阵
| 方法 | 类型 | 优点 | 缺点 | 适用条件 | 推荐度 |
|------|------|------|------|----------|--------|
| AHP | 主观 | 逻辑性强、可解释 | 主观性、CR检验 | 层次清晰、专家可得 | |
| EWM熵权法 | 客观 | 客观、自动计算 | 忽略专家经验 | 数据量大、差异明显 | |
| CRITIC | 客观 | 考虑相关性 | 计算复杂 | 指标相关性高 | |
| 组合权重 | 混合 | 综合优势 | 需确定组合系数 | 两种信息都有 | |

三、AHP一致性保障
```python
# AHP一致性检验
def check_consistency(matrix):
    n = matrix.shape[0]
    eigenvalues, _ = np.linalg.eig(matrix)
    lambda_max = max(eigenvalues.real)
    CI = (lambda_max - n) / (n - 1)
    RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    CR = CI / RI[n]
    return CR < 0.1  # 通过条件
```

四、组合权重公式
```
W_combined = α × W_AHP + (1-α) × W_EWM
推荐：α = 0.5（平衡主客观）
或：α通过博弈论优化
```

五、权重敏感性预检
- 关键指标权重变化±20%，排序是否剧变？
- 若剧变，需在论文中重点说明

输出：权重方法选择报告 + 选择理由 + 实施方案
```

---

## 提示词4：AHP层次分析法实现

```
【AHP层次分析法】- 配合@executor使用

评价层次结构：{层次图}
参考：E题/src/models/ahp_model.py

请实现完整AHP流程：

一、判断矩阵构建
```python
# 1-9标度法
scale_meaning = {
    1: "同等重要",
    3: "稍微重要",
    5: "明显重要",
    7: "强烈重要",
    9: "极端重要",
    2,4,6,8: "中间值"
}

# 示例：3个准则的判断矩阵
criteria_matrix = np.array([
    [1,   3,   5],   # C1 vs C1, C2, C3
    [1/3, 1,   3],   # C2 vs C1, C2, C3
    [1/5, 1/3, 1]    # C3 vs C1, C2, C3
])
```

二、权重计算方法
```python
def calculate_weights(matrix, method='eigenvalue'):
    """
    三种权重计算方法（论文中可对比展示）
    """
    if method == 'eigenvalue':
        # 特征值法（最精确）
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, max_idx].real
        weights = weights / weights.sum()
        
    elif method == 'geometric_mean':
        # 几何平均法（计算简单）
        gm = np.prod(matrix, axis=1) ** (1/matrix.shape[0])
        weights = gm / gm.sum()
        
    elif method == 'arithmetic_mean':
        # 算术平均法（最简单）
        normalized = matrix / matrix.sum(axis=0)
        weights = normalized.mean(axis=1)
    
    return weights
```

三、一致性检验与自修复
```python
def consistency_check_and_repair(matrix, max_iterations=100):
    """
    一致性检验 + 自动修复
    O奖论文常用的稳健性保障
    """
    RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    n = matrix.shape[0]
    
    for iteration in range(max_iterations):
        # 计算CR
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max = max(eigenvalues.real)
        CI = (lambda_max - n) / (n - 1)
        CR = CI / RI[n] if RI[n] > 0 else 0
        
        if CR < 0.1:
            print(f"✓ 一致性通过: CR = {CR:.4f} < 0.1")
            return matrix, CR
        
        # 自动修复：调整偏离最大的元素
        consistent = calculate_consistent_matrix(matrix)
        deviation = np.abs(matrix - consistent)
        max_idx = np.unravel_index(np.argmax(deviation), deviation.shape)
        
        # 向一致性矩阵方向调整
        matrix[max_idx] = 0.9 * matrix[max_idx] + 0.1 * consistent[max_idx]
        matrix[max_idx[1], max_idx[0]] = 1 / matrix[max_idx]
    
    raise ValueError(f"无法通过一致性检验，最终CR = {CR:.4f}")
```

四、层次总排序
```python
def hierarchical_synthesis(criteria_weights, indicator_weights_matrix):
    """
    层次总排序：综合各层权重得到最终权重
    """
    # indicator_weights_matrix: (n_indicators, n_criteria)
    # 每列是在某准则下各指标的权重
    final_weights = indicator_weights_matrix @ criteria_weights
    return final_weights / final_weights.sum()
```

五、必需输出
- 所有判断矩阵及其CR值
- 权重分布图（饼图/条形图）
- 层次总排序表

输出：完整AHP代码 + 权重结果 + 一致性报告
```

---

## 提示词5：TOPSIS综合评价

```
【TOPSIS综合评价】- 配合@executor使用

决策矩阵：{m方案 × n指标}
指标权重：{来自AHP/EWM}
参考：E题/src/models/topsis_model.py

请实现完整TOPSIS流程：

一、数据预处理
```python
def preprocess_decision_matrix(X, indicator_types):
    """
    处理不同类型指标
    indicator_types: list of 'benefit' or 'cost'
    """
    X_processed = X.copy()
    
    for j, ind_type in enumerate(indicator_types):
        if ind_type == 'cost':
            # 成本型转效益型：取倒数或用max-x
            X_processed[:, j] = 1 / X[:, j]  # 或 X[:, j].max() - X[:, j]
    
    return X_processed
```

二、标准化方法
```python
def normalize(X, method='vector'):
    """
    两种常用标准化方法
    """
    if method == 'vector':
        # 向量标准化（TOPSIS经典方法）
        norm = np.sqrt((X ** 2).sum(axis=0))
        return X / norm
        
    elif method == 'minmax':
        # 极差标准化
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
```

三、理想解确定
```python
def find_ideal_solutions(X_normalized, indicator_types):
    """
    确定正负理想解
    """
    ideal_positive = np.zeros(X_normalized.shape[1])
    ideal_negative = np.zeros(X_normalized.shape[1])
    
    for j, ind_type in enumerate(indicator_types):
        if ind_type == 'benefit':
            ideal_positive[j] = X_normalized[:, j].max()
            ideal_negative[j] = X_normalized[:, j].min()
        else:  # cost
            ideal_positive[j] = X_normalized[:, j].min()
            ideal_negative[j] = X_normalized[:, j].max()
    
    return ideal_positive, ideal_negative
```

四、距离计算与评分
```python
def calculate_topsis_score(X_normalized, weights, indicator_types):
    """
    完整TOPSIS评分流程
    """
    # 加权标准化
    X_weighted = X_normalized * weights
    
    # 理想解
    ideal_pos, ideal_neg = find_ideal_solutions(X_weighted, indicator_types)
    
    # 距离计算（欧氏距离）
    D_positive = np.sqrt(((X_weighted - ideal_pos) ** 2).sum(axis=1))
    D_negative = np.sqrt(((X_weighted - ideal_neg) ** 2).sum(axis=1))
    
    # 相对接近度
    C = D_negative / (D_positive + D_negative + 1e-10)
    
    return C, D_positive, D_negative
```

五、结果解释模板
```markdown
## TOPSIS评价结果

### 排序结果
| 排名 | 方案 | 综合得分 | D+ | D- | 评价 |
|------|------|----------|----|----|------|
| 1 | A | 0.85 | 0.12 | 0.68 | 优秀 |
| 2 | B | 0.72 | 0.24 | 0.62 | 良好 |
| ... | | | | | |

### 关键发现
- 方案A距离正理想解最近（D+=0.12），综合表现最优
- 方案C在指标X上表现突出，但指标Y拖累整体排名
- 敏感性分析显示排名对权重W1最敏感
```

输出：TOPSIS代码 + 评价结果表 + 可视化排序图
```

---

## 提示词6：熵权法(EWM)客观赋权

```
【熵权法客观赋权】- 配合@executor使用

原始数据矩阵：{m样本 × n指标}
参考：E题/src/models/topsis_model.py (EntropyWeightMethod)

请实现熵权法：

一、核心原理
```
信息熵原理：指标差异越大 → 信息量越大 → 权重越高
熵值越小 → 指标区分度越高 → 权重越大
```

二、完整实现
```python
def entropy_weight_method(X, indicator_types):
    """
    熵权法计算权重
    
    Parameters:
    -----------
    X : ndarray, shape (m, n)
        原始数据矩阵，m个样本，n个指标
    indicator_types : list
        指标类型列表，'benefit' 或 'cost'
    
    Returns:
    --------
    weights : ndarray, shape (n,)
        各指标权重
    """
    m, n = X.shape
    
    # Step 1: 数据标准化（极差法）
    X_norm = np.zeros_like(X, dtype=float)
    for j in range(n):
        x_min, x_max = X[:, j].min(), X[:, j].max()
        if indicator_types[j] == 'benefit':
            X_norm[:, j] = (X[:, j] - x_min) / (x_max - x_min + 1e-10)
        else:  # cost
            X_norm[:, j] = (x_max - X[:, j]) / (x_max - x_min + 1e-10)
    
    # Step 2: 计算比重
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    P = np.clip(P, 1e-10, 1)  # 避免log(0)
    
    # Step 3: 计算熵值
    k = 1 / np.log(m)  # 常数
    E = -k * (P * np.log(P)).sum(axis=0)
    
    # Step 4: 计算权重
    D = 1 - E  # 差异系数
    weights = D / D.sum()
    
    return weights, E, D
```

三、熵权法结果解读
```markdown
## 熵权法权重结果

| 指标 | 熵值E | 差异系数D | 权重W |
|------|-------|-----------|-------|
| X1 | 0.92 | 0.08 | 0.15 |
| X2 | 0.78 | 0.22 | 0.42 |  ← 权重最高，区分度最大
| X3 | 0.85 | 0.15 | 0.28 |
| X4 | 0.95 | 0.05 | 0.15 |  ← 权重最低，差异小

解读：X2指标的熵值最低（0.78），说明各样本在该指标上差异最大，
因此获得最高权重（0.42）。这符合"差异即信息"的原则。
```

四、EWM vs AHP对比
```python
def compare_weights(ahp_weights, ewm_weights, indicator_names):
    """权重对比分析"""
    comparison = pd.DataFrame({
        'Indicator': indicator_names,
        'AHP': ahp_weights,
        'EWM': ewm_weights,
        'Difference': np.abs(ahp_weights - ewm_weights),
        'Combined': 0.5 * ahp_weights + 0.5 * ewm_weights
    })
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(indicator_names))
    width = 0.3
    ax.bar(x - width, ahp_weights, width, label='AHP (Subjective)')
    ax.bar(x, ewm_weights, width, label='EWM (Objective)')
    ax.bar(x + width, comparison['Combined'], width, label='Combined')
    ax.set_xticks(x)
    ax.set_xticklabels(indicator_names, rotation=45)
    ax.legend()
    ax.set_title('Weight Comparison: AHP vs EWM')
    plt.tight_layout()
    plt.savefig('output/weight_comparison.png', dpi=300)
    
    return comparison
```

输出：熵权法代码 + 权重结果 + AHP对比分析
```

---

## 提示词7：敏感性分析（权重扰动）

```
【敏感性分析 - E题专用】- 配合@executor使用
> 美赛必需章节，E题尤其重要

模型：{TOPSIS/AHP-TOPSIS/灰色关联}
权重向量：{w1, w2, ..., wn}

请设计敏感性分析方案：

一、权重扰动分析
```python
def weight_sensitivity_analysis(X, base_weights, indicator_types, 
                                perturbation_range=0.2, steps=11):
    """
    权重敏感性分析
    分析单个权重变化对最终排序的影响
    """
    results = []
    n_indicators = len(base_weights)
    
    for i in range(n_indicators):
        # 对第i个指标的权重进行扰动
        perturbations = np.linspace(
            base_weights[i] * (1 - perturbation_range),
            base_weights[i] * (1 + perturbation_range),
            steps
        )
        
        for p in perturbations:
            # 调整权重（保持归一化）
            new_weights = base_weights.copy()
            delta = p - base_weights[i]
            new_weights[i] = p
            # 其他权重按比例调整
            for j in range(n_indicators):
                if j != i:
                    new_weights[j] -= delta * base_weights[j] / (1 - base_weights[i])
            new_weights = new_weights / new_weights.sum()
            
            # 计算新排序
            scores = calculate_topsis_score(X, new_weights, indicator_types)[0]
            ranking = np.argsort(-scores) + 1
            
            results.append({
                'indicator': i,
                'perturbation': p,
                'top_1': np.argmax(scores),
                'ranking': ranking.tolist()
            })
    
    return pd.DataFrame(results)
```

二、排序稳定性分析
```python
def ranking_stability(sensitivity_results, base_ranking):
    """
    分析排序的稳定性
    """
    # 计算Kendall tau相关系数
    from scipy.stats import kendalltau
    
    stability_scores = []
    for _, row in sensitivity_results.iterrows():
        tau, _ = kendalltau(base_ranking, row['ranking'])
        stability_scores.append(tau)
    
    avg_stability = np.mean(stability_scores)
    min_stability = np.min(stability_scores)
    
    print(f"平均稳定性 (Kendall τ): {avg_stability:.3f}")
    print(f"最低稳定性: {min_stability:.3f}")
    print(f"稳定性评级: {'高' if avg_stability > 0.8 else '中' if avg_stability > 0.6 else '低'}")
    
    return stability_scores
```

三、敏感性可视化
```python
def plot_sensitivity_heatmap(sensitivity_results, indicator_names):
    """
    敏感性热力图
    """
    # 计算每个指标的敏感度（排序变化频率）
    sensitivity_matrix = []
    
    for i, name in enumerate(indicator_names):
        indicator_data = sensitivity_results[sensitivity_results['indicator'] == i]
        ranking_changes = indicator_data['ranking'].apply(
            lambda x: len(set(x)) != len(x)  # 是否有排名变化
        ).mean()
        sensitivity_matrix.append(ranking_changes)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(indicator_names, sensitivity_matrix, color='steelblue')
    ax.axhline(y=0.3, color='red', linestyle='--', label='Threshold')
    ax.set_ylabel('Ranking Change Frequency')
    ax.set_title('Sensitivity Analysis: Impact of Weight Perturbation')
    
    # 标注高敏感指标
    for bar, val in zip(bars, sensitivity_matrix):
        if val > 0.3:
            bar.set_color('red')
            ax.annotate('High Sensitivity', xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/sensitivity_heatmap.png', dpi=300)
```

四、结果呈现模板
```markdown
## Sensitivity Analysis Results

### 4.1 Weight Perturbation Analysis

We varied each indicator weight by ±20% and observed the impact on final rankings.

| Indicator | Base Weight | Perturbation Range | Ranking Stability | Sensitivity |
|-----------|-------------|-------------------|-------------------|-------------|
| Economic | 0.35 | [0.28, 0.42] | 0.92 | Low |
| Social | 0.30 | [0.24, 0.36] | 0.85 | Low |
| Environmental | 0.35 | [0.28, 0.42] | 0.68 | **High** |

**Key Finding**: The model is most sensitive to Environmental indicator weights.
When Environmental weight increases beyond 0.40, Alternative C overtakes Alternative A
as the top-ranked option.

### 4.2 Robustness Conclusion

The overall Kendall τ correlation between perturbed and base rankings is 0.82,
indicating **moderate-to-high stability**. The top 3 alternatives remain unchanged
in 87% of perturbation scenarios, demonstrating the robustness of our evaluation.
```

输出：敏感性分析代码 + 热力图 + 稳定性报告
```

---

## 提示词8：鲁棒性验证

```
【鲁棒性验证 - E题专用】- 配合@executor使用

请验证评价结果的鲁棒性：

一、多方法交叉验证
```python
def cross_method_validation(X, weights, indicator_types):
    """
    使用多种方法验证结果一致性
    """
    results = {}
    
    # Method 1: TOPSIS
    topsis_scores = calculate_topsis_score(X, weights, indicator_types)[0]
    results['TOPSIS'] = np.argsort(-topsis_scores) + 1
    
    # Method 2: 简单加权求和 (SAW)
    X_norm = normalize(X, method='minmax')
    saw_scores = (X_norm * weights).sum(axis=1)
    results['SAW'] = np.argsort(-saw_scores) + 1
    
    # Method 3: 灰色关联度
    gra_scores = grey_relational_analysis(X, weights, indicator_types)
    results['GRA'] = np.argsort(-gra_scores) + 1
    
    # 一致性检验
    from scipy.stats import kendalltau, spearmanr
    
    consistency = {}
    methods = list(results.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            tau, _ = kendalltau(results[methods[i]], results[methods[j]])
            consistency[f"{methods[i]}-{methods[j]}"] = tau
    
    return results, consistency
```

二、Bootstrap置信区间
```python
def bootstrap_ranking_confidence(X, weights, indicator_types, n_bootstrap=1000):
    """
    Bootstrap方法估计排序的置信度
    """
    m = X.shape[0]
    rank_distributions = np.zeros((m, n_bootstrap))
    
    for b in range(n_bootstrap):
        # 重采样
        idx = np.random.choice(m, m, replace=True)
        X_boot = X[idx]
        
        # 计算排序
        scores = calculate_topsis_score(X_boot, weights, indicator_types)[0]
        rank_distributions[:, b] = np.argsort(-scores) + 1
    
    # 计算每个方案的排名区间
    rank_ci = {}
    for i in range(m):
        rank_ci[f'Alternative_{i+1}'] = {
            'median': np.median(rank_distributions[i]),
            'CI_lower': np.percentile(rank_distributions[i], 2.5),
            'CI_upper': np.percentile(rank_distributions[i], 97.5)
        }
    
    return rank_ci
```

三、数据扰动测试
```python
def data_perturbation_test(X, weights, indicator_types, noise_level=0.05):
    """
    测试评价结果对数据噪声的敏感度
    """
    base_scores = calculate_topsis_score(X, weights, indicator_types)[0]
    base_ranking = np.argsort(-base_scores) + 1
    
    stability_count = 0
    n_simulations = 100
    
    for _ in range(n_simulations):
        # 添加随机噪声
        noise = np.random.normal(0, noise_level, X.shape) * X
        X_noisy = X + noise
        
        # 计算新排序
        scores = calculate_topsis_score(X_noisy, weights, indicator_types)[0]
        new_ranking = np.argsort(-scores) + 1
        
        # 检查Top-1是否变化
        if new_ranking[0] == base_ranking[0]:
            stability_count += 1
    
    stability_rate = stability_count / n_simulations
    print(f"Top-1 Stability Rate: {stability_rate:.1%}")
    return stability_rate
```

四、鲁棒性总结模板
```markdown
## Robustness Validation

### Multi-Method Consistency
| Method Pair | Kendall τ | Interpretation |
|-------------|-----------|----------------|
| TOPSIS-SAW | 0.89 | High consistency |
| TOPSIS-GRA | 0.82 | High consistency |
| SAW-GRA | 0.85 | High consistency |

**Conclusion**: Three different MCDM methods produce highly consistent rankings
(average τ = 0.85), validating the robustness of our evaluation.

### Ranking Confidence Intervals (95%)
| Alternative | Median Rank | CI Lower | CI Upper |
|-------------|-------------|----------|----------|
| A | 1 | 1 | 2 |
| B | 2 | 1 | 3 |
| C | 3 | 2 | 4 |

**Conclusion**: Alternative A consistently ranks within top 2 in 95% of
bootstrap samples, supporting our recommendation.
```

输出：鲁棒性验证代码 + 多方法对比 + 置信区间
```

---

## 提示词9：Red Cell终极攻击（E题版）

```
【Red Cell终极攻击 - E题版】- 配合@redcell使用

请以SIAM期刊审稿人 + MCM评委会主席的身份审阅：

{论文内容/模型描述}

E题特别攻击维度：

一、指标体系攻击（E题核心）
| 问题 | 检查项 | 评分(1-5) | 改进建议 |
|------|--------|-----------|----------|
| 完整性 | 是否遗漏重要指标？ | | |
| 独立性 | 指标间是否高度相关？ | | |
| 可测性 | 指标数据能否获取？ | | |
| 合理性 | 指标定义是否清晰？ | | |

二、权重方法攻击
- AHP判断矩阵来源？专家是谁？
- EWM熵权法适用条件检验了吗？
- 组合权重的α系数如何确定？
- 为什么不用其他方法（CRITIC/BWM）？

三、TOPSIS攻击
- 为什么选TOPSIS而非其他MCDM方法？
- 标准化方法是否合适？
- 理想解定义是否合理？
- 距离度量是否最优（欧氏 vs 马氏）？

四、敏感性攻击（E题必需）
- 权重扰动范围是否足够（±20%）？
- 是否测试了极端情况？
- 敏感指标的影响是否充分讨论？
- 排序稳定性结论是否有数据支撑？

五、结果攻击
- 排序结果符合直觉吗？异常需要解释
- 决策建议可操作吗？
- 不确定性如何传达给决策者？

六、格式攻击
- 页数 ≤ 25页？
- 无身份信息泄露？
- 所有假设都有Justification？

输出格式：
| 等级 | 问题 | 位置 | E题专属改进 | 优先级 |
|------|------|------|-------------|--------|
| 致命 | | | | 立即 |
| 严重 | | | | 高 |
| 一般 | | | | 中 |
| 建议 | | | | 低 |
```

---

## 提示词10：最终提交检查（E题版）

```
【最终提交检查 - E题版】- 配合@redcell使用

□ 格式规范
  □ 页数 ≤ 25页（不含附录）
  □ 字体 12pt Times New Roman
  □ PDF命名：控制号.pdf
  □ 无身份信息泄露
  □ 控制号仅在首页

□ E题结构完整
  □ 摘要在首页
  □ 指标体系章节存在且完整
  □ 权重计算方法有论证
  □ TOPSIS/综合评价结果清晰
  □ 敏感性分析章节存在（E题必需）
  □ 优缺点讨论存在
  □ 决策建议具体可操作

□ E题特有内容检查
  □ 指标体系图（层次结构图）
  □ 判断矩阵及CR值（如用AHP）
  □ 权重对比图（主观vs客观）
  □ TOPSIS排序结果表
  □ 敏感性分析热力图
  □ 鲁棒性验证结果

□ 技术检查
  □ 所有公式编号正确
  □ AHP一致性比率CR < 0.1
  □ 权重之和 = 1
  □ TOPSIS得分范围[0,1]
  □ 图表编号连续

□ 附件准备
  □ 代码整理 + README
  □ 数据源说明
  □ Summary Sheet单独文件

最终确认：{签名/日期/时间}
```

---

## E题快速决策流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    E题O奖级评价决策流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 问题拆解 ──→ 评价什么？用什么评？给谁看？                   │
│       ↓                                                         │
│  2. 指标体系 ──→ 层次结构 + SMART-C原则 + 相关性分析            │
│       ↓                                                         │
│  3. 权重选择 ──→ AHP(主观) / EWM(客观) / 组合权重               │
│       ↓                                                         │
│  4. 综合评价 ──→ TOPSIS / 灰色关联 / DEA                        │
│       ↓                                                         │
│  5. 敏感性 ──→ 权重扰动 + 排序稳定性 + 关键指标识别             │
│       ↓                                                         │
│  6. 鲁棒性 ──→ 多方法交叉 + Bootstrap + 数据扰动                │
│       ↓                                                         │
│  7. 决策建议 ──→ 排序结果 + 不确定性说明 + 可操作建议           │
│       ↓                                                         │
│  8. Red Cell ──→ 指标攻击 + 权重攻击 + 结果攻击                 │
│       ↓                                                         │
│  9. 提交检查 ──→ 格式 + E题特有结构 + 技术正确性                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## E题核心原则

```
• 指标体系是灵魂，必须完整、独立、可测
• 权重方法需要论证，不能"我选我觉得"
• 敏感性分析是评委必看章节，必须认真做
• 多方法交叉验证增强可信度
• 决策建议必须具体可操作，不能空泛
• AHP一致性检验必须通过（CR < 0.1）
```
