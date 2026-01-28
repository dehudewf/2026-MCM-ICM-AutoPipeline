# MCM 2026 E题 Battle Quick Reference Card (E题作战快速参考卡)

## 打印此页面，贴在桌面

---

## 1. Team Division - E题特化分工

| 角色 | E题职责 | 使用Agent | 时间重心 |
|------|---------|-----------|----------|
| **主策师 Master Strategist** | 指标体系设计+权重方法选择+审核 | @strategist, @redcell | 指标90% → 调度50% → 审核90% |
| **技术导演 Tech Director** | AHP计算+TOPSIS实现+敏感性分析 | @executor, @strategist | 数据30% → 建模90% → 支援20% |
| **内容架构师 Content Architect** | 写作+评价图表+决策建议整合 | @executor, @redcell | 框架40% → 内容60% → 整合100% |

---

## 2. E题 Critical Checkpoints (E题关键节点)

| 时间 | E题必须完成 | 未完成则 | 责任人 |
|------|-------------|----------|--------|
| **Hour 6** | 评价目标+评价对象确定 | 延长到Hour 8 | Commander |
| **Hour 12** | 指标体系设计完成（3层结构） | Commander强制拍板 | Commander |
| **Hour 18** | 指标数据来源确认 | 用代理变量 | Engineer |
| **Hour 24** | 数据收集+AHP判断矩阵 | 简化指标数量 | Engineer |
| **Hour 30** | AHP权重计算+CR检验通过 | 调整矩阵或用EWM | Engineer |
| **Hour 36** | EWM权重计算+组合权重 | 只用单一方法 | Engineer |
| **Hour 48** | TOPSIS评价完成+初步排序 | 用SAW简化方法 | Engineer |
| **Hour 60** | 敏感性分析完成（权重扰动±20%） | 简化分析 | Engineer |
| **Hour 72** | 结果冻结（禁止改数据/权重） | 全员转写作 | All |
| **Hour 84** | 全文整合+摘要+决策建议 | 只保核心内容 | Narrator |
| **Hour 96** | 论文完成（可PDF） | 只保核心章节 | Narrator |
| **Hour 98** | @redcell终极攻击完成 | 只修致命问题 | Commander |
| **Hour 100** | 提交 | - | Commander |

---

## 3. E题 Agent Triggers (Agent触发词)

```
@strategist  → 指标体系设计、权重方法选择、创新角度
@executor    → AHP计算、TOPSIS实现、敏感性分析、写作执行
@redcell     → 指标攻击、权重攻击、排序攻击、终极审核
```

---

## 4. E题 Knowledge Base (知识库调用)

```
@knowledge:model   → E题模型库（AHP/EWM/TOPSIS/灰色关联）
@knowledge:paper   → E题O奖论文结构
@knowledge:viz     → E题可视化规范
@knowledge:data    → E题数据源（World Bank/UN/WHO）
@knowledge:method  → E题建模方法
```

---

## 5. E题 Tool Assignment (工具分配)

| 任务 | 工具 | 使用者 |
|------|------|--------|
| AHP/TOPSIS代码 | Trae/Qoder + Claude Opus 4.5 | Engineer |
| 论文撰写 | Claude Opus 4.5 + Overleaf | Narrator |
| 层次结构图 | Nano Banana Pro / draw.io | Narrator |
| 权重对比图 | Matplotlib + Seaborn | Narrator |
| 敏感性热力图 | Matplotlib + Seaborn | Narrator |
| TOPSIS排序图 | Origin + Matplotlib | Narrator |

---

## 6. E题 Emergency Protocols (紧急预案)

| 情况 | E题立即行动 |
|------|-------------|
| AHP CR > 0.1 | 使用自修复算法调整矩阵 |
| 指标数据缺失 | 设计代理变量或简化指标 |
| 权重差异过大（AHP vs EWM） | 检查数据质量，考虑组合权重 |
| TOPSIS排序反直觉 | 检查指标方向和标准化方法 |
| 敏感性过高 | 在论文中说明，不隐藏 |
| Hour 72论文<50% | 全员停止建模，转入写作 |

---

## 7. E题 Golden Structure (黄金结构)

```
1. Summary Sheet (摘要) ← 重中之重，首页
2. Introduction (引言)
3. Assumptions (假设) ← 评审关键
4. Notations (符号)
5. Indicator System (指标体系) ← E题特有
6. Weight Calculation (权重计算) ← E题核心
7. Comprehensive Evaluation (综合评价)
8. Results (结果)
9. Sensitivity Analysis (敏感性) ← E题必需
10. Evaluation (评价) ← 必需
11. Decision Recommendations (决策建议) ← E题特有
12. Conclusion (结论)
13. References (参考)
```

---

## 8. E题 Nano Banana Quick Prompts

```
层次结构图: "Hierarchical structure diagram for multi-criteria evaluation. 
           Goal-Criteria-Indicators layout, academic style, clean connections."

权重对比图: "Comparative bar chart showing AHP vs EWM weights. 
           Side-by-side bars, professional color scheme, Nature quality."

敏感性热力图: "Sensitivity analysis heatmap for weight perturbation. 
             Red-blue colormap, clear annotations, publication ready."

TOPSIS排序图: "TOPSIS ranking visualization with D+ and D- distances.
             Horizontal bar chart, clear labels, academic style."

雷达图: "Radar chart comparing alternatives across multiple criteria.
        Clean design, professional colors, informative labels."
```

---

## 9. E题 Core Code Templates

### AHP权重计算
```python
import numpy as np
SEED = 42
np.random.seed(SEED)

# AHP一致性检验
RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}

def ahp_weights(matrix):
    n = matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_idx].real
    weights = np.abs(weights) / np.abs(weights).sum()
    
    # 一致性检验
    lambda_max = max(eigenvalues.real)
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI[n] if RI[n] > 0 else 0
    
    assert CR < 0.1, f"CR={CR:.4f} > 0.1, 请调整判断矩阵"
    return weights, CR
```

### 熵权法
```python
def ewm_weights(X, indicator_types):
    m, n = X.shape
    # 标准化
    X_norm = np.zeros_like(X, dtype=float)
    for j in range(n):
        x_min, x_max = X[:, j].min(), X[:, j].max()
        if indicator_types[j] == 'benefit':
            X_norm[:, j] = (X[:, j] - x_min) / (x_max - x_min + 1e-10)
        else:
            X_norm[:, j] = (x_max - X[:, j]) / (x_max - x_min + 1e-10)
    
    # 熵值
    P = X_norm / X_norm.sum(axis=0, keepdims=True)
    P = np.clip(P, 1e-10, 1)
    E = -1/np.log(m) * (P * np.log(P)).sum(axis=0)
    
    # 权重
    weights = (1 - E) / (1 - E).sum()
    return weights
```

### TOPSIS评价
```python
def topsis(X, weights, indicator_types):
    # 向量标准化
    X_norm = X / np.sqrt((X**2).sum(axis=0))
    X_weighted = X_norm * weights
    
    # 理想解
    ideal_pos = np.where(
        np.array(indicator_types) == 'benefit',
        X_weighted.max(axis=0),
        X_weighted.min(axis=0)
    )
    ideal_neg = np.where(
        np.array(indicator_types) == 'benefit',
        X_weighted.min(axis=0),
        X_weighted.max(axis=0)
    )
    
    # 距离与得分
    D_pos = np.sqrt(((X_weighted - ideal_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((X_weighted - ideal_neg)**2).sum(axis=1))
    scores = D_neg / (D_pos + D_neg + 1e-10)
    
    return scores, D_pos, D_neg
```

### 组合权重
```python
def combine_weights(ahp_w, ewm_w, alpha=0.5):
    """alpha=0.5表示主客观各占50%"""
    return alpha * ahp_w + (1 - alpha) * ewm_w
```

---

## 10. E题 Final Checklist (提交前检查)

**致命项（必检）:**
- [ ] PDF命名: `队伍控制号.pdf`
- [ ] 页数 ≤ 25页（不含Summary Sheet和附录）
- [ ] 无任何身份信息泄露（页眉页脚、元数据、图片水印）
- [ ] Summary Sheet在第一页
- [ ] 每页页眉有论文编号

**E题结构完整:**
- [ ] Summary Sheet: 半页-1页，包含评价方法和核心结论
- [ ] Introduction + Literature Review存在
- [ ] Assumptions + Justifications存在
- [ ] Notations表格存在
- [ ] **Indicator System章节存在（层次结构图）**
- [ ] **Weight Calculation章节存在（AHP/EWM/组合）**
- [ ] **Comprehensive Evaluation章节存在（TOPSIS结果）**
- [ ] Results + Analysis存在
- [ ] **Sensitivity Analysis存在（权重扰动分析）**
- [ ] Model Evaluation (Strengths/Weaknesses)存在
- [ ] **Decision Recommendations存在（可操作建议）**
- [ ] Conclusion存在
- [ ] References存在

**E题技术检查:**
- [ ] AHP一致性比率 CR < 0.1（必须通过）
- [ ] 所有权重之和 = 1
- [ ] TOPSIS得分范围 [0, 1]
- [ ] 敏感性分析覆盖权重扰动±20%
- [ ] 指标方向正确（效益型/成本型）

**E题图表检查:**
- [ ] 指标层次结构图
- [ ] AHP判断矩阵及CR值
- [ ] 权重对比图（AHP vs EWM）
- [ ] TOPSIS排序结果图
- [ ] 敏感性分析热力图
- [ ] 每张图有编号、图注、被引用

**附件:**
- [ ] 代码整理 + README
- [ ] 数据源说明
- [ ] Summary Sheet单独文件

---

## 11. E题 Method Quick Reference

| 方法 | 类型 | 用途 | 关键检验 |
|------|------|------|----------|
| **AHP** | 主观赋权 | 专家经验→权重 | CR < 0.1 |
| **EWM熵权法** | 客观赋权 | 数据差异→权重 | 样本量>10 |
| **CRITIC** | 客观赋权 | 考虑相关性 | 指标相关性分析 |
| **组合权重** | 混合 | 综合主客观 | α系数论证 |
| **TOPSIS** | 综合评价 | 距离理想解排序 | 标准化方法 |
| **灰色关联** | 综合评价 | 关联度排序 | 分辨系数ρ |
| **DEA** | 效率评价 | 相对效率 | DMU数量 |

---

## 12. E题 vs C题 Quick Comparison

| 维度 | C题（预测类） | E题（评价类） |
|------|---------------|---------------|
| 核心任务 | 预测未来值 | 评价排序 |
| 核心模型 | ARIMA/LSTM/XGBoost | AHP/EWM/TOPSIS |
| 数据类型 | 时间序列 | 多维指标 |
| 结果形式 | R²/RMSE/MAE | Score/Rank/Level |
| 敏感性 | 参数敏感性 | **权重敏感性** |
| 关键章节 | Forecasting | **Indicator System + Weight** |

---

## The E题 Formula

```
E题O奖 = (完整指标体系 × 合理权重论证 × 严谨敏感性分析) ^ 团队协作
```

**Good Luck! Go for O-Award!**
