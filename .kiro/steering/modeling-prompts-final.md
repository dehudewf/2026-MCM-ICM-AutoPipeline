# 建模提示词最终版 (Agent-Enhanced Edition)

> 本文档是基础版建模提示词的升级版，整合了Multi-Agent系统和O奖评审标准。

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
```

---

## 提示词1：O奖级问题拆解

```
【O奖级问题拆解】- 配合@strategist使用

面对建模任务：{题目全文}

请按O奖标准分析：

一、三问定位（快速定性）
1. 预测什么？
   - 目标变量本质（连续/离散/时序）
   - 时间范围和空间范围
   - 成功的量化定义

2. 用什么预测？
   - 可获得的信息源
   - 信息与目标的因果链
   - 预测时刻哪些信息已知

3. 预测给谁看？
   - 受众技术水平（评委/决策者）
   - 可解释性等级要求
   - 关键要回答的问题

二、隐藏要求挖掘
- 题目中的模糊概念，如何定义和量化？
- 评委可能期待看到的内容（弦外之音）
- 必须显式声明的假设

三、评审对标
- 初评10分钟能看到什么亮点？
- 终评深度审阅的关键点是什么？
- 如何体现"建模的创造性"？

四、风险评估
- 数据可能缺什么？备选方案？
- 时间最可能卡在哪里？
- 最大的技术不确定性？

输出格式：
| 维度 | 分析 | O奖策略 |
|------|------|---------|
```

---

## 提示词2：差异化创新发散

```
【差异化创新发散】- 配合@strategist使用

基于题目分析：{问题定义}

请用以下方法发散创新角度：

一、SCAMPER创新法
- Substitute：能用什么替代常规方法？
- Combine：能组合哪些跨学科方法？
- Adapt：其他领域的成功案例能否迁移？
- Modify：现有方法能否改进/增强？
- Put to other uses：能否产生意料之外的洞察？
- Eliminate：能否简化复杂性？
- Reverse：反向思考会得到什么？

二、评委视角审视
常规队伍会怎么做？
↓
我们如何差异化？
↓
什么会让评委眼前一亮？

三、反直觉角度
- 有哪些"违背直觉"但可能正确的假设？
- 有哪些被忽视的因素？
- 有哪些"大家都这么想"的思维盲区？

四、跨学科类比
| 本问题 | 类似问题 | 可迁移的方法 |
|--------|----------|--------------|

五、极端假设测试
- 如果数据无限多会怎么做？
- 如果只有1个特征会怎么做？
- 如果必须3分钟解释清楚会说什么？

输出：创新角度清单（至少5条）+ 可行性 + 预期评委反应
```

---

## 提示词3：数据价值金字塔

```
【数据价值金字塔】- 配合@knowledge:data使用

预测目标：{目标}
参考数据源：.kiro/steering/data-sources-and-brainstorm.md

请构建数据价值金字塔：

一、金字塔分层
```
        △ 独特数据（竞争优势）
       ／＼
      △  △ 增强数据（提升精度）
     ／＼／＼
    △  △  △ 核心数据（基本要求）
   ／＼／＼／＼
  △  △  △  △ 公开数据（人人都有）
```

二、数据源评估表
| 数据类型 | 字段 | 来源 | 可得性 | 质量 | 价值 |
|----------|------|------|--------|------|------|

三、代理变量设计
对于不可直接获取的重要因素，设计代理变量：
| 目标因素 | 代理变量 | 代理逻辑 | 数据源 |
|----------|----------|----------|--------|

四、数据关联设计
- 主表：{核心实体}
- 关联键：{year, entity_id}
- 外部数据关联方式

五、数据质量风险
| 风险类型 | 具体描述 | 应对策略 |
|----------|----------|----------|
| 缺失 | | |
| 异常 | | |
| 时效 | | |

输出：数据获取优先级列表 + 备选方案 + 风险清单
```

---

## 提示词4：O奖级特征工程

```
【O奖级特征工程】- 配合@executor使用

预测目标：{目标}
可用数据：{字段列表}
参考：.kiro/steering/modeling-prompts微观.md §一

请按以下框架设计特征：

一、因果链驱动的特征设计
目标 → 影响因素 → 可观测代理 → 特征
（绘制完整因果链图）

二、四类特征思维
1. 历史记忆型
   | 特征 | 计算公式 | 业务含义 | 预期重要性 |
   
2. 趋势动量型
   | 特征 | 计算公式 | 业务含义 | 预期重要性 |
   
3. 相对位置型
   | 特征 | 计算公式 | 业务含义 | 预期重要性 |
   
4. 交互效应型
   | 特征 | 计算公式 | 业务含义 | 预期重要性 |

三、【必需】反直觉特征
设计至少1个"反直觉"特征：
- 来源：跨学科类比/隐藏机制/逆向思维
- 假设：为什么可能有效
- 验证：如何验证假设

四、特征质量检验
对每个特征回答：
□ 一句话能解释含义吗？
□ 预测时这个值已知吗？（防数据泄露）
□ 缺失如何处理？
□ 与目标是线性还是非线性？

五、特征自动进化循环
```python
# 自动化特征进化框架
for iteration in range(max_iterations):
    # 1. 生成候选特征
    candidate_features = generate_features(current_best)
    
    # 2. 评估
    scores = evaluate_features(candidate_features, target)
    
    # 3. 选择
    selected = select_top_k(candidate_features, scores, k=10)
    
    # 4. 更新
    current_best = merge(current_best, selected)
```

输出：特征定义表 + 重要性预期排序 + 反直觉特征说明
```

---

## 提示词5：模型选择决策矩阵

```
【模型选择决策矩阵】- 配合@executor使用

数据特点：{样本量n, 特征数p, 数据结构}
可解释性要求：{高/中/低}
参考：.kiro/steering/modeling-prompts微观.md §二
参考：知识库/模型库*.xlsx

请按以下框架选择模型：

一、模型即假设（核心思想）
| 模型 | 隐含假设 | 假设在本数据成立？ | 可解释性 | 推荐度 |
|------|----------|-------------------|----------|--------|
| 线性回归 | 线性关系 | | ⭐⭐⭐⭐⭐ | |
| 岭回归 | 多重共线性 | | ⭐⭐⭐⭐⭐ | |
| 决策树 | 分段常数 | | ⭐⭐⭐⭐ | |
| 随机森林 | 集成改进 | | ⭐⭐⭐ | |
| XGBoost | 残差学习 | | ⭐⭐⭐ | |
| ARIMA | 时序平稳 | | ⭐⭐⭐⭐ | |
| Prophet | 多季节性 | | ⭐⭐⭐⭐ | |

二、可解释性决策树
需要解释"为什么"？
├── 是 → 需要解释"哪些因素重要"？
│   ├── 是 → 需要解释"改变什么能改变结果"？
│   │   ├── 是 → 线性模型/GAM/决策树
│   │   └── 否 → 树模型 + SHAP
│   └── 否 → 线性模型
└── 否 → 任意模型

三、复杂度-精度权衡
| 模型复杂度 | 预期精度 | 可解释性 | 选择？ |
|------------|----------|----------|--------|
| 低（基线） | | | |
| 中 | | | |
| 高 | | | |

原则：精度差距<5%时，选择更简单的模型

四、推荐组合策略
| 角色 | 模型 | 用途 |
|------|------|------|
| 基线 | | 比较基准 |
| 主力 | | 核心预测 |
| 增强 | | 提升精度 |
| 集成 | | 组合方法 |

五、创新点识别
- 模型本身的创新（改进算法）
- 应用的创新（跨领域迁移）
- 组合的创新（新颖集成）

输出：模型选择报告 + 创新点说明 + 解释方案
```

---

## 提示词6：自修复代码生成

```
【自修复代码生成】- 配合@executor + Claude Opus 4.5使用

任务：{具体实现任务}

请生成符合以下规范的代码：

一、必需结构
```python
"""
Module: {模块名}
Purpose: {功能描述}
Author: MCM Team
Date: 2026
"""

import numpy as np
import pandas as pd
import random

# 可复现性保证
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 自修复装饰器
def self_healing(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[Attempt {attempt+1}/{max_retries}] Error: {e}")
                    if attempt == max_retries - 1:
                        raise
            return None
        return wrapper
    return decorator
```

二、模块化要求
- data_loader.py：数据加载
- feature_engineer.py：特征工程
- model_trainer.py：模型训练
- evaluator.py：评估验证
- visualizer.py：可视化

三、输出规范
每个关键函数必须输出：
```python
def train_model(X, y, **kwargs):
    """
    训练模型
    
    Returns:
        dict: {
            'model': 训练好的模型,
            'metrics': 评估指标字典,
            'feature_importance': 特征重要性,
            'shap_values': SHAP解释值（如适用）
        }
    """
```

四、验证测试
```python
def validate_pipeline():
    """验证整个流程"""
    # 1. 数据验证
    assert data.shape[0] > 0, "数据为空"
    
    # 2. 特征验证
    assert not np.any(np.isinf(features)), "特征包含无穷值"
    
    # 3. 预测验证
    assert predictions.shape == expected_shape, "预测维度错误"
    
    print("✓ Pipeline validation passed")
```

五、SHAP解释（必需）
```python
import shap

def explain_model(model, X):
    """生成SHAP解释"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 保存图表
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('output/shap_summary.png', dpi=300)
    
    return shap_values
```

输出：完整可运行代码 + 使用说明 + 预期输出
```

---

## 提示词7：不确定性估计

```
【不确定性估计】- 配合@executor使用

预测任务：{任务描述}
置信水平：95%
参考：.kiro/steering/modeling-prompts微观.md §三

请设计不确定性估计方案：

一、不确定性来源分解
| 类型 | 来源 | 可减少? | 估计方法 |
|------|------|---------|----------|
| 模型不确定性 | 模型选择、参数估计 | 是 | |
| 数据不确定性 | 固有随机性 | 否 | |
| 输入不确定性 | 特征测量误差 | 部分 | |

二、方法选择（基于认识论）
你相信什么？→ 推荐方法
- "我相信模型是对的" → 残差法
- "我不确定哪个模型对" → 集成方差法
- "我只相信数据" → Bootstrap
- "我有先验知识" → 贝叶斯方法
- "我只要覆盖保证" → 共形预测

三、实现代码
```python
def estimate_uncertainty(models, X, y, method='ensemble'):
    """多方法不确定性估计"""
    
    if method == 'ensemble':
        # 集成方差法
        predictions = np.array([m.predict(X) for m in models])
        point_estimate = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        lower = point_estimate - 1.96 * std
        upper = point_estimate + 1.96 * std
        
    elif method == 'bootstrap':
        # Bootstrap
        n = len(X)
        bootstrap_preds = []
        for _ in range(1000):
            idx = np.random.choice(n, n, replace=True)
            model.fit(X[idx], y[idx])
            bootstrap_preds.append(model.predict(X))
        lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper = np.percentile(bootstrap_preds, 97.5, axis=0)
    
    return {'point': point_estimate, 'lower': lower, 'upper': upper}
```

四、区间校准验证
```python
def validate_coverage(y_true, lower, upper, target=0.95):
    """验证覆盖率"""
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    print(f"Target coverage: {target:.1%}")
    print(f"Actual coverage: {coverage:.1%}")
    print(f"Status: {'✓ PASS' if abs(coverage - target) < 0.05 else '✗ FAIL'}")
    return coverage
```

输出：不确定性估计方案 + 代码 + 校准结果
```

---

## 提示词8：敏感性分析

```
【敏感性分析】- 配合@executor使用
> 美赛必需章节

模型：{模型名称}
关键参数：{参数列表}

请设计敏感性分析方案：

一、参数敏感性
```python
def parameter_sensitivity(model, param_name, param_range):
    """单参数敏感性分析"""
    results = []
    for value in param_range:
        model.set_params(**{param_name: value})
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results.append({'param_value': value, 'score': score})
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot([r['param_value'] for r in results], 
             [r['score'] for r in results], 'o-')
    plt.xlabel(param_name)
    plt.ylabel('Performance')
    plt.title(f'Sensitivity Analysis: {param_name}')
    plt.savefig(f'output/sensitivity_{param_name}.png', dpi=300)
    
    return results
```

二、特征重要性稳定性
```python
def feature_stability(model, X, y, n_iterations=10):
    """特征重要性稳定性检验"""
    importance_records = []
    for _ in range(n_iterations):
        idx = np.random.choice(len(X), len(X), replace=True)
        model.fit(X.iloc[idx], y.iloc[idx])
        importance_records.append(model.feature_importances_)
    
    mean_importance = np.mean(importance_records, axis=0)
    std_importance = np.std(importance_records, axis=0)
    cv = std_importance / (mean_importance + 1e-8)
    
    return pd.DataFrame({
        'feature': X.columns,
        'mean_importance': mean_importance,
        'std': std_importance,
        'cv': cv,
        'stable': cv < 0.3  # CV < 30% 视为稳定
    }).sort_values('mean_importance', ascending=False)
```

三、极端情况测试
```python
def stress_test(model, X, y):
    """极端情况测试"""
    results = {}
    
    # 1. 极值测试
    X_extreme = X.copy()
    X_extreme.iloc[0] = X.max() * 2  # 极大值
    X_extreme.iloc[1] = X.min() / 2  # 极小值
    results['extreme'] = model.predict(X_extreme[:2])
    
    # 2. 缺失测试
    X_missing = X.copy()
    X_missing.iloc[:, 0] = np.nan  # 第一列缺失
    try:
        results['missing'] = model.predict(X_missing)
        results['handles_missing'] = True
    except:
        results['handles_missing'] = False
    
    # 3. 时间外推测试
    # 如果是时序数据，测试未来时间点
    
    return results
```

四、结果呈现模板
| 参数/条件 | 基准值 | 变化范围 | 影响幅度 | 结论 |
|-----------|--------|----------|----------|------|

输出：敏感性分析报告 + 所有图表 + 鲁棒性结论
```

---

## 提示词9：Red Cell终极攻击

```
【Red Cell终极攻击】- 配合@redcell使用

请以SIAM期刊审稿人 + MCM评委会主席的身份审阅：

{论文内容/模型描述}

攻击维度：

一、假设攻击（评审关键）
| 假设 | 合理性评分(1-5) | 攻击点 | 改进建议 |
|------|----------------|--------|----------|

二、模型攻击
- 模型选择最优吗？有更简单的替代吗？
- 数学推导有错误吗？
- 过拟合风险评估？

三、数据攻击
- 数据来源可靠吗？
- 存在选择偏差吗？
- 缺失值处理合理吗？

四、结果攻击
- 结果可信吗？如何验证？
- 不确定性估计合理吗？
- 因果推断正确吗？（相关≠因果）

五、表达攻击
- 逻辑链有断裂吗？
- 图表有误导性吗？
- 摘要准确反映内容吗？

六、格式攻击
- 页数符合要求？（≤25页）
- 有身份信息泄露？
- 引用格式规范？

输出格式：
| 等级 | 问题 | 位置 | 改进建议 | 修复优先级 |
|------|------|------|----------|------------|
| 致命 | | | | 立即 |
| 严重 | | | | 高 |
| 一般 | | | | 中 |
| 建议 | | | | 低 |
```

---

## 提示词10：最终提交检查

```
【最终提交检查】- 配合@redcell使用

□ 格式规范
  □ 页数 ≤ 25页（不含附录）
  □ 字体 12pt Times New Roman
  □ PDF命名：控制号.pdf
  □ 无身份信息泄露
  □ 控制号仅在首页

□ 结构完整
  □ 摘要在首页
  □ 所有问题都有回答
  □ 灵敏度分析章节存在
  □ 优缺点讨论存在
  □ 结论与摘要一致

□ 内容质量
  □ 摘要第一句话吸引人
  □ 所有假设都有论证
  □ 每张图都有洞察性图注
  □ 文中引用了所有图表
  □ 参考文献已引用

□ 技术检查
  □ 所有公式编号正确
  □ 所有引用存在
  □ 图表编号连续
  □ 无拼写错误
  □ 数学符号一致

□ 附件准备
  □ 代码整理完毕
  □ 数据源说明完毕
  □ 摘要单独文件

最终确认：{签名/日期/时间}
```

---

## 快速决策流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    O奖级建模决策流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 问题拆解 ──→ 三问定位 + 隐藏要求 + 评审对标                 │
│       ↓                                                         │
│  2. 创新发散 ──→ SCAMPER + 反直觉 + 跨学科类比                  │
│       ↓                                                         │
│  3. 数据评估 ──→ 价值金字塔 + 代理变量 + 风险清单               │
│       ↓                                                         │
│  4. 特征设计 ──→ 四类思维 + 反直觉特征 + 自动进化               │
│       ↓                                                         │
│  5. 模型选择 ──→ 假设匹配 + 可解释优先 + 创新点                 │
│       ↓                                                         │
│  6. 代码实现 ──→ 自修复 + 可复现 + SHAP解释                     │
│       ↓                                                         │
│  7. 不确定性 ──→ 来源分解 + 方法匹配 + 覆盖验证                 │
│       ↓                                                         │
│  8. 敏感性 ──→ 参数 + 特征稳定 + 极端测试                       │
│       ↓                                                         │
│  9. Red Cell ──→ 六维攻击 + 问题分级 + 改进建议                 │
│       ↓                                                         │
│  10. 提交检查 ──→ 格式 + 结构 + 内容 + 技术                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心原则

```
• 建模是讲故事，不是套模型
• 可解释性是专业性的体现
• 承认不确定性是诚实的表现
• 验证是发现问题，不是证明正确
• 简单模型优先，复杂需要理由
• 反直觉特征是差异化的关键
```
