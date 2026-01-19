# Structured A2A Usage Examples (结构化通信使用示例)

> 这是 Phase 2: Structured A2A 的实战示例，展示如何在真实场景中使用。

---

## 示例 1: 完整工作流（审题 → 建模 → 审核）

### Step 1: 用户启动 @strategist

```
用户: @strategist 审题

题目: 预测2028年夏季奥运会各国奖牌数
数据: data/medals.csv (历年奥运会奖牌数据)
```

### Step 2: @strategist 输出结构化战略

```markdown
## 【战略分析】2028奥运奖牌预测

### 问题解构
- **预测目标**: 87个参赛国2028年奖牌数
- **时间跨度**: 基于1896-2024年历史数据
- **关键挑战**: 新兴国家趋势、主办国效应、疫情异常数据

### 创新路径发散

我发散出3条差异化路径...

[自然语言分析...]

---

### 【结构化输出】

```json:a2a:strategist_to_executor
{
  "message_type": "StrategistToExecutor",
  "task_id": "mcm_2026_medal_prediction",
  "timestamp": "2026-01-19T10:00:00Z",
  
  "selected_paths": [
    {
      "path_id": "path_1",
      "name": "时间序列分析路径",
      "priority": 1,
      "innovation_point": "Prophet + 反直觉特征：将主办国劣势转为特征",
      
      "modeling_plan": {
        "primary_model": "Prophet",
        "fallback_model": "ARIMA",
        "ensemble_option": "Prophet + XGBoost",
        "rationale": "时间序列捕捉趋势，XGBoost捕捉国家特征交互"
      },
      
      "data_requirements": {
        "source_files": ["data/medals.csv", "data/gdp.csv"],
        "required_columns": ["Year", "Country", "Medals", "GDP", "Population"],
        "preprocessing_steps": [
          "去除2020异常数据（疫情影响）",
          "按国家分组",
          "处理缺失值（使用前向填充）",
          "标准化GDP和人口"
        ],
        "quality_checks": [
          "检查时间连续性（是否有断档）",
          "检查数据泄露（未来信息不可用）",
          "验证主办国标记正确性"
        ]
      },
      
      "feature_engineering": {
        "time_features": ["Year", "Trend", "Quadrennial_Pattern"],
        "country_features": ["GDP_per_capita", "Population", "Historical_Mean"],
        "lag_features": ["Medals_lag1", "Medals_lag4"],
        "interaction_features": ["GDP_x_Population"],
        "anti_intuitive_features": [
          "HostDisadvantage: 主办国次届下滑效应（反直觉）"
        ]
      },
      
      "success_criteria": {
        "metrics": {
          "mape": {"target": "<15%", "critical": "<20%"},
          "rmse": {"target": "<5", "critical": "<8%"},
          "r2": {"target": ">0.85", "critical": ">0.75"}
        },
        "validation_method": "Leave-one-out cross validation",
        "baseline_comparison": "必须优于简单移动平均法"
      },
      
      "sensitivity_requirements": {
        "parameters_to_test": [
          "GDP变化±10%",
          "Population变化±20%",
          "HostEffect系数×0.5/×2"
        ],
        "threshold": "预测变化<5%为稳定"
      },
      
      "expected_outputs": [
        "predictions_2028.csv",
        "prophet_forecast_plot.png",
        "shap_explanation.png",
        "sensitivity_heatmap.png",
        "model_results_summary.md"
      ]
    }
  ],
  
  "assumptions": [
    {
      "id": "A1",
      "level": "基础假设",
      "content": "未来4年无重大疫情影响",
      "justification": "基于历史规律，重大疫情间隔>10年",
      "impact_if_violated": "预测区间扩大50%",
      "monitoring_method": "关注WHO公告"
    },
    {
      "id": "A2",
      "level": "工作假设",
      "content": "主办国效应为+30%奖牌数",
      "justification": "文献[Smith 2019]显示平均提升28-32%",
      "impact_if_violated": "主办国预测偏低",
      "monitoring_method": "敏感性分析测试±20%"
    }
  ],
  
  "known_challenges": [
    "小国家样本量不足（<30次参赛）",
    "新兴国家趋势难以捕捉",
    "2020数据异常需特殊处理"
  ],
  
  "time_constraints": {
    "deadline": "Hour 52（结果冻结前）",
    "checkpoints": [
      {"hour": 24, "milestone": "数据+特征完成"},
      {"hour": 40, "milestone": "主力模型完成"}
    ]
  }
}
```

**然后说**: "@executor 请根据上述JSON实现 path_1"

```

---

### Step 3: @executor 读取JSON并实现

```markdown
## 【接收到战略任务】

我已读取 StrategistToExecutor JSON:

- **任务ID**: mcm_2026_medal_prediction
- **路径**: path_1 (时间序列分析路径)
- **主力模型**: Prophet
- **目标指标**: MAPE < 15%, RMSE < 5
- **截止时间**: Hour 52

开始实现...

### 实现进展

1. ✅ 数据加载 (data/medals.csv, data/gdp.csv)
2. ✅ 预处理 (去除2020, 处理缺失值)
3. ✅ 特征工程 (8个特征生成)
4. ✅ Prophet训练 (156样本, 交叉验证)
5. ✅ SHAP解释生成
6. ✅ 敏感性分析 (3个参数测试)
7. ✅ 可视化输出 (4张图表)

### 【结构化输出】

```json:a2a:executor_to_redcell
{
  "message_type": "ExecutorToRedCell",
  "task_id": "mcm_2026_medal_prediction",
  "execution_id": "exec_20260119_001",
  "timestamp": "2026-01-19T14:30:00Z",
  "path_id": "path_1",
  
  "execution_summary": {
    "status": "success",
    "total_time_seconds": 2714,
    "code_files": [
      "src/data_loader.py",
      "src/feature_engineer.py",
      "src/model_trainer.py",
      "src/visualizer.py"
    ],
    "lines_of_code": 856
  },
  
  "models_trained": [
    {
      "model_type": "Prophet",
      "version": "1.1.1",
      "train_samples": 156,
      "test_samples": 20,
      "features_used": 8,
      "hyperparameters": {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.05,
        "yearly_seasonality": true
      },
      "training_time_seconds": 45.2
    }
  ],
  
  "results": {
    "predictions": {
      "file_path": "output/predictions_2028.csv",
      "sample_predictions": [
        {"Country": "USA", "Predicted_2028": 113, "CI_Lower": 105, "CI_Upper": 121},
        {"Country": "China", "Predicted_2028": 88, "CI_Lower": 82, "CI_Upper": 94},
        {"Country": "France", "Predicted_2028": 52, "CI_Lower": 47, "CI_Upper": 57}
      ],
      "total_countries": 87
    },
    
    "performance_metrics": {
      "mape": 12.3,
      "mape_target": 15.0,
      "mape_status": "✅ 满足",
      "rmse": 4.7,
      "rmse_target": 5.0,
      "rmse_status": "✅ 满足",
      "r2": 0.89,
      "r2_target": 0.85,
      "r2_status": "✅ 满足",
      "baseline_comparison": {
        "baseline_mape": 23.5,
        "improvement": "47.7% 优于基线"
      }
    },
    
    "feature_importance": [
      {"feature": "Historical_Medals_Mean", "importance": 0.42},
      {"feature": "GDP_per_capita", "importance": 0.23},
      {"feature": "HostEffect", "importance": 0.15},
      {"feature": "Trend", "importance": 0.12},
      {"feature": "HostDisadvantage", "importance": 0.08}
    ],
    
    "shap_analysis": {
      "file_path": "output/shap_explanation.png",
      "summary": "GDP_per_capita对发达国家影响最大，HostEffect对主办国效应显著，HostDisadvantage验证了次届下滑假说"
    },
    
    "sensitivity_analysis": {
      "parameters_tested": [
        {
          "parameter": "GDP变化+10%",
          "mape_change": "+2.1%",
          "max_prediction_change": "+3 medals",
          "stability": "稳定"
        },
        {
          "parameter": "HostEffect×0.5",
          "mape_change": "+4.5%",
          "max_prediction_change": "-15 medals (主办国)",
          "stability": "⚠️ 主办国敏感"
        }
      ],
      "overall_stability": "模型对GDP变化稳定，对HostEffect参数敏感",
      "file_path": "output/sensitivity_heatmap.png"
    }
  },
  
  "artifacts": [
    {
      "type": "notebook",
      "path": "output/modeling.ipynb",
      "description": "完整建模过程",
      "size_mb": 2.4
    },
    {
      "type": "figure",
      "path": "output/prophet_forecast.png",
      "description": "2028年预测趋势图（含置信区间）",
      "resolution": "300 DPI"
    },
    {
      "type": "figure",
      "path": "output/shap_explanation.png",
      "description": "SHAP特征解释"
    },
    {
      "type": "csv",
      "path": "output/predictions_2028.csv",
      "description": "所有国家2028预测结果"
    }
  ],
  
  "known_issues": [
    {
      "severity": "medium",
      "category": "data_quality",
      "issue": "小国家（奖牌数<10）预测误差较大",
      "affected_entities": ["6个小国家"],
      "quantification": "这6个国家的MAPE>25%",
      "impact_on_overall": "对整体MAPE影响+0.8%",
      "root_cause": "样本量不足（<30次参赛记录）",
      "recommendation": "在论文Limitations部分说明小样本局限性",
      "paper_section": "Model Evaluation - Limitations"
    },
    {
      "severity": "low",
      "category": "model_limitation",
      "issue": "主办国效应参数敏感",
      "quantification": "HostEffect×0.5时MAPE增加4.5%",
      "recommendation": "在敏感性分析部分展示此结果",
      "paper_section": "Sensitivity Analysis"
    }
  ],
  
  "assumptions_compliance": [
    {
      "assumption_id": "A1",
      "status": "已满足",
      "implementation": "2020数据已排除"
    },
    {
      "assumption_id": "A2",
      "status": "已验证",
      "implementation": "主办国效应系数设为1.3，敏感性测试±20%"
    }
  ],
  
  "compliance_check": {
    "strategist_requirements_met": true,
    "all_expected_outputs_generated": true,
    "success_criteria_status": {
      "mape": "✅ 满足",
      "rmse": "✅ 满足",
      "validation": "✅ 完成"
    },
    "ready_for_redcell_review": true
  },
  
  "recommended_redcell_focus": [
    "小国家预测误差问题是否影响整体结论",
    "主办国效应参数是否需要更多文献支撑",
    "是否需要补充ARIMA对比实验"
  ]
}
```

**然后说**: "@redcell 请审核上述执行结果，重点关注recommended_redcell_focus中的3个问题"

```

---

### Step 4: @redcell 读取JSON并攻击

```markdown
## 【接收到执行结果】

我已读取 ExecutorToRedCell JSON:

- **执行ID**: exec_20260119_001
- **路径**: path_1
- **状态**: success
- **关键指标**: MAPE=12.3% ✅, RMSE=4.7 ✅
- **已知问题**: 小国家误差大, 主办国参数敏感
- **建议关注**: 3个问题点

开始六维攻击...

---

### 【攻击1: 假设攻击】

✅ **A1 (无疫情假设)**: 已在文中说明，2020数据已排除  
⚠️ **A2 (主办国+30%)**: 仅引用[Smith 2019]，样本量仅23届

**发现问题**:
- 文献支撑不足，评委可能质疑30%系数合理性
- 建议补充2-3篇文献或通过历史数据验证

---

### 【攻击2: 模型攻击】

✅ Prophet适合时间序列  
⚠️ **未测试ARIMA对比**: @strategist推荐的fallback_model未实现

**发现问题**:
- 评委可能质疑为何不尝试ARIMA
- 建议补充ARIMA对比，展示Prophet优势

---

### 【攻击3: 数据攻击】

⚠️ **小国家样本不足**: JSON中已标注，影响+0.8% MAPE

**评估**:
- 技术上可接受（整体指标仍满足）
- 需在论文Limitations说明

---

### 【攻击4: 结果攻击】

✅ MAPE=12.3% 满足<15%目标  
✅ 敏感性分析完整  
⚠️ **置信区间可视化缺失**: 虽有数据，但未生成图表

---

### 【攻击5: 表达攻击】

✅ 所有图表已生成  
⚠️ **图表缺少详细图注**: 需要补充自解释的图注

---

### 【攻击6: 格式攻击】

✅ CSV格式正确  
✅ 图表PNG 300 DPI

---

### 【结构化输出】

```json:a2a:redcell_feedback
{
  "message_type": "RedCellFeedback",
  "task_id": "mcm_2026_medal_prediction",
  "review_id": "review_20260119_001",
  "timestamp": "2026-01-19T16:00:00Z",
  "target_agent": "@executor",
  
  "attack_dimensions": [
    {
      "dimension": "assumption_attack",
      "findings": [
        {
          "severity": "high",
          "issue": "假设A2（主办国+30%）缺乏文献支撑",
          "evidence": "仅引用[Smith 2019]，该文献样本量仅23届",
          "impact": "评委可能质疑假设合理性",
          "recommendation": "补充2-3篇文献，或通过历史数据验证30%系数",
          "action_required": "@executor 补充文献或数据验证",
          "priority": "high"
        }
      ]
    },
    {
      "dimension": "model_attack",
      "findings": [
        {
          "severity": "medium",
          "issue": "未与ARIMA进行对比实验",
          "evidence": "@strategist推荐的fallback_model未测试",
          "impact": "评委可能质疑Prophet的优越性",
          "recommendation": "补充ARIMA对比，展示Prophet的优势",
          "action_required": "@executor 补充对比实验",
          "priority": "medium"
        }
      ]
    },
    {
      "dimension": "result_attack",
      "findings": [
        {
          "severity": "low",
          "issue": "置信区间可视化缺失",
          "evidence": "数据有，但未生成带CI的预测图",
          "impact": "可视化完整性不足",
          "recommendation": "生成置信区间可视化图",
          "action_required": "@executor 补充图表",
          "priority": "low"
        }
      ]
    }
  ],
  
  "overall_assessment": {
    "quality_score": 8.5,
    "quality_breakdown": {
      "technical_correctness": 9.0,
      "innovation": 8.0,
      "completeness": 9.0,
      "presentation": 8.0
    },
    "o_award_readiness": "85%",
    "critical_issues_count": 1,
    "blocking_issues": false,
    "estimated_fix_time": "2 hours"
  },
  
  "action_items": [
    {
      "id": "AI_001",
      "assignee": "@executor",
      "priority": "high",
      "task": "补充主办国效应的文献或数据验证",
      "estimated_time": "1 hour",
      "deadline": "Hour 50"
    },
    {
      "id": "AI_002",
      "assignee": "@executor",
      "priority": "medium",
      "task": "补充ARIMA对比实验",
      "estimated_time": "1 hour",
      "deadline": "Hour 51"
    },
    {
      "id": "AI_003",
      "assignee": "@executor",
      "priority": "low",
      "task": "生成置信区间可视化图",
      "estimated_time": "30 min",
      "deadline": "Hour 51"
    }
  ],
  
  "approval_status": "conditional",
  "approval_conditions": [
    "完成AI_001后可进入论文撰写",
    "AI_002可选，但强烈建议完成",
    "AI_003优先级低，时间不足可放弃"
  ]
}
```

**总结**: 发现1个high级、1个medium级、1个low级问题。完成AI_001后可进入下一阶段。

```

---

## 示例 2: 信息无损传递对比

### 传统方式（信息丢失）

```
用户: @strategist 路径评估

AI: 我推荐路径1，用Prophet模型，因为它适合时间序列...
    需要GDP和人口数据，目标MAPE<15%...

用户: @executor 实现路径1

AI: 好的，我来实现Prophet...
    [问题] 等等，目标MAPE是多少？需要哪些特征？
```

### 结构化方式（信息无损）

```
用户: @strategist 路径评估

AI: [输出 JSON:strategist_to_executor，包含完整信息]

用户: @executor 实现 path_1

AI: 我已读取JSON，开始实现:
    - 目标: MAPE<15% (来自success_criteria.metrics.mape.target)
    - 特征: 8个 (来自feature_engineering)
    - 假设: 2条 (来自assumptions)
    [无需追问，直接开始]
```

---

## 实施建议

### 1. 渐进式启用

| 阶段 | 使用场景 | 收益 |
|------|----------|------|
| **Week 1** | 只在关键交接使用 | 减少信息丢失 |
| **Week 2** | 所有Agent交接使用 | 形成习惯 |
| **Week 3** | 自动读取JSON字段 | 完全无损通信 |

### 2. 混合使用

```markdown
## 【推荐模式】JSON + 自然语言

```json:a2a:strategist_to_executor
{
  "selected_paths": [ ... ],
  "assumptions": [ ... ]
}
```

**简要说明**: 我选择了路径1（时间序列分析），因为...
这样既保证信息完整，又便于人类理解。
```

### 3. 检查清单

**@strategist 输出时**:
- [ ] 包含完整的 selected_paths
- [ ] 包含量化的 success_criteria
- [ ] 包含结构化的 assumptions

**@executor 输出时**:
- [ ] 包含所有 performance_metrics
- [ ] 标注所有 known_issues
- [ ] 列出所有 artifacts

**@redcell 输出时**:
- [ ] 六维攻击完整
- [ ] action_items 可执行
- [ ] 明确 approval_status
