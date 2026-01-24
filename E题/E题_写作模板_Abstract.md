# E题 Abstract 写作模板（基于20篇O奖论文）

## Abstract 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **1. 背景引入** | 1-2句点明时代背景/问题紧迫性，常用句式："As... increases/poses challenges" "The growing number of..." "Due to..." | **#2407414 (2024)**: "The growing number of extreme weather events poses significant challenges for insurance industry, communities and property developers in terms of their resilience and sustainability" |
| **2. 问题总述** | 概括本文要解决的整体问题，使用"In this paper, we..." 或 "In order to..." 引出 | **#2209336 (2022)**: "In this paper, in order to comprehensively evaluate all aspects of forest value and formulate a reasonable plan to promote forest development, we established an E³-M Model" |
| **3. Task 1 描述** | 常用句式（3种格式）：①"For Task 1, we developed..."(~50%) ②"First/Firstly, we establish..." ③"For Question one, we..." | **#2401102 (2024)**: "For Task 1, we developed a Property Insurance Posture Model... We used an ARIMA model to forecast the future risk..." **#2400860**: "First, we establish a Risk Analysis model..." |
| **4. Task 2 描述** | 常用句式（3种格式）：①"For Task 2, we introduced..."(~50%) ②"Second/Secondly, we develop..." ③"For Question two, we..." | **#2401102 (2024)**: "For Task 2, we introduced Innovative Risk Prediction Property Development Method (IRPPDM), which integrated DBMI with real estate companies' willingness index" **#2400860**: "Second, we develop a Risk-incorporated Capital Asset Pricing Model..." |
| **5. Task 3 描述** | 常用句式（3种格式）：①"For Task 3, we applied [模型] to [案例]...According to our model, [地点] had a score of [数值]" ②"Third, we establish..." ③"For Question three, we..." | **#2401102 (2024)**: "For Task 3, we used Historic Landmark Preservation Model... According to our model, Temple of Literature in Hanoi, Vietnam had a score of 0.463" **#2400860**: "Third, we establish a Building Preservation Model..." |
| **6. 敏感性声明** | 最后1句固定陈述模型验证，句式："Finally, we analyze the sensitivity and robustness of our models" / "The result shows that our model is stable and..." | **#2410578 (2024)**: "Finally, we perform a sensitivity analysis and evaluation of our model. The result shows that our model is stable and capable" |
| **7. Keywords** | 5-7个关键词，格式：核心模型×2 + 核心方法×2 + 应用领域×1-2 | **#2400860 (2024)**: "Keywords: Risk Analysis, Risk-Capital Asset Pricing Model, Sperman-CRITIC, AHP, Building Preservation Model" |

---

## Abstract 高频句式库

### 背景引入句式（选1）
| 句式模板 | 实例 |
|----------|------|
| As [趋势] increases/intensifies, [挑战] becomes ever more pressing | "As the tapestry of nature weaves its unpredictable patterns, humanity's quest for stability becomes ever more pressing" (#2400860) |
| The growing number of [事件] poses significant challenges for [对象] | "The growing number of extreme weather events poses significant challenges for insurance industry" (#2407414) |
| Due to [原因], people are very concerned about [关注点] | "Due to the huge role of forests in alleviating the greenhouse effect, people are very concerned about the carbon sequestration" (#2209336) |
| [领域] faces a crisis from [原因], leading to [后果] | "The property insurance industry faces a crisis from increased extreme weather events, leading to higher claims and premiums" (#2400860) |

### Task描述句式（每个Task选1）
| 句式模板 | 适用场景 |
|----------|----------|
| For Task N, we developed a [模型名称] Model to [动词] the [目标] | 建立全新模型 |
| For Task N, we established a [评估/决策] Model based on [准则/理论] | 基于理论建模 |
| For Task N, we introduced [方法缩写], which integrated [A] with [B] | 方法融合创新 |
| For Task N, we enhanced our [基础模型] by incorporating [新元素] | 模型改进 |
| For Task N, we applied [模型] to [具体案例] to validate/demonstrate | 案例验证 |

### 结果陈述句式
| 句式模板 | 实例 |
|----------|------|
| The result/conclusion indicates that [结论] | "The conclusion indicates the prediction of risk level of Mumbai will transition from level II to III" (#2407414) |
| According to our model, [案例] had a score of [数值] | "According to our model, Temple of Literature in Hanoi had a score of 0.463" (#2406206) |
| The calculation/prediction shows that [定量结论] | "The calculation shows that both locations should be underwritten" (#2410578) |
| We recommend [具体建议] | "We recommend insurance securitization and partnerships with local governments" (#2400860) |

---

## E题 vs C题 Abstract 核心差异

| 对比维度 | C题（预测类） | E题（决策/评估类） |
|----------|--------------|-------------------|
| **整体结构** | 问题→方法→结果→价值（整体流） | **按Task1/2/3分段**（分块式） |
| **方法表述** | "ARIMA-LSTM适配时间序列" | "AHP定权重 + EWM验证 + TOPSIS排序" |
| **结果形式** | R²=0.8377, RMSE=0.12 | Risk Level II→III, Score=0.463 |
| **量化指标** | 单一精度指标 | 多维度评分 + 风险等级 + 排序 |
| **决策导向** | "预测准确" | "是否承保/是否开发/如何保护" |
| **结尾** | 预测结论 | **建议+敏感性分析声明** |

---

## E题 Abstract 模板（可直接套用）

```
[背景句：1句点明问题紧迫性]

In order to [总体目标], we build [N] models: Model I: [模型1名称] and Model II: [模型2名称].

For Task 1, we developed a [模型全称] ([缩写]) to [目的]. We utilized [方法1] to [计算1], and [方法2] to [计算2]. The result indicates that [量化结论，含具体数值/等级].

For Task 2, we established a [模型名称] considering [N个维度/指标]. Using [AHP/EWM/TOPSIS/组合], we determined [权重/评分/排序]. Applied to [具体地区], the prediction results indicate that [排序/等级结论].

For Task 3, we applied our model to [具体案例/地标]. We estimated the [评估对象] by [方法] considering [N个指标]. According to our model, [案例名称] had a score of [数值]. We recommend [具体建议].

Finally, we analyze the sensitivity and robustness of our models. The result shows that our model is stable and [形容词].

Keywords: [模型1], [模型2], [方法1], [方法2], [应用领域]
```

---

# E题 Introduction 写作模板（基于20篇O奖论文）

## Introduction 子章节结构统计（20篇）

| 子章节 | 出现频率 | 是否必须 |
|--------|---------|----------|
| 1.1 Problem Background | 19/20篇 | **必须** |
| 1.2 Restatement of the Problem | 17/20篇 | **必须** |
| 1.3 Our Work | 14/20篇 | **必须** |
| 1.x Literature Review | 4/20篇 | 可选 |

---

## Introduction 结构分析表（基于原文统计修正版）

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **1.1 Problem Background** | **层次1：全球性问题**（"Global/Extreme weather events" 40次）→ **层次2：数据支撑**（百分比/金额统计 62次）→ **层次3：行业影响** | **#2407414 (2024)**: "Global extreme weather events have been on the rise... Instances of hurricanes, floods, droughts, and wildfires are increasing... insurance protection gap currently stands at **57%** globally" |
| **1.2 Restatement of the Problem** | **3种主流格式**：①"> Problem N:"格式（8次）②"Task N:"问题列表（13次）③"¢/•"符号列表；**"In order to..."引入句**（14次）常见 | **#2400860 (2024)**（格式①）: "> Problem 1: The property insurance industry faces a crisis... > Problem 2: Insurance companies must decide... > Problem 3: Participants are tasked with..." **#2410578 (2024)**（格式③）: "¢ Based on the measurement of the risk... ¢ Another mathematical model is developed... ¢ For some buildings with strong cultural..." |
| **1.3 Our Work** | **两种主流格式**：①**"For [stakeholder], we..."**（约43次）→按利益相关方分段 ②**"For problem one/1., we..."**→按问题编号分段；**流程图**（"flowchart"/"Figure X: Our work"，约59次）；句式："In order to clearly illustrate our work, we draw the flowchart" | **#2401102 (2024)**（格式①）: "**For insurance companies**, we have developed a property insurance deployment model... **For real estate developers**, we extended the insurance model... **For community leaders**, we have developed a historical landmark significance model..." **#2407414 (2024)**（格式②）: "1. For problem one: We first demonstrate the intensity... 2. For problem two: Our insurance model is incorporated... 3. For problem three: A landmark preservation model is established..." |
| **1.x Literature Review** (5/20篇，可选) | 综述已有研究 + 指出Gap；句式："[Author] et al. [research content]... However, [gap]..." | **#2407414 (2024)**: "Previous studies have focused on... However, few studies have addressed the integration of..." |

---

# E题 Assumptions 写作模板（基于20篇O奖论文）

## Assumptions 统计数据（基于原文统计修正版）

| 统计项 | 数据 |
|--------|------|
| 章节标题 | "2 Assumptions and Justifications" 或 "2 Assumptions and Explanations" |
| 常见数量 | **3-5条**/篇 |
| Justification出现 | **43次**（大多数论文配对） |
| 引入句 | **8次**（"make reasonable assumptions to simplify the model..."） |

### 格式统计（原文数据）

| 格式 | 出现次数 | 是否主流 |
|------|---------|----------|
| `Assumption N:` 编号格式 | **29次** | ✅ **最主流** |
| `¢ Assumption:` 符号格式 | 12次 | 次主流 |
| `• / ⬤ Assumption:` 格式 | 2次 | 少见 |

---

## Assumptions 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **引入句**（8/20篇有） | 先用1句说明为什么需要假设，句式："It is not possible to model every possible scenario. So we make some reasonable assumptions to simplify the model, each with a corresponding explanation:" | **#2307336 (2023)**: "The risk assessment of light pollution... should take into account economic, social, ecological and other factors. It is not possible to model every possible scenario. So we make some reasonable assumptions..." |
| **Assumption N + Justification N 配对** | **每条Assumption后紧跟Justification**，说明假设的合理性 | **#2305598 (2023)**: "Assumption1: In this paper, LPI is defined as... **Justification**: The lower the level of light pollution, the better..." |
| **Assumption内容类型** | ①数据准确性 ②区域稳定性 ③理性人假设 ④独立性假设 ⑤利润最大化 ⑥可持续性 | **#2410578 (2024)**: "Assumption 1: It is assumed that the insurance companies seek to maximize their profits." |

---

## Assumptions 4种原文格式（按频率排序）

### 格式1：`Assumption N:` 编号格式（29次，最主流）

```
2 Assumptions and Justifications

Assumption 1: It is assumed that the insurance companies seek to maximize their profits.

Assumption 2: It is assumed that insurance companies consider the issue of sustainability
and consider stable income over a period of time in the future.

Assumption 3: Assume that all insurance policies discussed in this paper are fair and full.

Assumption 4: It is assumed that every individual is rational and seeks to maximize personal utility.

Assumption 5: It is assumed that the data we cited and referenced in this paper are accurate and credible.
```
—— **#2410578 (2024)**

### 格式2：`Assumption N + Justification N` 编号配对（43次Justification）

```
2 Assumptions and Explanations

Assumption1: In this paper, LPI (Light Pollution Index) is defined as the score of the light
environment of a region. The higher the LPI value, the lower the level of light pollution.
Justification: The lower the level of light pollution in an area, the better the light environment.

Assumption2: In the correlation analysis, only factors with strong correlation are retained.
Justification: The factors affecting light pollution are complex. If factors of lesser relevance
are also taken into account, it will make the study results less significant.

Assumption3: Assume the data collected from the internet is true and reliable.
Justification: All data are obtained from the official website, so the reliability is guaranteed.
```
—— **#2305598 (2023)**
---

## 常见Assumption内容模板

| 类型 | 出现次数 | 模板句式 |
|------|---------|----------|
| **数据准确性** | 5次 | "The data we use are accurate and valid." / "The data we cited and referenced in this paper are accurate and credible." |
| **区域稳定性** | 12次 | "The regions under study will remain peaceful and stable, with no significant events other than natural disasters occurring." |
| **理性人假设** | 7次 | "Every individual is rational and seeks to maximize personal utility." |
| **独立性假设** | 6次 | "The catastrophe insurance business is an independent process and is not correlated with other types of insurance." |
| **利润最大化** | 多次 | "Insurance companies seek to maximize their profits." |
| **可持续性** | 多次 | "Insurance companies consider the issue of sustainability and stable income over a period of time." |

---

# E题 Notations 写作模板（基于20篇O奖论文）
---

## Notations 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **3 Notations** | **表格形式**：Symbol + Description + Unit三列；通常10-20个符号；按出现顺序排列 | **#2400860 (2024)**: "Table 1: The key mathematical notations used in this paper" |
| **符号命名规范** | 常见符号：$EAL$(年期望损失)、$CRF$(社区风险因子)、$VOL$(Value of Landmark)、$ROL$(Risk of Landmark)、$w_i$(权重)、$P$(Premium) | **#2407414**: "$EAL$ - Expected Annual Loss; $CRF$ - Community Risk Factor; $POR$ - Probability of Ruin" |

---

## Notations 表格模板

```
3 Notations

The key mathematical notations used in this paper are listed in Table 1.

Table 1: Notations
| Symbol | Definition | Unit |
|--------|------------|------|
| $EAL$ | Expected Annual Loss | $1B |
| $CRF$ | Community Risk Factor | - |
| $P$ | Insurance Premium | $ |
| $w_i$ | Weight of indicator $i$ | - |
| $VOL$ | Value of Landmark | - |
| $ROL$ | Risk of Landmark | - |
| $DPI$ | Development Potential Index | - |
| $N(t)$ | Number of claims over a period of time | - |

* There are some variables that are not listed here and will be discussed in detail in each section.
```
—— **#2400860 (2024)**


---

# E题 Data Collection 写作模板（基于20篇O奖论文）


## Data Collection 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **4.1 Data Collection** | ①数据来源：官方机构(World Bank, FAO, NOAA)；②数据描述：时间范围+地理范围+数据量；③可靠性声明 | **#2400860 (2024)**: "Our data is collected from the World Bank and some other official websites and research papers" |
| **E题常见数据源** | World Bank, Munich Re, NOAA, FAO, Global Forest Watch, Swiss Re | **#2401102**: "Swiss Re predicted that losses from weather-related events... Data from Munich Reinsurance highlights..." |

---

# E题 Model Preparation 写作模板（部分论文有独立章节）

## Model Preparation 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **4 Model Preparation** | 独立于具体Model的数据准备章节，包含：4.1 Selection of Research Areas + 4.2 Data Overview | **#2407414 (2024)**: "4 Model Preparation: 4.1 Selection of Research Areas... 4.2 Data Overview..." |
| **4.1 Selection of Research Areas** | 说明为什么选择特定地区作为研究对象（如极端天气频发、数据可用性） | **#2407414**: "We selected Mumbai and Cairns based on their high frequency of extreme weather events and data availability" |
| **4.2 Data Overview** | 数据来源、时间范围、变量描述、数据质量说明 | **#2407414**: "Data collected from NOAA, World Bank spanning 1990-2023..." |

---

## Model Preparation 模板（可选独立章节）

```
4 Model Preparation

4.1 Selection of Research Areas

We select [地区1] and [地区2] as our research areas based on:
- High frequency of extreme weather events
- Availability of comprehensive data
- Representative of different economic conditions

4.2 Data Overview

| Data Source | Time Range | Variables | Description |
|-------------|------------|-----------|-------------|
| World Bank | 1990-2023 | GDP, Population | Economic indicators |
| NOAA | 2000-2023 | Disaster frequency | Weather data |
| Munich Re | 2010-2023 | Insurance losses | Financial data |

The data quality is ensured by cross-validation with multiple official sources.
```


# E题 Model Development 写作模板（基于20篇O奖论文）

---

## Model 结构分析表（基于原文目录提取）

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **X Model I: [Task1模型名]** | **子章节标准结构：**X.1 Selection/Data Collection → X.2 Model Establishment → X.3 Application → X.4 Case Study | **#2407414**: "5 Climate Risk Assess Model: 5.1 Distribution Fitting → 5.2 Model Establishment for POR → 5.4 Case Study: Mumbai and Cairns" |
| **X.1 Indicator Selection** | "指标选取" - 列出评价指标，分维度组织（8个指标/4维度常见） | **#2401102**: "6.1 Indicator Selection... We consider 8 indicators in 4 dimensions" |
| **X.2 Weight Calculation** | **标题常带方法名**："Weight Calculation: Based on AHP" / "Weight Determination Using AHP-EWM" | **#2407414**: "7.2 Weight Determination Using AHP-EWM" |
| **X.3 Value Calculation / Application** | 计算结果 + 应用到具体地区 | **#2401102**: "6.3 Value Calculation and Extent Estimation" |
| **X.4 Case Study: [Location]** | 结尾嵌入案例验证，选择真实地点 | **#2401102**: "6.4 Application: Temple of Literature in Hanoi, Vietnam" |

---

## E题 Model 标准结构模板

```
4 Model I: [Task1模型全称]

4.1 Data Collection
   - 数据来源: World Bank / NOAA / FAO...
   - 数据描述: 时间范围, 地理范围

4.2 [Risk Analysis / Indicator Selection]
   4.2.1 [Sub-indicator 1]
   4.2.2 [Sub-indicator 2]
   4.2.3 [Calculation Method]

4.3 [Weight Calculation / Model Establishment]
   - AHP构建判断矩阵
   - EWM客观验证
   - 综合权重 = α×AHP + (1-α)×EWM

4.4 Application / Case Study
   - 选择真实地点: Mumbai / Tokyo / Vietnam...
   - 计算结果: Score = X.XX
   - 结论与建议

5 Model II: [Task2模型全称]
   (结构同上)
```

---

## E题 vs C题 Model Development 差异

| 对比维度 | C题（预测类） | E题（决策/评估类） |
|----------|--------------|-------------------|
| **模型类型** | ARIMA/LSTM/XGBoost预测模型 | **AHP+EWM+TOPSIS评价体系** |
| **核心流程** | 特征工程→模型训练→预测 | **指标选取→权重计算→综合评价** |
| **结果形式** | R², RMSE, MAE | **Score, Risk Level, Ranking** |
| **案例验证** | 测试集验证 | **真实地点案例**(Mumbai, Tokyo, Hanoi) |
| **决策输出** | 预测值 | **是否承保/是否开发/保护等级** |

# E题 Sensitivity & Robustness Analysis 写作模板

## 统计数据

| 统计项 | 数据 |
|--------|------|
| Sensitivity Analysis | 23次 |
| Robustness Analysis | 9次 |
| 章节位置 | 通常在Model后、S&W前 |

---

## Sensitivity & Robustness 结构分析表（基于原文句式提取）

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **开篇句式** | "Finally, we analyze/perform/conduct the sensitivity analysis..." - **"Finally"开头是主流** | **#2400860 (2024)**: "Finally, we analyze the sensitivity and robustness of our models" |
| **Sensitivity Analysis** | 变化关键参数(0~20%, ±10%) → 观察结果变化 → 得出结论 | **#2407414 (2024)**: "altering the compensation ratio from 0 to 20 percent resulted in a marginal modification in the POR value, **confined to a narrow band of 2 percent**" |
| **Robustness Analysis** | 结论句式："the model is robust to/against [perturbations/changes]" / "the slight error will **not affect** the result" | **#2400860**: "the **slight error** of the risk factor calculation will **not affect** the models' result, which **verifies the robustness**" |
| **结论句式** | "Therefore, the model is robust to changes in [X]" / "which verifies the sensitivity and robustness" | **#2407414**: "Therefore, the model is robust to changes in a single indicator" |

---

## Sensitivity Analysis 模板（基于原文句式）

```
7 Sensitivity and Robustness Analysis

Finally, we analyze the sensitivity and robustness of our models.

7.1 Sensitivity Analysis

To test the sensitivity of our model, we alter the [compensation ratio / key parameter]
from 0 to 20 percent (or by ±10%, ±20%).

Result: The modification in the [POR value / output] is confined to a narrow band of [X] percent.

Conclusion: The model is [not sensitive to / sensitive to] [parameter name].

7.2 Robustness Analysis

To verify the robustness, we introduce slight errors to the [risk factor / input data].

Result: The slight error of the [X] calculation will not affect the models' result.

Conclusion: Therefore, the model is robust to changes in [single indicator / data perturbations],
which verifies the robustness of our models.
```
—— **综合#2400860, #2407414 (2024)原文**

| 对比维度 | C题（预测类） | E题（决策/评估类） |
|----------|--------------|-------------------|
| **参数类型** | 模型超参数 | **权重、风险因子、市场回报率** |
| **变化幅度** | ±5%, ±10% | **±10%, ±20%** |
| **结果解读** | 对预测精度影响 | **对决策结果影响**(premium, risk level) |
| **稳健性标准** | 波动范围 | **“结果不受影响”“波动<3%”** |

---

# E题 Strength and Weakness 写作模板

## 统计数据

| 统计项 | 数据 |
|--------|------|
| S&W章节 | 9篇有独立章节 |
| 标题格式 | "8 Model Evaluation" 或 "Strength and Weakness" |

---

## Strength and Weakness 结构分析表（基于原文关键词统计）

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **8.1 Strengths** | 按频率：①**comprehensive**(22次) ②practical(8次) ③accurate(8次) ④innovative(4次) ⑤flexible(3次)；用bullet列表，每点1-2句 | **#2400860**: "• The model provides a **comprehensive** assessment... • **Practical** decision-making framework..." |
| **8.2 Weaknesses** | 主要关注：①simplification(12次) ②assumption limitations；句式："The model simplifies..." / "The assumption may not hold..." | **#2410578**: "• The model simplifies complex scenarios... • The assumption of stable market may not hold..." |

---

## Strength and Weakness 模板（基于原文关键词）

```
8 Model Evaluation  或  9 Strengths and Weaknesses

8.1 Strengths

• Comprehensive Assessment: Our model provides a comprehensive evaluation considering [N] dimensions.
• Practical Application: The model offers practical and actionable recommendations for [stakeholders].
• Accurate Results: The model achieves accurate results validated by real-world cases in [location].
• Innovative Integration: We innovatively combine [AHP] and [EWM] to balance subjective and objective weighting.

8.2 Weaknesses

• Simplification: The model simplifies complex real-world scenarios which may overlook some factors.
• Assumption Constraints: The assumption of [stable market / rational behavior] may not hold during extreme events.
• Data Dependency: The model relies on historical data, which may not fully capture future trends.
```
—— **综合#2400860, #2410578 原文**

---

# E题 References 写作模板

## 统计数据

| 统计项 | 数据 |
|--------|------|
| 引用标记 | 40次出现 |
| 平均每篇 | 6-10篇参考文献 |

---

## References 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **References** | ①数量: 6-10篇；②格式: IEEE或APA统一；③类型: 官方报告 + 领域经典 + 近年研究 | **#2401102**: "[1] Swiss Re Institute Report... [2] Munich Reinsurance Data... [3] IPCC Climate Report..." |
| **引用类型** | 官方机构报告(World Bank, IPCC, Munich Re)、学术论文、行业数据 | - |

---

# E题 Appendix 写作模板

## Appendix 结构分析表

| 目录章节 | 共性行文思路（固定逻辑） | O奖案例示例 |
|----------|--------------------------|-------------|
| **Appendix A: Code** | 核心模型代码 + 注释；Python/MATLAB | AHP权重计算代码、TOPSIS排序代码 |
| **Appendix B: Data** | 补充数据表、完整指标数据 | 各地区风险指标完整表 |
| **Appendix C: Calculation** | 公式推导、矩阵计算过程 | AHP判断矩阵、一致性检验 |
| **Letter/Memo** | 给决策者的建议信(非E题必须) | 给社区领导的建築保护建议 |

---

## E题 vs C题 Appendix 差异

| 对比维度 | C题（预测类） | E题（决策/评估类） |
|----------|--------------|-------------------|
| **代码类型** | ARIMA/LSTM/XGBoost | **AHP权重计算、TOPSIS排序** |
| **数据表** | 时间序列完整数据 | **多维指标数据表** |
| **公式推导** | GMM/时间序列分解 | **AHP矩阵、熔合权重公式** |
| **Letter** | 预测报告 | **决策建议信** |
