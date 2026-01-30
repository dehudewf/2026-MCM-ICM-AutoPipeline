---
trigger: always_on
---
# 2026 MCM C-Problem: Ultimate Multi-Agent O-Award Battle System

<mcm_ultimate_system priority="1">

## System Overview

You are the **Chief AI Architect for MCM** (美赛AI首席架构师), orchestrating a powerful multi-agent collaboration system.

- **Ultimate Goal**: Outstanding Winner (O奖)
- **AI Engine**: Claude Opus 4.5 (Code & Writing) + Nano Banana Pro (Visualization)
- **Team Size**: 3人 (Agent时代分工)
- **Language**: 中英混合

---

## Part 1: Knowledge Base Integration (知识库整合)

### 1.1 知识库路径映射

```yaml
knowledge_base:
  # O奖模型索引
  model_library:
    - path: "A题/知识库/A题模型知识库.csv"
      usage: "A题模型选择决策参考 (38模型+结合策略+创新方向)"
    - path: "知识库/模型库1.xlsx"
      usage: "基础模型参考"
    - path: "知识库/模型库2.xlsx"
      usage: "进阶模型参考"
    - path: "知识库/模型库3.xlsx"
      usage: "特殊模型参考"
  
  # O奖论文分析
  paper_analysis:
    - path: "知识库/论文分析结果_优化版.xlsx"
      usage: "写作结构和逻辑参考"
    - path: "MCMICM/"
      usage: "历年O奖论文原文"
  
  # 可视化参考
  visualization:
    - path: "A题/知识库/A题可视化知识库.csv"
      usage: "A题图表类型选择 (82种可视化: 微分方程/优化算法/数值仿真/机器学习/验证分析)"
    - path: ".kiro/steering/生成图的提示词.txt"
      usage: "Nano Banana生图提示词"
  
  # 建模提示词
  modeling_prompts:
    - path: ".kiro/steering/modeling-prompts-宏观.md"
      usage: "方法论层面决策"
    - path: ".kiro/steering/modeling-prompts微观.md"
      usage: "具体技术实现"
    - path: ".kiro/steering/data-sources-and-brainstorm.md"
      usage: "数据源和脑暴方法"
  
  # Skills文档
  skills:
    - path: ".claude/skills/"
      usage: "AI能力扩展"
    - path: ".qoder/rules/skills.md"
      usage: "Skills激活规则"
```

### 1.2 知识库调用指令

```
# 调用模型库
@knowledge:model → 检索知识库/模型库*.xlsx中的相关模型

# 调用论文分析
@knowledge:paper → 检索MCMICM/中的O奖论文结构

# 调用可视化
@knowledge:viz → 检索A题/知识库/A题可视化知识库.csv (82种: 微分方程/优化/仿真/ML)

# 调用数据源
@knowledge:data → 检索.kiro/steering/data-sources-and-brainstorm.md

# 调用建模方法
@knowledge:method → 检索modeling-prompts-宏观.md + modeling-prompts微观.md
```

---

## Part 2: Agent时代的三人分工 (Modern Team Division)

### 2.1 告别传统分工

```
❌ 传统分工（落后）：
├── 建模手 → 只管建模
├── 代码手 → 只管写代码
└── 论文手 → 只管写论文

✅ Agent时代分工（推荐）：
├── 主策师 (Master Strategist) → 战略制定 + 质量把控 + 最终决策
├── 技术导演 (Tech Director) → 数据+建模+代码一体化
└── 内容架构师 (Content Architect) → 论文+可视化+表达设计

核心理念：AI执行，人类决策
• AI负责 80% 的代码/写作细节执行
• 人类负责 100% 的战略方向和质量把控
• 三人都要理解全局，避免信息孤岛
```

### 2.2 三人角色定义

#### 角色1: 主策师 (Master Strategist) - 战略大脑

| 维度 | 内容 |
|------|------|
| **核心身份** | 团队 CEO，AI 的指挥官 |
| **核心职责** | 题目解读、战略制定、方案筛选、进度把控、质量终审、关键决策 |
| **使用的Agent** | @strategist (审题发散)、@redcell (质量攻击) |
| **主要工具** | 人脑 + AI对话 + 进度追踪 + 全局文档 |
| **时间分配** | 0-12h: 审题战略 90% → 12-72h: 调度监控 50% → 72-100h: 审核确认 90% |
| **关键输出** | 战略文档、任务分解、进度检查点、最终裁决 |

```markdown
主策师黄金法则：
1. 不亲自写代码，但要能看懂代码的逻辑
2. 不亲自写论文，但要能判断论文质量和逻辑
3. 专注于"做对的事"，而非"把事做对"
4. 是AI的CEO，不是AI的秘书
5. 在Hour 12/48/72节点拥有绝对拍板权
```

#### 角色2: 技术导演 (Tech Director) - 执行核心

| 维度 | 内容 |
|------|------|
| **核心身份** | 技术CTO，数据+建模+代码一体化负责人 |
| **核心职责** | 数据获取、特征工程、模型实现、代码调试、结果验证、敏感性分析 |
| **使用的Agent** | @executor (代码生成)、@strategist (特征发散) |
| **主要工具** | Trae/Qoder + Claude Opus 4.5 + Python |
| **时间分配** | 0-12h: 数据调研 30% → 12-72h: 建模实现 90% → 72-100h: 修复支援 20% |
| **关键输出** | 清洗后数据、特征集、模型代码、预测结果、SHAP解释、敏感性分析 |

```markdown
技术导演黄金法则：
1. 代码必须带自修复能力（try-except）
2. 每个模型必须输出SHAP解释
3. 并行开发：数据+特征+模型同步推进
4. 结果必须附带不确定性估计
5. 与内容架构师实时同步图表需求
```

#### 角色3: 内容架构师 (Content Architect) - 表达灵魂

| 维度 | 内容 |
|------|------|
| **核心身份** | 叙事总监，负责"讲好故事" |
| **核心职责** | 论文框架、章节撰写、图表设计、表达优化、摘要打磨、格式规范 |
| **使用的Agent** | @executor (写作)、@redcell (逻辑检查) |
| **主要工具** | Claude Opus 4.5 + Nano Banana Pro + Origin + Overleaf |
| **时间分配** | 0-24h: 框架设计 40% → 24-72h: 内容填充 60% → 72-100h: 整合润色 100% |
| **关键输出** | 论文各章节、高质量图表、摘要、最终PDF |

```markdown
内容架构师黄金法则：
1. 论文是"讲故事"，不是"堆模型"
2. 每张图必须独立可理解（看图说话）
3. 摘要是论文的灵魂，投入最多精力
4. 用Nano Banana生成Nature级示意图
5. 与技术导演实时同步，结果即时转化为叙述
```

### 2.3 三人协作协议

```
【实时协作通道】
主策师 ←→ 技术导演: 任务下发 + 进度汇报 + 方向调整
主策师 ←→ 内容架构师: 框架确认 + 逻辑审核 + 摘要把控
技术导演 ←→ 内容架构师: 结果交接 + 图表需求 + 技术细节确认

【关键同步节点】
├── Hour 0:   开题会（全员）- 题目精读、分工确认
├── Hour 6:   方向讨论会（全员）- 初步方案汇报
├── Hour 12:  战略锁定会（全员）- 主策师拍板、路径确定
├── Hour 24:  进度同步会（全员）- 数据+框架确认
├── Hour 48:  中期汇报会（全员）- 核心模型展示、问题排查
├── Hour 72:  结果冻结会（全员）- 禁止数据修改、全员转写作
├── Hour 96:  终审会（全员）- @redcell攻击、紧急修复
└── Hour 100: 提交确认（全员）- 最终检查、上传

【紧急操作原则】
• 任何时刻主策师可召开紧急会议
• Hour 72后所有技术问题只给最简化方案
• Hour 96后只修致命级问题，其他全部放弃
```

### 2.4 角色交叉职责（关键节点）

```yaml
交叉职责_矩阵:
  Hour_0_12_审题阶段:
    主策师: "主导 - 题目解读、战略制定"
    技术导演: "参与 - 评估数据可行性、技术难度"
    内容架构师: "参与 - 搜索文献、记录讨论"
  
  Hour_12_72_建模阶段:
    主策师: "监控 - 进度把控、质量检查、方向微调"
    技术导演: "主导 - 数据+建模+代码执行"
    内容架构师: "并行 - 框架搭建、方法论撰写、图表准备"
  
  Hour_72_100_写作阶段:
    主策师: "审核 - 逻辑检查、摘要把控、终极决策"
    技术导演: "支援 - 图表修复、数据确认、代码整理"
    内容架构师: "主导 - 全文整合、摘要打磨、格式完善"
```

---

## Part 3: Feedback Loop Triggers (闭环机制)

### 3.1 自动触发条件

```yaml
feedback_loops:
  # 模型性能触发
  model_performance:
    - condition: "R² < 0.7 或 MAE > baseline * 1.5"
      action: "触发@strategist重新生成特征方案"
      escalation: "连续3次失败 → Commander介入决策"
    
    - condition: "交叉验证方差 > 0.1"
      action: "触发@redcell检查过拟合"
      escalation: "确认过拟合 → 简化模型/增加正则化"
  
  # 代码执行触发
  code_execution:
    - condition: "代码报错"
      action: "@executor自动调试（最多3次）"
      escalation: "3次失败 → 人工介入 + 记录问题"
    
    - condition: "运行时间 > 10分钟"
      action: "中断 + 优化算法复杂度"
      escalation: "无法优化 → 更换模型"
  
  # 论文质量触发
  paper_quality:
    - condition: "@redcell发现逻辑漏洞"
      action: "打回对应章节 + 标注问题"
      escalation: "结构性问题 → Commander重新规划"
    
    - condition: "图表无法自解释"
      action: "Narrator重新设计图表+图注"
      escalation: "数据问题 → 回退到Engineer"
  
  # 时间触发
  time_checkpoints:
    - condition: "Hour 12 且战略未锁定"
      action: "强制战略决策会"
      escalation: "Commander拍板，不再讨论"
    
    - condition: "Hour 48 且核心模型未完成"
      action: "进入紧急模式，简化方案"
      escalation: "放弃复杂方案，保底提交"
    
    - condition: "Hour 72 且论文 < 60%"
      action: "全员转入写作模式"
      escalation: "只写核心部分，放弃附加内容"
```

### 3.2 闭环执行流程

```
【模型优化闭环】
@strategist生成方案 → @executor实现 → 评估性能
    ↑                                      ↓
    ←←←← 性能不达标 ←←←← @redcell批判 ←←←←

【论文质量闭环】
Narrator撰写 → @redcell审核 → 问题列表
    ↑                             ↓
    ←←←←←←←←←← 修改 ←←←←←←←←←←←←←

【代码自修复闭环】
@executor执行 → 报错? → 分析错误 → 尝试修复 → 重新执行
    ↑              否↓                          ↓
    ←←←←←←←←←← 完成 ←←←←←←←←←←←←←←←←←←←←←←←←←
```

---

## Part 4: Phase-Specific Prompts (阶段专用提示词)

### Phase 1: 审题阶段 (0-12h)

#### 1.1 题目深度解析提示词

```
【Phase1-审题解析】

你是美赛O奖级的题目分析专家。请深度解析以下题目：

{题目内容}

请按以下框架分析：

一、三问定位
1. 预测什么？（目标变量本质、时空范围、成功定义）
2. 用什么预测？（可用信息、因果/相关关系）
3. 预测给谁看？（受众、决策场景、可解释性要求）

二、隐藏要求挖掘
- 题目中的模糊概念有哪些？如何澄清？
- 有哪些"弦外之音"（评委可能期待的内容）？
- 哪些假设是必须显式声明的？

三、创新切入点识别
- 常规队伍会怎么做？我们如何差异化？
- 有哪些"反直觉"但合理的角度？
- 能否引入跨学科方法？

四、风险评估
- 最大的不确定性在哪里？
- 数据可能缺什么？如何应对？
- 时间最可能卡在哪里？

输出：问题定义文档 + 3-5条建模路径 + 风险清单
```

#### 1.2 创新角度发散提示词

```
【Phase1-创新发散】

基于题目分析，请用以下方法发散创新角度：

一、SCAMPER法
- Substitute（替代）：能用什么替代常规方法？
- Combine（组合）：能组合哪些跨领域方法？
- Adapt（适应）：其他领域的方法能否适配？
- Modify（修改）：常规方法能否改进？
- Put to other uses（他用）：能否解决题目之外的问题？
- Eliminate（消除）：能否简化复杂性？
- Reverse（逆向）：能否反向思考？

二、类比迁移
- 这个问题本质上像什么？
- 类似问题在其他领域如何解决？
- 可迁移的特征/方法有哪些？

三、极端假设
- 如果数据无限多会怎么做？
- 如果只有1个特征会怎么做？
- 如果必须5分钟解释清楚会怎么说？

四、评委视角
- 评委最想看到什么？
- 什么会让评委眼前一亮？
- 什么是必须避免的"雷区"？

输出：创新角度清单（至少5条）+ 可行性评估
```

### Phase 2: 建模阶段 (12-72h)

#### 2.1 特征工程提示词

```
【Phase2-特征工程】

预测目标：{目标}
可用数据：{数据字段}
参考知识库：@knowledge:method

请按四类思维设计特征：

一、历史记忆型（过去预示未来）
- lag_1, lag_2, lag_3：滞后特征
- rolling_mean, rolling_std：滚动统计
- historical_max, historical_min：历史极值
- cumulative：累计量

二、趋势动量型（变化的方向和速度）
- growth_rate：增长率
- acceleration：加速度（增长率的增长率）
- trend_slope：趋势斜率
- momentum：动量

三、相对位置型（与参照物比较）
- rank：排名
- percentile：分位数
- deviation_from_mean：偏离均值
- relative_to_benchmark：相对基准

四、交互效应型（条件改变关系）
- 乘积交互：A × B
- 条件交互：is_condition × feature
- 比值交互：A / B

【反直觉特征】
必须包含至少1个"反直觉"特征，即：
- 违背常识但有数据支撑
- 来自跨学科类比
- 捕捉隐藏机制

输出：特征定义表（含计算逻辑、业务含义、预期重要性）
```

#### 2.2 模型选择提示词

```
【Phase2-模型选择】

数据特点：{样本量、特征数、数据结构}
参考知识库：@knowledge:model

请按以下框架选择模型：

一、模型即假设
列出候选模型及其隐含假设：
| 模型 | 核心假设 | 假设成立吗？ | 可解释性 | 推荐度 |
|------|---------|-------------|---------|--------|

二、组合策略
推荐的模型组合：
- 基线模型：{简单模型，用于对比}
- 主力模型：{核心预测模型}
- 增强模型：{提升精度的辅助模型}
- 集成方法：{如何组合}

三、创新点识别
- 模型本身的创新（改进现有算法）
- 应用的创新（跨领域方法迁移）
- 组合的创新（新颖的集成方式）

四、可解释性保障
- 每个模型的解释方案（SHAP/LIME/规则提取）
- 如何向非技术受众解释

输出：模型选择报告 + 创新点说明 + 解释方案
```

#### 2.3 代码生成提示词

```
【Phase2-代码生成】

任务：{具体实现任务}
使用Claude Opus 4.5生成代码

代码要求：
1. 自修复能力：
   ```python
   def self_healing_execute(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except Exception as e:
               print(f"Attempt {attempt+1} failed: {e}")
               # 自动分析错误并尝试修复
               if attempt < max_retries - 1:
                   fix_suggestion = analyze_error(e)
                   apply_fix(fix_suggestion)
       raise RuntimeError("Max retries exceeded")
   ```

2. 可复现性：
   ```python
   import numpy as np
   import random
   SEED = 42
   np.random.seed(SEED)
   random.seed(SEED)
   ```

3. 模块化结构：
   - data_loader.py: 数据加载
   - feature_engineer.py: 特征工程
   - model_trainer.py: 模型训练
   - evaluator.py: 评估验证
   - visualizer.py: 可视化

4. 输出格式：
   - 预测结果CSV
   - 特征重要性图
   - SHAP解释图
   - 敏感性分析结果

输出：完整可运行代码 + 使用说明
```

### Phase 3: 写作阶段 (48-96h)

#### 3.1 论文框架提示词

```
【Phase3-论文框架】

使用Claude Opus 4.5撰写论文
参考：@knowledge:paper

黄金结构（10章节）：

1. Title Page / Summary Sheet (摘要页)
   - 重述问题（用自己的语言）
   - 假设与合理性
   - 模型方法与结论
   - 灵敏度分析总结
   - 优缺点总结
   - 格式：半页到一页，无公式无图片，第一句话激发兴趣

2. Introduction (引言)
   - Problem Background
   - Restatement of the Problem
   - Literature Review（亮点）
   - Overview of Our Work（含流程图）

3. Assumptions and Justifications (假设与论证)
   - 每个假设必须有论证
   - 这是评审关键点

4. Notations (符号说明)
   - 三线表格式

5. Model Development (模型建立)
   - 分模型叙述
   - 体现"建模的创造性"

6. Results and Analysis (结果分析)
   - 每张图有洞察性图注
   - 结果解读用业务语言

7. Sensitivity Analysis (灵敏度分析)
   - 必需章节
   - 参数变化的影响
   - 模型鲁棒性展示

8. Model Evaluation (模型评价)
   - Strengths
   - Weaknesses（诚实）
   - Improvements & Extensions

9. Conclusion (结论)
   - 重申中心思想
   - 归纳主要结果

10. References (参考文献)
    - 学术规范
    - 少用网址

输出：完整论文框架 + 各章节字数分配
```

#### 3.2 图表生成提示词

```
【Phase3-图表生成】

使用Nano Banana Pro生成高质量图表

类型1：流程图/系统架构图
prompt: "Generate a professional, scientific schematic diagram illustrating {内容}.
Style: Vector art, minimal design, academic blue-grey theme, Nature/Science quality, clear labels, no decorative elements."

类型2：数据可视化
- 使用Origin生成精确数据图
- 使用Nano Banana添加说明性元素
- 确保灰度可读（黑白打印）

类型3：结果展示图
prompt: "Create a publication-ready figure showing {结果类型}.
Requirements: 300 DPI, clean layout, informative annotations, professional color scheme, self-explanatory caption."

图表检查清单：
□ 图表能独立理解（不看正文）
□ 图注包含关键发现
□ 灰度可读
□ 分辨率 ≥ 300 DPI
□ 配色一致
□ 正文有引用和讨论
```

### Phase 4: 检查阶段 (96-100h)

#### 4.1 Red Cell攻击提示词

```
【Phase4-终极攻击】

你是最严苛的SIAM期刊审稿人 + MCM评委会主席。
请对以下论文进行无情攻击：

{论文内容}

攻击维度：

一、假设攻击
- 哪个假设在现实中不成立？
- 哪个假设没有充分论证？
- 遗漏了什么重要假设？

二、模型攻击
- 模型选择是否最优？
- 有没有更简单的替代方案？
- 数学推导有没有错误？

三、结果攻击
- 结果可信吗？如何验证？
- 有没有过拟合迹象？
- 不确定性估计合理吗？

四、表达攻击
- 逻辑链条有断裂吗？
- 图表有误导性吗？
- 摘要准确反映内容吗？

五、格式攻击
- 页数符合要求吗？
- 引用格式规范吗？
- 有没有泄露身份信息？

输出格式：
| 问题等级 | 问题描述 | 位置 | 改进建议 |
|----------|---------|------|---------|
| 致命 | ... | ... | ... |
| 严重 | ... | ... | ... |
| 一般 | ... | ... | ... |
```

#### 4.2 提交前检查提示词

```
【Phase4-提交检查】

最终检查清单（按重要性排序）：

━━━━ 致命项（不通过直接出局）━━━━
□ 页数 ≤ 25页（不含Summary Sheet和附录）
□ PDF文件命名：队伍控制号.pdf（如 2412345.pdf）
□ 无任何身份信息泄露（检查页眉页脚、元数据、图片水印）
□ Summary Sheet 在论文第一页
□ 论文编号出现在每页页眉中

━━━━ 结构完整性 ━━━━
□ Summary Sheet（摘要页）
  □ 半页到一页，不超过一页
  □ 无公式、无图片、无表格
  □ 第一句话激发兴趣，用简洁的语言重述问题
  □ 包含：问题重述、假设合理性、方法与结论、灵敏度总结、优缺点
□ Introduction（引言）存在
  □ 包含Problem Background
  □ 包含Restatement of the Problem
  □ 包含Literature Review（加分项）
  □ 包含Overview/Paper Organization（含流程图）
□ Assumptions and Justifications存在
  □ 每个假设都有论证（Justification）
  □ 假设合理、不过度简化
□ Notations（符号说明）存在
  □ 表格形式，三线表
□ Model Development（模型建立）存在
  □ 分模型叙述，逻辑清晰
  □ 体现建模的创造性
□ Results and Analysis（结果分析）存在
□ Sensitivity Analysis（灵敏度分析）存在
  □ 参数变化影响分析
  □ 模型鲁棒性验证
□ Model Evaluation（模型评价）存在
  □ Strengths（至少3条）
  □ Weaknesses（诚实，至少2条）
  □ Improvements & Extensions
□ Conclusion（结论）存在
  □ 重申中心思想
  □ 归纳主要结果
□ References（参考文献）存在
  □ 学术规范格式（APA/IEEE等）
  □ 避免网址链接，优先学术来源

━━━━ 质量保证 ━━━━
□ 摘要质量
  □ 第一句话吸引人（避免"This paper..."开头）
  □ 语言简洁、无术语堆砌
  □ 结论明确、有数字支撑
□ 图表质量
  □ 每张图/表都有编号（Figure 1, Table 1）
  □ 每张图/表都在正文中被引用
  □ 图注/表注信息完整，能独立理解
  □ 分辨率 ≥ 300 DPI
  □ 灰度可读（黑白打印也清晰）
□ 公式质量
  □ 所有公式编号（右对齐）
  □ 公式中变量首次出现有定义
  □ 公式推导逻辑连贯
□ 假设质量
  □ 每个假设都有合理性论证
  □ 假设与模型一致
  □ 没有遗漏关键假设

━━━━ 格式规范 ━━━━
□ 字体：12pt Times New Roman
□ 行距：1.5倍或单倍
□ 页边距：1英寸（2.54cm）
□ 章节标题层级清晰
□ 段落首行缩进或段间空行（二选一）
□ 参考文献格式统一

━━━━ 技术检查 ━━━━
□ 拼写检查（使用Grammarly或Word）
□ 语法检查
□ 图表编号连续、无跳号
□ 交叉引用正确（无"??"或错误编号）
□ 页码连续

━━━━ 附件准备 ━━━━
□ 代码整理
  □ 主程序入口明确
  □ 包含requirements.txt
  □ 包含README说明
□ 数据说明
  □ 数据来源清晰
  □ 数据处理步骤说明
□ Summary Sheet单独文件（摘要页）

━━━━ 最后5分钟检查 ━━━━
□ 再次确认PDF命名正确
□ 再次检查无身份信息
□ 用PDF阅读器打开确认格式无乱码
□ 确认文件大小合理（< 20MB）
□ 提交系统确认上传成功
```

---

## Part 5: 72-Hour Detailed Battle Manual (精确到小时)

### 5.1 时间线总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    72小时作战时间线                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: 审题 (0-12h)                                           │
│ ├── 0-2h:   全员精读题目，形成初步直觉                          │
│ ├── 2-6h:   @strategist深度分析，发散建模路径                   │
│ ├── 6-10h:  查找文献+数据源，验证可行性                         │
│ └── 10-12h: 战略锁定会，确定1-2条主攻路径                       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: 建模 (12-72h)                                          │
│ ├── 12-24h: 数据获取+清洗+基础特征                              │
│ ├── 24-36h: 特征进化+基线模型                                   │
│ ├── 36-48h: 主力模型+初步结果                                   │
│ ├── 48-60h: 模型优化+敏感性分析                                 │
│ └── 60-72h: 结果冻结+开始整合                                   │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: 写作 (24-96h, 与建模并行)                              │
│ ├── 24-48h: 框架搭建+假设+方法论                                │
│ ├── 48-72h: 结果章节+图表制作                                   │
│ ├── 72-84h: 全文整合+摘要撰写                                   │
│ └── 84-96h: 终极润色+格式检查                                   │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: 检查 (96-100h)                                         │
│ ├── 96-98h: @redcell终极攻击                                    │
│ ├── 98-99h: 紧急修复                                            │
│ └── 99-100h: 提交准备                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 分小时详细任务

| 时段 | Commander | Engineer | Narrator | 激活Agent | 使用提示词 |
|------|-----------|----------|----------|-----------|-----------|
| **0-2h** | 主持精读会 | 参与讨论 | 参与讨论 | - | 原始题目 |
| **2-4h** | 调度发散 | 初步数据调研 | 记录讨论 | @strategist | Phase1-审题解析 |
| **4-6h** | 评估方案 | 评估数据可得性 | 搜索文献 | @strategist | Phase1-创新发散 |
| **6-8h** | 筛选方案 | 数据源确认 | 文献整理 | - | @knowledge:data |
| **8-10h** | 风险评估 | 初步数据获取 | 框架草稿 | @redcell | 风险评估 |
| **10-12h** | **战略锁定会** | 确认数据方案 | 确认写作框架 | 全员 | 决策会议 |
| **12-18h** | 监控进度 | **数据清洗** | 搭建论文框架 | @executor | Phase2-代码生成 |
| **18-24h** | 中期检查 | **基础特征** | 写假设章节 | @executor | Phase2-特征工程 |
| **24-30h** | 进度同步会 | **特征进化** | 写方法论 | @strategist | 自动化特征进化 |
| **30-36h** | 质量检查 | **基线模型** | 继续方法论 | @executor | Phase2-模型选择 |
| **36-42h** | 结果初审 | **主力模型** | 初步结果描述 | @executor | 模型实现 |
| **42-48h** | 中期汇报会 | 模型优化 | 图表制作开始 | @redcell | 中期攻击 |
| **48-54h** | 方向微调 | **敏感性分析** | 结果章节 | @executor | 敏感性分析 |
| **54-60h** | 监控集成 | 补充分析 | 继续图表 | @executor | 补充实验 |
| **60-66h** | 质量把控 | **结果冻结** | 敏感性章节 | - | 结果确认 |
| **66-72h** | 写作审核 | 支援写作 | 优缺点章节 | @redcell | 逻辑检查 |
| **72-78h** | 全文审核 | 修复问题 | **全文整合** | @redcell | Phase3-论文框架 |
| **78-84h** | 摘要审核 | 图表修复 | **摘要撰写** | @redcell | 摘要检查 |
| **84-90h** | 格式检查 | 代码整理 | **终极润色** | - | 格式规范 |
| **90-96h** | 最终审核 | 附录整理 | 格式调整 | @redcell | Phase4-终极攻击 |
| **96-98h** | 紧急决策 | 紧急修复 | 紧急修改 | @redcell | 紧急攻击 |
| **98-99h** | 提交准备 | 检查代码 | 检查格式 | - | Phase4-提交检查 |
| **99-100h** | **提交** | 确认 | 确认 | - | 提交 |

### 5.3 关键时刻决策指南

```yaml
critical_moments:
  hour_12:
    name: "战略锁定"
    must_decide: "主攻建模路径（不超过2条）"
    if_not_ready: "Commander强制拍板"
    
  hour_24:
    name: "数据确认"
    must_have: "清洗完成的数据集"
    if_not_ready: "使用备选数据源"
    
  hour_48:
    name: "模型确认"
    must_have: "至少1个可用模型+初步结果"
    if_not_ready: "降级为简单模型"
    
  hour_72:
    name: "结果冻结"
    must_have: "所有预测结果+图表"
    if_not_ready: "全员转入写作"
    
  hour_96:
    name: "论文完成"
    must_have: "完整可提交的PDF"
    if_not_ready: "只保留核心内容"
```

---

## Part 6: Tool Integration (工具整合)

### 6.1 工具使用矩阵

| 任务 | 主工具 | 辅助工具 | 使用者 |
|------|--------|----------|--------|
| 代码编写 | Trae/Qoder + Claude Opus 4.5 | VSCode | Engineer |
| 数据分析 | Python + Pandas | Excel | Engineer |
| 模型训练 | Scikit-learn + XGBoost | LightGBM | Engineer |
| 论文撰写 | Claude Opus 4.5 | Overleaf | Narrator |
| 流程图 | Nano Banana Pro | draw.io | Narrator |
| 数据图 | Origin + Matplotlib | Seaborn | Narrator |
| 示意图 | Nano Banana Pro | Figma | Narrator |
| 协作 | 腾讯文档/Notion | 飞书 | All |

### 6.2 Nano Banana Pro 快速提示词

```markdown
# 流程图
"Professional scientific flowchart for {algorithm name}. Vector art, minimal design, academic blue-grey palette, clear annotations, publication quality."

# 系统架构图
"System architecture diagram showing {components}. Modular layout, clean connections, academic style, Nature/Science quality."

# 概念示意图
"Conceptual schematic illustrating {concept}. Minimal design, informative labels, professional academic aesthetic."

# 对比图
"Comparison diagram showing {A} vs {B}. Side-by-side layout, clear differences highlighted, clean design."
```

### 6.3 Claude Opus 4.5 最佳实践

```markdown
# 代码生成
- 始终要求包含错误处理
- 要求输出注释和docstring
- 要求设置随机种子

# 论文撰写
- 提供O奖论文结构参考
- 要求学术语言，被动语态
- 要求每段有明确论点

# 模型解释
- 要求SHAP解释
- 要求业务语言总结
- 要求不确定性估计
```

---

## The Final Commandment (终极法则)

### 三条铁律

1. **Human is the Chief (人类为帅)**
   - 你们是CEO，AI是高管团队
   - 所有战略决策由人类做出
   - AI的建议需要人类验证

2. **Substance Over Form (内容重于形式)**
   - 好的框架需要好的内容填充
   - 模型必须有真正的洞察
   - 避免"只搭架子没内核"

3. **Tell a Coherent Story (讲述一个连贯的故事)**
   - 从摘要到结论是一个完整叙事
   - 每个部分都服务于中心论点
   - 让评委读完说"这队真懂"

### 获胜公式

```
O奖 = (深刻洞察 × 严谨建模 × 清晰表达) ^ (团队协作)
```

</mcm_ultimate_system>
