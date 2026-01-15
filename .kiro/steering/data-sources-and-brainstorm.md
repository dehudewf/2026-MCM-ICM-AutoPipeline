# 数据来源与脑暴方法指南

> 本文档提供建模任务中数据获取、尺度统一和系统性脑暴的完整参考。

---

## 一、数据来源大全

### 1.1 综合性国际数据平台

| 平台 | 网址 | 数据类型 | 可获得性 | 特点 |
|------|------|---------|---------|------|
| **世界银行** | data.worldbank.org | GDP、人口、教育、健康、基础设施 | ⭐⭐⭐⭐⭐ | 覆盖200+国家，时间跨度50+年，API支持 |
| **联合国数据** | data.un.org | 人口、发展指标、SDG指标 | ⭐⭐⭐⭐⭐ | 官方权威，多语言 |
| **IMF数据** | imf.org/en/Data | 经济、金融、汇率、贸易 | ⭐⭐⭐⭐⭐ | 宏观经济首选 |
| **OECD数据** | data.oecd.org | 发达国家经济社会指标 | ⭐⭐⭐⭐ | 数据质量高，更新及时 |
| **Our World in Data** | ourworldindata.org | 综合社会经济数据 | ⭐⭐⭐⭐⭐ | 可视化好，直接下载CSV |
| **Gapminder** | gapminder.org/data | 发展指标、健康、教育 | ⭐⭐⭐⭐⭐ | 数据清洗好，适合教学 |

【搜索指数】
├── 百度指数: index.baidu.com（免费，需登录）
├── Google Trends: trends.google.com（免费）
├── 微信指数: 微信搜一搜（免费）
└── 用法：搜索"奥运"热度 → 代理公众关注度

【卫星/遥感数据】
├── Google Earth Engine: earthengine.google.com（免费，需申请）
├── NASA Earthdata: earthdata.nasa.gov（免费）
├── 中国资源卫星: cresda.com（部分免费）
└── 用法：夜间灯光强度 → 代理经济活动水平

【另类经济指标】
├── 克强指数：铁路货运量 + 工业用电量 + 银行贷款
├── 挖掘机指数：工程机械销量 → 基建活跃度
├── 快递指数：快递业务量 → 消费活跃度
└── 来源：行业协会报告、新闻报道

【学术数据库】
├── Harvard Dataverse: dataverse.harvard.edu
├── Figshare: figshare.com
├── Zenodo: zenodo.org
└── 用法：搜索论文附带的原始数据集

### 1.2 中国数据源

| 平台 | 网址 | 数据类型 | 可获得性 |
|------|------|---------|---------|
| **国家统计局** | stats.gov.cn | 经济、人口、社会、行业 | ⭐⭐⭐⭐⭐ |
| **中国气象数据网** | data.cma.cn | 气象历史数据、气候 | ⭐⭐⭐⭐ |
| **生态环境部** | mee.gov.cn | 空气质量、污染物、环境 | ⭐⭐⭐⭐ |
| **交通运输部** | mot.gov.cn | 交通流量、运输统计 | ⭐⭐⭐ |
| **中国人民银行** | pbc.gov.cn | 金融、货币、利率 | ⭐⭐⭐⭐ |
| **海关总署** | customs.gov.cn | 进出口贸易数据 | ⭐⭐⭐⭐ |
| **国家体育总局** | sport.gov.cn | 体育产业、赛事数据 | ⭐⭐⭐ |

### 1.3 体育/奥运专题数据

| 平台 | 网址 | 数据类型 | 可获得性 |
|------|------|---------|---------|
| **国际奥委会** | olympics.com/ioc | 历届奖牌、参赛国、项目 | ⭐⭐⭐⭐ |
| **Sports Reference** | sports-reference.com/olympics | 详细运动员数据、历史记录 | ⭐⭐⭐⭐⭐ |
| **Olympedia** | olympedia.org | 奥运历史数据库、运动员档案 | ⭐⭐⭐⭐⭐ |
| **Kaggle奥运数据集** | kaggle.com/datasets | 整理好的奥运历史数据 | ⭐⭐⭐⭐⭐ |
| **Wikipedia奥运页面** | wikipedia.org | 奖牌榜、参赛国信息 | ⭐⭐⭐⭐ |

### 1.4 机器学习/竞赛数据平台

| 平台 | 网址 | 特点 |
|------|------|------|
| **Kaggle** | kaggle.com/datasets | 各类竞赛数据，有代码参考 |
| **UCI机器学习库** | archive.ics.uci.edu | 经典ML数据集，学术标准 |
| **天池** | tianchi.aliyun.com | 中文竞赛数据 |
| **和鲸社区** | heywhale.com | 中文数据科学社区 |
| **GitHub Awesome Datasets** | github.com/awesomedata | 数据集合集 |

### 1.5 API接口资源

```python
# 世界银行API（免费，无需注册）
import wbdata
# 获取GDP数据
gdp = wbdata.get_dataframe({"NY.GDP.MKTP.CD": "GDP"}, country="all")

# 或直接请求
# https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json

# OpenWeatherMap（天气，需注册免费key）
# https://api.openweathermap.org/data/2.5/weather?q=Beijing&appid=YOUR_KEY

# Alpha Vantage（金融，需注册免费key）
# https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=YOUR_KEY

# 中国空气质量（免费）
# https://air.cnemc.cn:18007/
```

### 1.6 数据获取优先级决策

```
【数据获取决策树】

需要什么数据？
    │
    ├── 经济指标 → 世界银行/IMF/国家统计局
    │
    ├── 人口指标 → 联合国人口司/世界银行
    │
    ├── 体育数据 → Sports Reference/Olympedia/Kaggle
    │
    ├── 气象数据 → 中国气象数据网/OpenWeatherMap
    │
    ├── 金融数据 → Wind(付费)/东方财富/Alpha Vantage
    │
    └── 特定领域 → 先搜Kaggle，再搜政府官网
```

---

## 二、数据尺度统一方法

### 2.1 常见问题场景

- GDP（万亿级） vs 人口（亿级） vs 奖牌数（百级）
- 不同单位：美元 vs 人民币，公里 vs 米
- 不同分布：正态 vs 偏态 vs 长尾

### 2.2 标准化方法对比

| 方法 | 公式 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| **Z-Score标准化** | (x - μ) / σ | 正态分布数据 | 保留分布形状 | 对异常值敏感 |
| **Min-Max归一化** | (x - min) / (max - min) | 需要[0,1]范围 | 直观，保留关系 | 受极值影响大 |
| **对数变换** | log(x + 1) | 右偏分布、量级差异大 | 压缩量级差异 | 改变分布形状 |
| **分位数变换** | 映射到均匀/正态分布 | 异常值多、非正态 | 稳健 | 丢失原始分布信息 |
| **RobustScaler** | (x - median) / IQR | 有异常值 | 对异常值稳健 | 不常用 |

### 2.3 实践代码模板

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler

# ============ 方法1：对数变换 + 标准化（推荐组合）============
# 适用于：量级差异大的经济/人口数据

def scale_with_log(df, columns):
    """对数变换后标准化"""
    df_scaled = df.copy()
    
    # Step 1: 对数变换（处理量级差异）
    for col in columns:
        df_scaled[f'{col}_log'] = np.log1p(df_scaled[col])
    
    # Step 2: 标准化
    scaler = StandardScaler()
    log_cols = [f'{col}_log' for col in columns]
    df_scaled[log_cols] = scaler.fit_transform(df_scaled[log_cols])
    
    return df_scaled, scaler

# 使用示例
# df_scaled, scaler = scale_with_log(df, ['gdp', 'population'])


# ============ 方法2：分位数变换（处理异常值）============
# 适用于：有极端值、分布不规则的数据

def quantile_transform(df, columns, output_distribution='normal'):
    """分位数变换"""
    transformer = QuantileTransformer(
        output_distribution=output_distribution,  # 'normal' 或 'uniform'
        n_quantiles=min(len(df), 1000),
        random_state=42
    )
    df_transformed = df.copy()
    df_transformed[columns] = transformer.fit_transform(df[columns])
    return df_transformed, transformer


# ============ 方法3：按模型类型选择 ============

def scale_for_model(df, columns, model_type='tree'):
    """根据模型类型选择标准化方法"""
    
    if model_type in ['tree', 'xgboost', 'random_forest', 'lightgbm']:
        # 树模型：通常不需要标准化
        print("树模型不需要标准化，直接使用原始数据")
        return df, None
    
    elif model_type in ['linear', 'ridge', 'lasso', 'logistic']:
        # 线性模型：必须标准化
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled, scaler
    
    elif model_type in ['nn', 'neural_network', 'mlp']:
        # 神经网络：归一化到[0,1]或[-1,1]
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled, scaler
    
    elif model_type in ['svm', 'knn']:
        # 距离敏感模型：必须标准化
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled, scaler
    
    else:
        # 默认：标准化
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[columns] = scaler.fit_transform(df[columns])
        return df_scaled, scaler
```

### 2.4 标准化决策速查表

```
【标准化决策速查】

模型类型 → 是否需要标准化？

├── 线性回归/岭回归/Lasso → ✅ 必须（StandardScaler）
├── 逻辑回归 → ✅ 必须（StandardScaler）
├── SVM/KNN → ✅ 必须（StandardScaler）
├── 神经网络 → ✅ 必须（MinMaxScaler或StandardScaler）
├── 决策树 → ❌ 不需要
├── 随机森林 → ❌ 不需要
├── XGBoost/LightGBM → ❌ 不需要
├── ARIMA → ❌ 通常不需要（但可能需要差分）
└── PCA降维 → ✅ 必须（StandardScaler）

数据特点 → 推荐方法

├── 正态分布 → StandardScaler
├── 右偏分布（如GDP、人口）→ log变换 + StandardScaler
├── 有极端异常值 → RobustScaler 或 分位数变换
├── 需要[0,1]范围 → MinMaxScaler
└── 分布不规则 → QuantileTransformer
```

### 2.5 时间序列特殊注意

```python
# ⚠️ 时间序列标准化的关键：用训练集参数变换测试集

from sklearn.preprocessing import StandardScaler

# 正确做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # 只transform，不fit！

# 错误做法（数据泄露！）
# X_all_scaled = scaler.fit_transform(X_all)  # ❌ 不要这样做
```

---

## 三、系统性脑暴方法

### 3.1 PESTEL框架（宏观问题）

```
【PESTEL分析框架】

适用于：需要考虑外部环境因素的建模问题

P - Political（政治）
├── 政府政策变化
├── 国际关系
├── 法规监管
└── 示例：体育政策、奥运申办

E - Economic（经济）
├── GDP、人均收入
├── 经济增长率
├── 投资水平
└── 示例：体育预算、赞助收入

S - Social（社会）
├── 人口结构
├── 文化传统
├── 生活方式
└── 示例：体育参与率、运动文化

T - Technological（技术）
├── 技术进步
├── 创新能力
├── 基础设施
└── 示例：训练技术、运动科学

E - Environmental（环境）
├── 气候条件
├── 自然资源
├── 地理位置
└── 示例：冬季运动条件、海拔

L - Legal（法律）
├── 法规标准
├── 合规要求
├── 知识产权
└── 示例：反兴奋剂法规、参赛资格
```

### 3.2 因果链追溯法

```
【因果链追溯法】

核心思想：从目标变量出发，逆向追问"是什么导致了它？"

Step 1: 确定目标变量
        ↓
Step 2: 列出直接原因（一级因素）
        ↓
Step 3: 追问每个直接原因的原因（二级因素）
        ↓
Step 4: 继续追溯直到可观测/可量化
        ↓
Step 5: 评估数据可获得性

【示例：奥运奖牌预测】

奖牌数（目标）
    │
    ├── 运动员实力（直接原因）
    │   ├── 训练质量 ← 教练水平 ← 体育投入
    │   ├── 运动员数量 ← 人口基数 × 参与率
    │   └── 天赋选拔 ← 青训体系 ← 体育政策
    │
    ├── 参赛项目数（直接原因）
    │   ├── 项目覆盖率 ← 体育传统 + 投资分配
    │   └── 新增项目 ← 项目发展策略
    │
    ├── 东道主效应（直接原因）
    │   ├── 主场优势 ← 观众支持 + 适应性
    │   └── 资源倾斜 ← 国家重视 + 额外投入
    │
    └── 历史惯性（直接原因）
        ├── 历史成绩 ← 体育传统 + 优势项目
        └── 近期趋势 ← 政策变化 + 投入变化

【可量化的代理变量】
├── 体育投入 → GDP × 体育支出占比（世界银行）
├── 人口基数 → 总人口（联合国）
├── 参与率 → 青年人口比例（15-35岁）
├── 历史成绩 → 前几届奖牌数（Olympics数据）
├── 东道主 → 是否东道国（0/1变量）
└── 项目覆盖 → 参赛项目数/总项目数
```

### 3.3 类比迁移法

```
【类比迁移法】

核心思想：参考类似问题的解决方案和数据来源

Step 1: 识别问题的本质类型
Step 2: 搜索类似问题的研究/竞赛
Step 3: 提取可迁移的特征和方法
Step 4: 适配到当前问题

【示例：奥运奖牌预测】

类似问题：
├── 世界杯足球预测 → 国家实力指标、历史成绩
├── 诺贝尔奖预测 → 科研投入、教育水平
├── GDP预测 → 经济指标、人口因素
└── 选举预测 → 历史数据、趋势分析

可迁移的特征思路：
├── 历史表现 → 滞后变量、移动平均
├── 资源投入 → 预算、人员、设施
├── 规模因素 → 人口、经济体量
├── 趋势因素 → 增长率、加速度
└── 特殊事件 → 虚拟变量（0/1）
```

### 3.4 特征脑暴清单模板

```
【特征脑暴清单】

目标变量：________________

一、历史记忆型特征
□ lag_1：前1期值
□ lag_2：前2期值
□ lag_3：前3期值
□ rolling_mean_3：3期移动平均
□ rolling_max：历史最大值
□ rolling_min：历史最小值
□ historical_avg：历史平均

二、趋势动量型特征
□ growth_rate：增长率 = (当期-上期)/上期
□ acceleration：加速度 = 当期增长率 - 上期增长率
□ trend_slope：趋势斜率（线性回归）
□ momentum：动量 = 当期 - N期前

三、相对位置型特征
□ rank：排名
□ percentile：分位数位置
□ deviation_from_mean：偏离均值程度
□ z_score：标准化得分
□ relative_to_leader：与领先者的差距

四、比率效率型特征
□ per_capita：人均值 = 值/人口
□ per_gdp：单位GDP产出
□ efficiency：效率 = 产出/投入
□ ratio：占比 = 部分/整体
□ density：密度 = 数量/面积

五、交互效应型特征
□ product：乘积交互 = A × B
□ ratio_interaction：比值交互 = A / B
□ conditional：条件交互 = 条件 × 特征
□ polynomial：多项式 = A² 或 A × B

六、类别编码型特征
□ is_xxx：是否某类别（0/1）
□ region_onehot：地区独热编码
□ level_encoding：等级编码
□ target_encoding：目标编码

七、时间周期型特征
□ year：年份
□ cycle_position：周期位置（如奥运4年周期）
□ is_olympic_year：是否奥运年
□ years_since_last：距上次的年数
```

### 3.5 参考资料来源

```
【参考资料获取渠道】

一、学术论文
├── Google Scholar: scholar.google.com
│   搜索示例："Olympic medal prediction model"
├── 知网: cnki.net
│   搜索示例："奥运奖牌 预测模型"
├── arXiv: arxiv.org
│   搜索示例：stat.ML + sports prediction
└── ResearchGate: researchgate.net

二、竞赛优秀论文
├── 美赛官网: comap.com/contests/mcm-icm
├── 国赛: mcm.edu.cn
├── 项目中已有: MCMICM/ 文件夹
└── GitHub搜索: "MCM" + 年份

三、行业报告
├── 麦肯锡: mckinsey.com/insights
├── 德勤: deloitte.com/insights
├── 普华永道: pwc.com/insights
└── 搜索："体育产业白皮书"、"奥运经济报告"

四、技术博客
├── Towards Data Science: towardsdatascience.com
├── Medium: medium.com/tag/data-science
├── 知乎专栏: zhihu.com
└── CSDN: csdn.net

五、代码参考
├── Kaggle Notebooks: kaggle.com/code
├── GitHub: github.com
└── Papers with Code: paperswithcode.com
```

---

## 四、快速决策流程

```
【数据增强决策流程】

Step 1: 明确预测目标
        ↓
Step 2: 因果链追溯（列出所有可能影响因素）
        ↓
Step 3: 评估数据可获得性
        │
        ├── 公开免费 → 优先获取
        ├── 需要注册 → 评估价值后获取
        ├── 付费数据 → 寻找免费替代
        └── 无法获取 → 构建代理变量
        ↓
Step 4: 数据质量检查
        │
        ├── 缺失率 < 20% → 可用
        ├── 缺失率 20-50% → 需要插补
        └── 缺失率 > 50% → 考虑放弃
        ↓
Step 5: 尺度统一
        │
        ├── 树模型 → 不需要标准化
        └── 其他模型 → 选择合适的标准化方法
        ↓
Step 6: 特征工程
        │
        ├── 构建衍生特征
        ├── 检验相关性
        └── 筛选有效特征
        ↓
Step 7: 文档记录
        │
        ├── 数据来源
        ├── 处理方法
        └── 取舍理由
```

---

## 五、论文写作模板

### 5.1 外部数据考量章节模板

```markdown
### X.X 外部数据考量与特征工程

#### X.X.1 数据需求分析

基于因果链分析，我们识别了以下可能影响[目标变量]的外部因素：

| 因素类别 | 具体指标 | 理论依据 | 数据来源 | 可获得性 |
|---------|---------|---------|---------|---------|
| 经济因素 | GDP、人均GDP | [解释为什么相关] | 世界银行 | 公开 |
| 人口因素 | 总人口、年龄结构 | [解释为什么相关] | 联合国 | 公开 |
| ... | ... | ... | ... | ... |

#### X.X.2 数据获取与处理

成功引入的数据：
- **[数据名称]**：从[来源]获取，通过[关联键]与主数据表匹配，处理方法为[...]

未引入的数据及原因：
- **[数据名称]**：虽然理论上相关，但由于[数据粒度不匹配/获取成本过高/缺失率过高]，未纳入模型

#### X.X.3 特征相关性验证

[插入相关系数热力图]

如图所示，新引入的[特征名]与目标变量呈现[正/负]相关（r=[数值]），验证了引入该特征的合理性。
```

---

## 六、检查清单

```
【数据增强检查清单】

□ 数据来源
  □ 核心数据已获取
  □ 外部数据已识别并评估
  □ 数据来源已记录

□ 数据质量
  □ 缺失值已处理
  □ 异常值已检查
  □ 数据类型正确

□ 尺度统一
  □ 根据模型类型选择了合适的标准化方法
  □ 训练集和测试集分开处理
  □ 保存了标准化参数

□ 特征工程
  □ 历史特征已构建
  □ 趋势特征已构建
  □ 交互特征已考虑
  □ 无数据泄露

□ 文档记录
  □ 数据来源已说明
  □ 处理方法已记录
  □ 取舍理由已解释
```
