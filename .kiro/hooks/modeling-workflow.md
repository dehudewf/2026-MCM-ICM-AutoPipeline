# 建模工作流 Hooks 设计

## 工作流概览

```
数据获取 → 数据清洗 → 特征工程 → 模型训练 → 不确定性估计 → 可视化报告
```

## Hook 1: 数据获取触发器

当用户保存 `data/raw/` 目录下的文件时，自动触发数据质量检查。

```yaml
trigger: file_save
pattern: "data/raw/**"
action: "请检查新数据的质量：缺失值、异常值、数据类型"
```

## Hook 2: 特征工程触发器

当 `src/features/` 目录下的文件更新时，提醒更新特征文档。

```yaml
trigger: file_save
pattern: "src/features/**"
action: "特征代码已更新，请确保特征定义文档同步更新"
```

## Hook 3: 模型训练完成触发器

当模型文件保存到 `models/` 目录时，自动生成评估报告。

```yaml
trigger: file_save
pattern: "models/*.pkl"
action: "模型已保存，请运行评估脚本生成性能报告"
```

## 推荐的 Python 工具脚本

### 1. 数据清洗 Agent
位置: `src/agents/data_cleaner.py`

### 2. 特征筛选 Agent  
位置: `src/agents/feature_selector.py`

### 3. 预测 Agent（含共形预测）
位置: `src/agents/predictor.py`

### 4. 可视化 Agent
位置: `src/agents/visualizer.py`
```
