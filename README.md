# MCM 2026 O-Award Battle System 🏆

> **目标：Outstanding Winner (O奖)**  
> 一套完整的美赛多智能体协作系统，为三人战队提供从审题到提交的全流程支持。

---

## 🎯 系统概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MCM 2026 O-Award Multi-Agent System                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   👤 主策师 (Master Strategist)                                                  │
│   ├── @strategist 审题发散                                                       │
│   └── @redcell 质量攻击                                                          │
│                                                                                  │
│   👤 技术导演 (Tech Director)          👤 内容架构师 (Content Architect)          │
│   ├── @executor 代码生成                ├── @executor 论文撰写                   │
│   └── @knowledge:model 模型检索         └── @knowledge:viz 可视化检索            │
│                                                                                  │
│   ════════════════════════════════════════════════════════════════════════════  │
│                                                                                  │
│   📚 知识库: 模型库 | 论文分析 | 可视化规范 | 数据源                             │
│   🛠️ 工具: Qoder/Trae + Claude Opus 4.5 + Nano Banana Pro + Origin              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 项目结构

```
meisai/
├── .qoder/rules/                    # AI规则层（自动加载）
│   ├── agents.md                    # Agent完整规则（@strategist/@executor/@redcell）
│   ├── mcm_ultimate_system.md       # 主规则+时间线+检查清单
│   └── skills.md                    # Skills调用规则
│
├── .kiro/steering/                  # 人类参考层（手动查阅）
│   ├── modeling-prompts-final.md    # 10大建模提示词
│   ├── modeling-prompts-宏观.md     # 方法论层面决策
│   ├── modeling-prompts微观.md      # 技术实现细节
│   ├── data-sources-and-brainstorm.md # 数据源+脑暴方法
│   ├── battle-quick-reference.md    # 快速参考卡
│   ├── 生成图的提示词.txt            # Nano Banana提示词
│   ├── 人工写作总指南.pdf            # 写作规范+Golden Structure
│   ├── 比赛实战资源调用指南.md       # 资源调用流程图
│   └── 三人战队实战手册.md           # 三人分工手册（打印分发！）
│
├── 知识库/                          # 知识资产层（@knowledge调用）
│   ├── 模型库1.xlsx, 模型库2.xlsx, 模型库3.xlsx  # 300+模型参考
│   ├── C题模型知识库_优化版.xlsx    # C题专用模型索引
│   ├── 论文分析结果_优化版.xlsx     # O奖论文结构分析
│   ├── 可视化知识库.csv             # 图表类型&设计规范
│   └── 分析统计报告.txt             # 统计汇总
│
├── MCMICM/                          # 历年O奖论文原文
│   ├── 2022美赛优秀论文集/
│   ├── 2023/C/
│   └── 2024/C/
│
├── src/                             # 分析代码
│   ├── data/                        # 数据处理模块
│   ├── models/                      # 模型实现
│   ├── analysis/                    # 分析模块
│   └── output/                      # 输出模块
│
├── data/                            # 数据文件
├── output/                          # 输出结果
└── tests/                           # 测试文件
```

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/mcm-2026-battle-system.git
cd mcm-2026-battle-system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 使用 Qoder/Trae 打开项目
# Agent规则会自动加载
```

### 三人分工

| 角色 | 职责 | 常用指令 |
|------|------|----------|
| **👤 A 主策师** | 战略制定 + 质量把控 + 最终决策 | `@strategist 审题` `@redcell 攻击` |
| **👤 B 技术导演** | 数据+建模+代码一体化 | `@executor 实现路径N` `@knowledge:model` |
| **👤 C 内容架构师** | 论文+可视化+表达设计 | `@executor 撰写{章节}` `@knowledge:viz` |

### 100小时时间线

```
时间    0h ──── 12h ──── 48h ──── 72h ──── 96h ── 100h
        │        │        │        │        │       │
阶段    审题  →  锁定  →  中期  →  冻结  →  攻击  → 提交
        │        │        │        │        │       │
👤 A    发散    拍板    监控    审核    攻击    确认
👤 B    调研    建模    敏感性  冻结    修复    检查
👤 C    文献    框架    图表    整合    润色    检查
```

---

## 🤖 Agent调用指令

### @strategist 战略家

```bash
@strategist 审题              # 深度解析题目
@strategist 发散3条路径       # 系统性发散建模路径
@strategist 构建假设体系      # 设计分层假设
@strategist 评估路径N         # 评估可行性和竞争力
```

### @executor 执行者

```bash
@executor 实现路径N           # 实现指定建模路径
@executor 特征工程            # 四类特征设计
@executor 训练{模型类型}      # 训练指定模型
@executor 生成SHAP            # SHAP解释图
@executor 敏感性分析          # 参数敏感性测试
@executor 撰写{章节名}        # 撰写论文章节
```

### @redcell 批判者

```bash
@redcell 攻击假设             # 攻击假设合理性
@redcell 攻击模型             # 攻击模型选择
@redcell 全面审核             # 六维攻击
@redcell 提交前检查           # 50+项检查清单
```

### @knowledge 知识库

```bash
@knowledge:model              # 检索模型库
@knowledge:paper              # 检索O奖论文结构
@knowledge:viz                # 检索可视化规范
@knowledge:data               # 检索数据源
@knowledge:method             # 检索建模方法
```

---

## 📋 关键文档速查

| 文档 | 用途 | 使用者 |
|------|------|--------|
| `三人战队实战手册.md` | **打印分发给队友！** 完整的三人分工指南 | 全员 |
| `battle-quick-reference.md` | 关键节点速查卡 | 全员 |
| `modeling-prompts-final.md` | 10大建模提示词 | 全员 |
| `mcm_ultimate_system.md` | 提交检查清单（50+项） | 主策师 |
| `data-sources-and-brainstorm.md` | 数据源大全 | 技术导演 |
| `可视化知识库.csv` | 图表设计规范 | 内容架构师 |

---

## 🚨 提交前检查（致命项）

```
□ 页数 ≤ 25页（不含Summary Sheet和附录）
□ PDF命名 = 队伍控制号.pdf（如 2412345.pdf）
□ 无身份信息泄露（页眉页脚、元数据、图片水印）
□ Summary Sheet 在论文第一页
□ 每页页眉有论文编号
```

---

## 📊 知识库内容

### 模型库统计

| 分类 | 模型数量 | 来源 |
|------|----------|------|
| 基础模型 | 50+ | 模型库1.xlsx |
| 进阶模型 | 80+ | 模型库2.xlsx |
| 特殊模型 | 40+ | 模型库3.xlsx |
| C题专用 | 30+ | C题模型知识库_优化版.xlsx |

### O奖论文分析

- 覆盖年份：2022-2024
- 分析维度：结构、模型、图表、创新点
- 输出：论文分析结果_优化版.xlsx

---

## 🛠️ 技术栈

| 工具 | 用途 |
|------|------|
| **Qoder/Trae** | IDE + Agent系统 |
| **Claude Opus 4.5** | 代码生成 + 论文撰写 |
| **Nano Banana Pro** | Nature级示意图生成 |
| **Origin** | 数据可视化 |
| **Python** | 建模实现 |
| **Overleaf** | LaTeX论文排版 |

---

## 📝 更新日志

### v2.0.0 (2026-01-18)
- 🚀 全新Multi-Agent协作系统
- 👥 三人战队分工框架
- 🤖 Agent规则完整定义（@strategist/@executor/@redcell）
- 📚 知识库整合（模型/论文/可视化/数据源）
- ⏰ 100小时精确时间线
- ✅ 50+项提交检查清单
- 📋 三人战队实战手册

### v1.0.0 (2025-01-03)
- 📄 论文分析工具集
- 🤖 智能模型识别
- 📊 统计报告生成

---

## 🤝 团队

- 主策师 (Master Strategist)
- 技术导演 (Tech Director)
- 内容架构师 (Content Architect)

---

## 📄 许可证

本项目仅用于学习研究，论文版权归原作者所有。

---

**🏆 目标：O奖！祝比赛顺利！**

