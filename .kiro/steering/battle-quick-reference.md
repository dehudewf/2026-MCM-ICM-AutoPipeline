# MCM 2026 Battle Quick Reference Card (作战快速参考卡)

## 打印此页面，贴在桌面

---

## 1. Team Division (三人分工)

| 角色 | 职责 | 使用Agent | 时间重心 |
|------|------|-----------|----------|
| **主策师 Master Strategist** | 战略+调度+审核 | @strategist, @redcell | 审题90% → 调度50% → 审栈90% |
| **技术导演 Tech Director** | 数据+建模+代码 | @executor, @strategist | 数据30% → 建模90% → 支拴20% |
| **内容架构师 Content Architect** | 写作+图表+整合 | @executor, @redcell | 框架40% → 内容60% → 整合100% |

---

## 2. Critical Checkpoints (关键节点)

| 时间 | 必须完成 | 未完成则 | 责任人 |
|------|----------|----------|--------|
| **Hour 6** | 题目分析完成 | 延长到Hour 8 | Commander |
| **Hour 12** | 战略锁定（不超过2条路径） | Commander强制拍板 | Commander |
| **Hour 18** | 数据源确认 | 用备选数据 | Engineer |
| **Hour 24** | 数据清洗+基础特征 | 简化特征方案 | Engineer |
| **Hour 36** | 基线模型跑通 | 用更简单模型 | Engineer |
| **Hour 48** | 核心模型+初步结果 | 降级为简单模型 | Engineer |
| **Hour 60** | 敏感性分析完成 | 简化分析 | Engineer |
| **Hour 72** | 结果冻结（禁止改数据） | 全员转写作 | All |
| **Hour 84** | 全文整合+摘要完成 | 只保核心内容 | Narrator |
| **Hour 96** | 论文完成（可PDF） | 只保核心章节 | Narrator |
| **Hour 98** | @redcell终极攻击完成 | 只修致命问题 | Commander |
| **Hour 100** | 提交 | - | Commander |

---

## 3. Agent Triggers (Agent触发词)

```
@strategist  → 审题发散、创新角度、特征设计
@executor    → 代码生成、模型实现、写作执行
@redcell     → 质量攻击、逻辑检查、终极审核
```

---

## 4. Knowledge Base (知识库调用)

```
@knowledge:model   → 模型库
@knowledge:paper   → O奖论文
@knowledge:viz     → 可视化
@knowledge:data    → 数据源
@knowledge:method  → 建模方法
```

---

## 5. Tool Assignment (工具分配)

| 任务 | 工具 | 使用者 |
|------|------|--------|
| 代码 | Trae/Qoder + Claude Opus 4.5 | Engineer |
| 论文 | Claude Opus 4.5 + Overleaf | Narrator |
| 流程图 | Nano Banana Pro | Narrator |
| 数据图 | Origin + Matplotlib | Narrator |

---

## 6. Emergency Protocols (紧急预案)

| 情况 | 立即行动 |
|------|----------|
| 代码连续报错 | 简化方案，用基线模型 |
| 模型R²<0.5 | 检查特征，考虑更简单模型 |
| Hour 72论文<50% | 全员停止建模，转入写作 |
| Hour 96发现重大问题 | 只修致命问题，放弃次要问题 |

---

## 7. Golden Structure (黄金结构)

```
1. Summary Sheet (摘要) ← 重中之重，首页
2. Introduction (引言)
3. Assumptions (假设) ← 评审关键
4. Notations (符号)
5. Model Development (模型)
6. Results (结果)
7. Sensitivity Analysis (敏感性) ← 必需
8. Evaluation (评价) ← 必需
9. Conclusion (结论)
10. References (参考)
```

---

## 8. Nano Banana Quick Prompts

```
流程图: "Professional scientific flowchart for [X]. 
        Vector art, minimal, academic blue-grey, Nature quality."

架构图: "System architecture diagram showing [X]. 
        Modular layout, clean connections, publication ready."

示意图: "Conceptual schematic illustrating [X]. 
        Minimal design, informative labels, academic style."
```

---

## 9. Code Template

```python
# 必须包含
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 自修复模板
def safe_execute(func, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
    raise RuntimeError("Execution failed")
```

---

## 10. Final Checklist (提交前检查)

**致命项（必检）:**
- [ ] PDF命名: `队伍控制号.pdf`
- [ ] 页数 ≤ 25页（不含Summary Sheet和附录）
- [ ] 无任何身份信息泄露（页眉页脚、元数据、图片水印）
- [ ] Summary Sheet在第一页
- [ ] 每页页眉有论文编号

**结构完整:**
- [ ] Summary Sheet: 半页-1页，无公式/图片/表格
- [ ] Introduction + Literature Review存在
- [ ] Assumptions + Justifications存在
- [ ] Notations表格存在
- [ ] Model Development存在
- [ ] Results + Analysis存在
- [ ] Sensitivity Analysis存在
- [ ] Model Evaluation (Strengths/Weaknesses)存在
- [ ] Conclusion存在
- [ ] References存在

**质量检查:**
- [ ] 摘要第一句吸引人（避免"This paper..."）
- [ ] 每张图有编号、图注、被引用
- [ ] 每个假设有论证（Justification）
- [ ] 图表编号连续，无跳号
- [ ] 参考文献格式统一

**附件:**
- [ ] 代码整理 + README
- [ ] 数据源说明
- [ ] Summary Sheet单独文件

---

## The Formula

```
O奖 = (深刻洞察 × 严谨建模 × 清晰表达) ^ 团队协作
```

**Good Luck! Go for O-Award!**
