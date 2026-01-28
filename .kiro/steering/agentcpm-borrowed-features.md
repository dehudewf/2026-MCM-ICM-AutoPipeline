# AgentCPM Borrowed Features for MCM Paper Writing

> 从 AgentCPM 架构中提取的两个核心功能，解决 MCM 论文写作的关键痛点

---

## 🥇 P0: Jinja2 Template System (防止数据幻觉)

### 问题背景

| 问题 | 后果 | 严重程度 |
|-----|------|---------|
| AI 编造数据 | "USA: 92±5 medals" ← 完全虚构 | 🔴 致命 |
| 无法注入真实结果 | 论文数据与模型结果不一致 | 🔴 致命 |
| 模板与数据耦合 | 修改困难，复用性差 | 🟡 中等 |

### 解决方案：Jinja2 动态模板注入

```yaml
核心原理:
  1. 模板定义结构 (template.jinja)
  2. 数据独立存储 (model_results.json)
  3. 运行时注入 (render)
  4. AI 只能使用注入的数据，无法编造

文件结构:
  templates/
    ├── mcm_abstract.jinja      # Abstract 模板
    ├── mcm_assumptions.jinja   # Assumptions 模板
    ├── mcm_model_dev.jinja     # Model Development 模板
    └── mcm_sensitivity.jinja   # Sensitivity Analysis 模板
  
  data/
    ├── model_results.json      # 你的真实模型结果
    └── assumptions.json        # 你的假设列表
```

### Template Schema

#### `mcm_abstract.jinja`

```jinja
{# ================================================================
   MCM Abstract Template (O-Award Structure)
   Variables:
     - problem_description: str
     - model_results: dict
     - available_data: str
   ================================================================ #}

You are an MCM/ICM O-Award paper writing expert.

## Context
**Problem Description:**
{{ problem_description }}

**Your Model Results (USE ONLY THESE - NO FABRICATION):**
{% for task, result in model_results.items() %}
### {{ task }}
{% for key, value in result.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endfor %}

**Available Data:**
{{ available_data }}

## Structure Requirements (MANDATORY)

| Section | Format | Example |
|---------|--------|---------|
| Background | 1 sentence: "As [trend] intensifies, [challenge] becomes pressing" | As climate uncertainty increases... |
| Problem Statement | 1 sentence: "In order to [goal], we build [N] models..." | In order to predict medal counts... |
| Task 1 | "For Task 1, we developed [Model] to [purpose]. The result indicates [QUANTITATIVE from model_results]" | For Task 1, we developed ARIMA... |
| Task 2 | "For Task 2, we established [Model]. Applied to [location], results indicate..." | For Task 2, we established HAI... |
| Task 3 | "For Task 3, we applied [Model] to [case]. Score = [VALUE from model_results]" | For Task 3, we applied... |
| Sensitivity | "Finally, we analyze sensitivity and robustness. The model is stable and [adj]" | Finally, we analyze... |
| Keywords | 5-7 terms: [Model1], [Model2], [Method1], [Method2], [Domain] | ARIMA, XGBoost, AHP... |

## Critical Rules
1. ALL numerical results MUST come from `model_results` above
2. DO NOT fabricate any numbers - use ONLY provided results
3. Use "For Task N, we..." format EXACTLY
4. Word count: 250-350 words

## Output Format
<thought>
- Which results from model_results will I use?
- How will I ensure O-Award structure compliance?
</thought>

<action>
[Your Abstract in Markdown]
</action>
```

#### `mcm_assumptions.jinja`

```jinja
{# ================================================================
   MCM Assumptions Template
   Variables:
     - assumptions_list: list[dict] with keys: content, justification
   ================================================================ #}

You are an MCM/ICM O-Award paper writing expert.

## Provided Assumptions (USE EXACTLY AS GIVEN)
{% for i, assumption in enumerate(assumptions_list, 1) %}
**Assumption {{ i }}:** {{ assumption.content }}
**Justification:** {{ assumption.justification }}
{% endfor %}

## O-Award Format Requirements

| Component | Requirement | Frequency in O-Award Papers |
|-----------|-------------|----------------------------|
| Assumption N: | Numbered format required | 29/20 papers |
| Justification N: | MUST follow each assumption | 43 occurrences |
| Total count | 3-5 assumptions | Standard |
| Intro sentence | Optional but recommended | 8/20 papers |

## Output Format

```
2 Assumptions and Justifications

[Optional intro: "It is not possible to model every possible scenario. 
So we make some reasonable assumptions to simplify the model, each with 
a corresponding explanation:"]

Assumption 1: [content]
Justification: [justification]

Assumption 2: [content]
Justification: [justification]

[... continue for all assumptions ...]
```

<action>
[Your Assumptions section in Markdown]
</action>
```

### Python Implementation

```python
from jinja2 import Template, Environment, FileSystemLoader
import json

class MCMTemplateEngine:
    """Jinja2-based template engine for MCM paper writing"""
    
    def __init__(self, template_dir: str = "templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def render(self, template_name: str, **context) -> str:
        """
        Render a template with given context
        
        Args:
            template_name: e.g., "mcm_abstract.jinja"
            **context: Variables to inject (problem_description, model_results, etc.)
        
        Returns:
            Rendered prompt string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def render_abstract(self, problem_desc: str, model_results: dict) -> str:
        """Convenience method for Abstract generation"""
        return self.render(
            "mcm_abstract.jinja",
            problem_description=problem_desc,
            model_results=model_results,
            available_data="Olympic medal data 2000-2024, GDP, population"
        )

# Usage Example
engine = MCMTemplateEngine()

# Your REAL model results (from C题/output/)
model_results = {
    "Task 1": {
        "USA_medals": "113 (95% CI: 105-121)",
        "R_squared": 0.8377,
        "RMSE": 4.23
    },
    "Task 2": {
        "Host_Advantage_Index": "+18.7%",
        "AHP_consistency": 0.06
    },
    "Task 3": {
        "Swimming_ROI": "12.4%",
        "Athletics_ROI": "9.1%"
    }
}

prompt = engine.render_abstract(
    problem_desc="2024 MCM Problem C: Olympic Medal Prediction",
    model_results=model_results
)

# Now call LLM with this prompt - AI cannot fabricate numbers!
```

---

## 🥉 P2: Task State Management (迭代式改进)

### 问题背景

| 问题 | 后果 | 严重程度 |
|-----|------|---------|
| 单次生成质量不稳定 | 结构不符合 O-Award 要求 | 🟡 中等 |
| 无法自动验证输出 | 需要人工检查每个细节 | 🟡 中等 |
| 无上下文记忆 | 重新生成时丢失改进历史 | 🟡 中等 |

### 解决方案：MCMSection with Iterative Refinement

```yaml
核心原理:
  1. 每个章节 = 1 个 Task 对象
  2. 内置验证规则 (validation_rules)
  3. 自动迭代改进 (max_iterations)
  4. 保留对话历史 (conversation_history)

工作流程:
  Generate Draft → Validate → [Pass] → Return
                      ↓ [Fail]
             Add Critique → Regenerate → Loop (max 3x)
```

### MCMSection Schema

```python
from typing import Dict, List, Callable, Optional
import re

class MCMSection:
    """
    Represents one paper section with built-in validation and iterative refinement
    
    Attributes:
        name: Section name (e.g., "Abstract", "Assumptions")
        template_path: Path to Jinja2 template
        validation_rules: Dict of rule_name -> check_function
        conversation_history: Multi-turn memory for refinement
        max_iterations: Maximum refinement attempts
    """
    
    def __init__(
        self,
        name: str,
        template_path: str,
        validation_rules: Dict[str, Callable[[str], bool]]
    ):
        self.name = name
        self.template_path = template_path
        self.validation_rules = validation_rules
        self.conversation_history: List[Dict] = []
        self.max_iterations = 3
        self.generation_count = 0
    
    def generate(self, llm_client, context: Dict) -> str:
        """
        Generate section with automatic validation and refinement
        
        Args:
            llm_client: LLM client with create_completion method
            context: Template variables (problem_description, model_results, etc.)
        
        Returns:
            Validated section content
        """
        # Load and render template
        prompt = self._render_template(context)
        
        for iteration in range(self.max_iterations):
            self.generation_count += 1
            
            # Build messages with history
            messages = self.conversation_history + [
                {"role": "user", "content": prompt}
            ]
            
            # Generate
            response = llm_client.create_completion(messages)
            content = self._extract_action(response)
            
            # Validate
            validation = self.validate(content)
            
            if validation["passed"]:
                return content
            
            # Add critique for next iteration
            self.conversation_history.append({"role": "assistant", "content": response})
            self.conversation_history.append({
                "role": "user",
                "content": self._format_critique(validation["errors"])
            })
        
        # Return best attempt after max iterations
        return content
    
    def validate(self, content: str) -> Dict:
        """
        Validate content against all rules
        
        Returns:
            {"passed": bool, "errors": List[str]}
        """
        errors = []
        
        for rule_name, check_fn in self.validation_rules.items():
            try:
                if not check_fn(content):
                    errors.append(f"❌ Failed: {rule_name}")
            except Exception as e:
                errors.append(f"⚠️ Error in {rule_name}: {str(e)}")
        
        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "rules_checked": len(self.validation_rules),
            "rules_passed": len(self.validation_rules) - len(errors)
        }
    
    def _extract_action(self, text: str) -> str:
        """Extract content from <action>...</action> tags"""
        match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def _format_critique(self, errors: List[str]) -> str:
        """Format validation errors as critique prompt"""
        return f"""
Your previous output failed validation. Please fix these issues:

{chr(10).join(errors)}

Regenerate the section with these issues addressed.
"""
    
    def _render_template(self, context: Dict) -> str:
        """Render Jinja2 template with context"""
        from jinja2 import Template
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template = Template(f.read())
        return template.render(**context)
```

### Pre-defined Validation Rules

| Section | Rule Name | Check Function | O-Award Requirement |
|---------|-----------|----------------|---------------------|
| **Abstract** | `has_background` | `lambda c: "As " in c or "The growing" in c` | Background sentence |
| | `has_task_1` | `lambda c: "For Task 1" in c` | Task 1 description |
| | `has_task_2` | `lambda c: "For Task 2" in c` | Task 2 description |
| | `has_task_3` | `lambda c: "For Task 3" in c` | Task 3 description |
| | `has_sensitivity` | `lambda c: "sensitivity" in c.lower()` | Sensitivity statement |
| | `has_keywords` | `lambda c: "Keywords" in c` | Keywords list |
| | `word_count` | `lambda c: 250 <= len(c.split()) <= 350` | 250-350 words |
| **Assumptions** | `has_justification` | `lambda c: "Justification" in c` | Each assumption needs justification |
| | `count_3_to_5` | `lambda c: 3 <= c.count("Assumption") <= 5` | 3-5 assumptions |
| | `numbered_format` | `lambda c: "Assumption 1" in c` | Numbered format |
| **Model Dev** | `has_equations` | `lambda c: "$$" in c or "$" in c` | Mathematical formulas |
| | `has_figure_ref` | `lambda c: "Figure" in c` | Figure references |
| | `has_results` | `lambda c: "result" in c.lower()` | Results presented |

### Pre-configured Section Factory

```python
class MCMSectionFactory:
    """Factory for creating pre-configured MCM sections"""
    
    @staticmethod
    def create_abstract(template_path: str = "templates/mcm_abstract.jinja") -> MCMSection:
        return MCMSection(
            name="Abstract",
            template_path=template_path,
            validation_rules={
                "has_background": lambda c: "As " in c or "The growing" in c,
                "has_task_1": lambda c: "For Task 1" in c,
                "has_task_2": lambda c: "For Task 2" in c,
                "has_task_3": lambda c: "For Task 3" in c,
                "has_sensitivity": lambda c: "sensitivity" in c.lower(),
                "has_keywords": lambda c: "Keywords" in c or "keywords" in c,
                "word_count_min": lambda c: len(c.split()) >= 250,
                "word_count_max": lambda c: len(c.split()) <= 350,
            }
        )
    
    @staticmethod
    def create_assumptions(template_path: str = "templates/mcm_assumptions.jinja") -> MCMSection:
        return MCMSection(
            name="Assumptions",
            template_path=template_path,
            validation_rules={
                "has_justification": lambda c: c.count("Justification") >= 3,
                "count_min": lambda c: c.count("Assumption") >= 3,
                "count_max": lambda c: c.count("Assumption") <= 5,
                "numbered_format": lambda c: "Assumption 1" in c,
            }
        )
    
    @staticmethod
    def create_sensitivity(template_path: str = "templates/mcm_sensitivity.jinja") -> MCMSection:
        return MCMSection(
            name="Sensitivity Analysis",
            template_path=template_path,
            validation_rules={
                "has_parameter_test": lambda c: "parameter" in c.lower(),
                "has_robustness": lambda c: "robust" in c.lower() or "stable" in c.lower(),
                "has_quantitative": lambda c: "%" in c or any(char.isdigit() for char in c),
            }
        )
```

---

## Integration with @executor Agent

### 更新 Agent 调用流程

```yaml
# 在 @executor 执行写作任务时使用

@executor_write_section:
  trigger: 需要生成论文章节时
  
  workflow:
    1. 加载 MCMSection:
       section = MCMSectionFactory.create_abstract()
    
    2. 准备真实数据:
       model_results = load_from("C题/output/predictions_2028.csv")
    
    3. 渲染模板:
       prompt = template_engine.render("mcm_abstract.jinja", model_results=model_results)
    
    4. 迭代生成:
       content = section.generate(llm_client, context)
    
    5. 验证通过后输出:
       ```json:a2a:executor_to_redcell
       {
         "section_name": "Abstract",
         "content": content,
         "validation_results": section.validate(content),
         "iterations_used": section.generation_count,
         "data_source": "C题/output/predictions_2028.csv"
       }
       ```

  benefits:
    - ✅ 数据不会被编造（模板注入）
    - ✅ 结构自动验证
    - ✅ 迭代改进直到通过
    - ✅ 可追溯的生成历史
```

### Quick Commands Update

```
# 新增命令
@executor 写作{章节} --template={模板} --data={数据文件}
@executor 验证{章节} --rules={规则集}
@executor 迭代改进{章节} --max={次数}
```

---

## File Checklist

| 文件路径 | 用途 | 状态 |
|---------|------|------|
| `templates/mcm_abstract.jinja` | Abstract 模板 | 需创建 |
| `templates/mcm_assumptions.jinja` | Assumptions 模板 | 需创建 |
| `templates/mcm_model_dev.jinja` | Model Development 模板 | 需创建 |
| `templates/mcm_sensitivity.jinja` | Sensitivity 模板 | 需创建 |
| `src/template_engine.py` | Jinja2 引擎封装 | 需创建 |
| `src/mcm_section.py` | MCMSection 类 | 需创建 |
| `data/model_results.json` | 模型结果数据 | 从 C题/output/ 导出 |

---

## Summary Table

| Feature | Problem Solved | Implementation | Effort | Value |
|---------|---------------|----------------|--------|-------|
| **Jinja2 Templates** | 数据幻觉 | 模板注入真实数据 | 1h | 🔥🔥🔥 |
| **MCMSection** | 结构不合规 | 自动验证+迭代 | 1h | 🔥🔥 |
| **Validation Rules** | 人工检查成本 | 预定义规则集 | 30min | 🔥🔥 |
| **Factory Pattern** | 配置繁琐 | 预配置章节工厂 | 15min | 🔥 |
