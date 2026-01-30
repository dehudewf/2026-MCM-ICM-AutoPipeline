# Anti-Hallucination Enforcement System (反幻觉强制执行系统)

> **核心原则**: 所有 AI 输出必须有据可查，禁止凭空捏造

---

## 🔴 MANDATORY RULES (强制规则)

### Rule 1: Data Provenance Requirement (数据溯源强制)

| Rule | Enforcement | Violation Consequence |
|------|-------------|----------------------|
| **所有数字必须标注来源** | 格式: `value [Source: file:line]` | 🔴 拒绝输出 |
| **禁止 AI 生成统计量** | F-value, t-value, VIF 等必须来自代码 | 🔴 拒绝输出 |
| **引用必须可验证** | 文献需提供 DOI 或 URL | 🟡 警告 |

```python
# MANDATORY: Every numerical output must follow this format
"""
USA predicted medals: 113 [Source: A题/output/predictions_2028.csv:row_1]
R² = 0.8377 [Source: model_evaluation.py:line_45, executed at 2026-01-26T10:30:00]
RMSE = 4.23 [Source: model_evaluation.py:line_47]
"""

# FORBIDDEN: Numbers without source attribution
"""
USA predicted medals: 113  # ❌ NO SOURCE - REJECTED
R² = 0.84  # ❌ NO SOURCE - REJECTED
"""
```

---

## 📋 Validation Pipeline (校验流水线)

### Stage 1: Pre-Generation Validation (生成前校验)

```yaml
check_before_generate:
  required_inputs:
    - model_results_file: "必须提供模型结果文件路径"
    - data_source_manifest: "必须提供数据来源清单"
    - code_execution_log: "必须提供代码执行日志"
  
  validation_rules:
    - rule: "model_results_file must exist"
      check: "os.path.exists(model_results_file)"
      on_fail: "ABORT - Cannot generate without real data"
    
    - rule: "code_execution_log must be recent"
      check: "log_timestamp within 24 hours"
      on_fail: "WARN - Stale results, recommend re-run"
```

### Stage 2: In-Generation Validation (生成中校验)

```yaml
check_during_generate:
  # Every number extracted must be verified against source
  number_verification:
    pattern: "\\d+\\.?\\d*"
    action: |
      FOR each number in AI_output:
        IF number NOT IN provided_data:
          REJECT with "Hallucinated number detected: {number}"
        ELSE:
          ANNOTATE with source location
  
  # Structure compliance check
  structure_verification:
    abstract:
      - "For Task 1" must exist
      - "For Task 2" must exist  
      - "For Task 3" must exist
      - "sensitivity" keyword must exist
      - Word count: 250-350
    
    assumptions:
      - "Assumption N:" format required
      - "Justification:" must follow each assumption
      - Count: 3-5 assumptions
```

### Stage 3: Post-Generation Validation (生成后校验)

```yaml
check_after_generate:
  traceability_audit:
    - "Every number has [Source: ...] annotation"
    - "All sources are verifiable files"
    - "No fabricated citations"
  
  consistency_check:
    - "Numbers in Abstract match Model Development"
    - "Assumptions are consistent across sections"
    - "No contradictory statements"
  
  output_format:
    validation_report:
      total_numbers: int
      traced_numbers: int
      untraced_numbers: int  # MUST be 0
      structure_compliance: bool
      consistency_issues: list
```

---

## 🔧 Implementation: TracedOutput Class

```python
"""
TracedOutput: Enforces data provenance for all AI outputs
Every piece of data must have a verifiable source
"""

import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class DataSource:
    """Represents a verifiable data source"""
    file_path: str
    line_number: Optional[int] = None
    column_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checksum: Optional[str] = None  # For file integrity verification
    
    def verify(self) -> bool:
        """Verify the source file exists and hasn't changed"""
        path = Path(self.file_path)
        if not path.exists():
            return False
        if self.checksum:
            current_checksum = hashlib.md5(path.read_bytes()).hexdigest()
            return current_checksum == self.checksum
        return True
    
    def to_annotation(self) -> str:
        """Format as inline annotation"""
        parts = [f"Source: {self.file_path}"]
        if self.line_number:
            parts.append(f"line_{self.line_number}")
        if self.column_name:
            parts.append(f"col_{self.column_name}")
        return f"[{', '.join(parts)}]"


@dataclass
class TracedValue:
    """A value with mandatory source attribution"""
    value: Any
    source: DataSource
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.value} {self.source.to_annotation()}"
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "source": {
                "file": self.source.file_path,
                "line": self.source.line_number,
                "column": self.source.column_name,
                "timestamp": self.source.timestamp
            },
            "description": self.description
        }


class TracedOutputValidator:
    """
    Validates AI outputs for data provenance
    Rejects any output with untraced numbers
    """
    
    def __init__(self, allowed_sources: List[str]):
        """
        Args:
            allowed_sources: List of file paths that contain valid data
        """
        self.allowed_sources = allowed_sources
        self.source_data: Dict[str, Any] = {}
        self.validation_log: List[Dict] = []
        self._load_sources()
    
    def _load_sources(self):
        """Load all allowed source files into memory"""
        for source_path in self.allowed_sources:
            path = Path(source_path)
            if path.exists():
                if path.suffix == '.csv':
                    import pandas as pd
                    self.source_data[source_path] = pd.read_csv(path)
                elif path.suffix == '.json':
                    self.source_data[source_path] = json.loads(path.read_text())
                else:
                    self.source_data[source_path] = path.read_text()
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        # Match integers, decimals, percentages, scientific notation
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?'
        return re.findall(pattern, text)
    
    def find_number_in_sources(self, number: str) -> Optional[DataSource]:
        """
        Search for a number in allowed sources
        Returns DataSource if found, None if not found (hallucination!)
        """
        # Normalize number for comparison
        normalized = number.rstrip('%')
        try:
            num_value = float(normalized)
        except ValueError:
            return None
        
        for source_path, data in self.source_data.items():
            if isinstance(data, str):
                if number in data or normalized in data:
                    # Find line number
                    for i, line in enumerate(data.split('\n'), 1):
                        if number in line or normalized in line:
                            return DataSource(source_path, line_number=i)
            elif hasattr(data, 'values'):  # DataFrame
                import numpy as np
                for col in data.columns:
                    for idx, val in enumerate(data[col]):
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            if abs(float(val) - num_value) < 0.001:
                                return DataSource(
                                    source_path, 
                                    line_number=idx+2,  # +2 for header and 0-index
                                    column_name=col
                                )
        return None
    
    def validate(self, ai_output: str) -> Dict:
        """
        Validate AI output for data provenance
        
        Returns:
            {
                "valid": bool,
                "traced_numbers": [...],
                "untraced_numbers": [...],  # HALLUCINATIONS!
                "validation_timestamp": str,
                "error_message": str or None
            }
        """
        numbers = self.extract_numbers(ai_output)
        traced = []
        untraced = []
        
        # Common false positives to skip
        skip_patterns = [
            r'^[12]$',  # Task numbers
            r'^20\d{2}$',  # Years
            r'^[1-9]$',  # Section numbers
        ]
        
        for num in numbers:
            # Skip common false positives
            if any(re.match(p, num) for p in skip_patterns):
                continue
            
            source = self.find_number_in_sources(num)
            if source:
                traced.append({
                    "number": num,
                    "source": source.to_annotation()
                })
            else:
                untraced.append(num)
        
        result = {
            "valid": len(untraced) == 0,
            "traced_numbers": traced,
            "untraced_numbers": untraced,
            "validation_timestamp": datetime.now().isoformat(),
            "error_message": None if len(untraced) == 0 else 
                f"🔴 HALLUCINATION DETECTED: {untraced}"
        }
        
        self.validation_log.append(result)
        return result
    
    def annotate_output(self, ai_output: str) -> str:
        """
        Add source annotations to all numbers in output
        Raises error if any number cannot be traced
        """
        validation = self.validate(ai_output)
        
        if not validation["valid"]:
            raise ValueError(
                f"Cannot annotate - hallucinated numbers found: "
                f"{validation['untraced_numbers']}"
            )
        
        # Add annotations
        annotated = ai_output
        for item in validation["traced_numbers"]:
            num = item["number"]
            source = item["source"]
            # Only annotate first occurrence to avoid duplicates
            annotated = annotated.replace(
                num, 
                f"{num} {source}", 
                1
            )
        
        return annotated
    
    def get_audit_report(self) -> str:
        """Generate audit trail report"""
        report = ["# Validation Audit Report", ""]
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Allowed Sources: {len(self.allowed_sources)}")
        report.append(f"Total Validations: {len(self.validation_log)}")
        report.append("")
        
        for i, log in enumerate(self.validation_log, 1):
            status = "✅ PASS" if log["valid"] else "❌ FAIL"
            report.append(f"## Validation {i}: {status}")
            report.append(f"- Timestamp: {log['validation_timestamp']}")
            report.append(f"- Traced: {len(log['traced_numbers'])}")
            report.append(f"- Untraced: {len(log['untraced_numbers'])}")
            if log['untraced_numbers']:
                report.append(f"- Hallucinations: {log['untraced_numbers']}")
            report.append("")
        
        return "\n".join(report)
```

---

## 🔄 Continuous Validation Loop

```python
class ContinuousValidator:
    """
    Implements continuous validation for iterative generation
    Every iteration must pass validation before proceeding
    """
    
    def __init__(
        self,
        validator: TracedOutputValidator,
        structure_rules: Dict[str, callable],
        max_iterations: int = 3
    ):
        self.validator = validator
        self.structure_rules = structure_rules
        self.max_iterations = max_iterations
        self.iteration_history: List[Dict] = []
    
    def validate_and_iterate(
        self, 
        llm_client,
        initial_prompt: str,
        context: Dict
    ) -> Dict:
        """
        Generate content with continuous validation
        
        Returns:
            {
                "content": str,  # Final validated content
                "iterations": int,
                "validation_history": [...],
                "final_audit": {...}
            }
        """
        messages = [{"role": "user", "content": initial_prompt}]
        
        for iteration in range(self.max_iterations):
            # Generate
            response = llm_client.create_completion(messages)
            content = self._extract_content(response)
            
            # Validate data provenance
            data_validation = self.validator.validate(content)
            
            # Validate structure
            structure_validation = self._validate_structure(content)
            
            # Log iteration
            self.iteration_history.append({
                "iteration": iteration + 1,
                "data_valid": data_validation["valid"],
                "structure_valid": structure_validation["valid"],
                "issues": {
                    "hallucinations": data_validation["untraced_numbers"],
                    "structure_errors": structure_validation["errors"]
                }
            })
            
            # Check if both validations pass
            if data_validation["valid"] and structure_validation["valid"]:
                # Annotate with sources and return
                annotated_content = self.validator.annotate_output(content)
                return {
                    "content": annotated_content,
                    "iterations": iteration + 1,
                    "validation_history": self.iteration_history,
                    "final_audit": self.validator.get_audit_report()
                }
            
            # Prepare critique for next iteration
            critique = self._build_critique(data_validation, structure_validation)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": critique})
        
        # Max iterations reached - return with warnings
        return {
            "content": content,
            "iterations": self.max_iterations,
            "validation_history": self.iteration_history,
            "final_audit": self.validator.get_audit_report(),
            "warning": "⚠️ Max iterations reached - output may contain issues"
        }
    
    def _validate_structure(self, content: str) -> Dict:
        """Validate against structure rules"""
        errors = []
        for rule_name, check_fn in self.structure_rules.items():
            try:
                if not check_fn(content):
                    errors.append(f"❌ {rule_name}")
            except Exception as e:
                errors.append(f"⚠️ {rule_name}: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _build_critique(self, data_val: Dict, struct_val: Dict) -> str:
        """Build critique prompt for next iteration"""
        critique = ["Your output has issues that must be fixed:\n"]
        
        if not data_val["valid"]:
            critique.append("## 🔴 DATA PROVENANCE ERRORS (CRITICAL)")
            critique.append("The following numbers are NOT from provided data sources:")
            for num in data_val["untraced_numbers"]:
                critique.append(f"  - `{num}` ← HALLUCINATED - REMOVE OR USE REAL DATA")
            critique.append("")
            critique.append("You MUST use ONLY numbers from the provided model_results.")
            critique.append("")
        
        if not struct_val["valid"]:
            critique.append("## Structure Errors")
            for error in struct_val["errors"]:
                critique.append(f"  - {error}")
            critique.append("")
        
        critique.append("Regenerate with these issues fixed.")
        return "\n".join(critique)
    
    def _extract_content(self, text: str) -> str:
        """Extract content from <action> tags if present"""
        import re
        match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        return match.group(1).strip() if match else text
```

---

## 📊 Validation Report Format

| Field | Type | Description |
|-------|------|-------------|
| `content` | str | Final validated content |
| `iterations` | int | Number of generation attempts |
| `traced_numbers` | List | All numbers with source annotations |
| `untraced_numbers` | List | Hallucinated numbers (MUST be empty) |
| `structure_compliance` | bool | Passed all structure rules |
| `validation_timestamp` | ISO str | When validation occurred |
| `audit_trail` | str | Full audit report |

### Example Validation Output

```json
{
  "content": "USA is predicted to win 113 [Source: predictions_2028.csv, line_1, col_Medals] medals...",
  "iterations": 2,
  "validation_history": [
    {
      "iteration": 1,
      "data_valid": false,
      "structure_valid": true,
      "issues": {
        "hallucinations": ["92", "5.2"],
        "structure_errors": []
      }
    },
    {
      "iteration": 2,
      "data_valid": true,
      "structure_valid": true,
      "issues": {
        "hallucinations": [],
        "structure_errors": []
      }
    }
  ],
  "final_audit": "# Validation Audit Report..."
}
```

---

## 🎯 Integration with @executor

### Updated Command Syntax

```
@executor 写作{章节} 
    --sources={数据文件1},{数据文件2}
    --validate=strict
    --audit=true
```

### Workflow

```yaml
@executor_write_with_validation:
  1. Load allowed data sources:
     validator = TracedOutputValidator([
       "A题/output/predictions_2028.csv",
       "A题/output/model_metrics.json"
     ])
  
  2. Initialize continuous validator:
     loop = ContinuousValidator(
       validator=validator,
       structure_rules=ABSTRACT_RULES,
       max_iterations=3
     )
  
  3. Generate with validation:
     result = loop.validate_and_iterate(llm, prompt, context)
  
  4. Output with audit trail:
     - content: result["content"]
     - audit: result["final_audit"]
     - iterations: result["iterations"]
  
  5. Store audit log:
     save_to("logs/validation_audit_{timestamp}.md")
```

---

## ✅ Enforcement Checklist

| Check | When | Action on Fail |
|-------|------|----------------|
| Data source files exist | Before generation | 🔴 ABORT |
| All numbers traceable | During validation | 🔴 REJECT + Iterate |
| Structure rules pass | After generation | 🟡 Iterate |
| No contradictions | Final check | 🟡 Warn |
| Audit log created | After completion | 🔴 ABORT |

---

## 📝 Quick Reference

```python
# Minimal usage example
from pathlib import Path

# 1. Define allowed sources
sources = [
    "A题/output/predictions_2028.csv",
    "A题/output/sensitivity_results.json"
]

# 2. Create validator
validator = TracedOutputValidator(sources)

# 3. Validate any AI output
result = validator.validate(ai_generated_text)

if not result["valid"]:
    print(f"🔴 REJECTED: Hallucinated numbers: {result['untraced_numbers']}")
else:
    # Safe to use - annotate with sources
    annotated = validator.annotate_output(ai_generated_text)
    print(annotated)
```
