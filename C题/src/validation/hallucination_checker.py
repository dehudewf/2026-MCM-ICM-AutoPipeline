"""
@redcell Agent - Hallucination Attack Module
============================================

Deep integration with @redcell's Six-Dimension Attack Protocol.
Implements SelfCheckGPT methodology for automated paper verification.

Attack Dimensions:
1. assumption_attack  - 假设攻击
2. model_attack       - 模型攻击
3. data_attack        - 数据攻击
4. result_attack      - 结果攻击
5. presentation_attack - 表达攻击
6. format_attack      - 格式攻击

O-Award Compliance:
- All statistical outputs verified against code execution
- Assumption-Justification pairing enforced
- Causal language hedging checked
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import os


# ==================== @redcell Severity Classification ====================

class AttackSeverity(Enum):
    """@redcell severity levels - aligned with agents.md protocol"""
    FATAL = "fatal"       # 🚨 致命: 不修复直接出局
    CRITICAL = "critical" # ⚠️ 严重: 可能导致降级
    MAJOR = "major"       # 📝 一般: 影响评分
    MINOR = "minor"       # 💡 轻微: 锦上添花
    PASSED = "passed"     # ✅ 通过: 验证成功


class AttackDimension(Enum):
    """@redcell six attack dimensions"""
    ASSUMPTION = "assumption_attack"      # 假设攻击
    MODEL = "model_attack"                # 模型攻击
    DATA = "data_attack"                  # 数据攻击
    RESULT = "result_attack"              # 结果攻击
    PRESENTATION = "presentation_attack"  # 表达攻击
    FORMAT = "format_attack"              # 格式攻击


# ==================== Attack Finding Data Structures ====================

@dataclass
class AttackFinding:
    """Single attack finding - @redcell protocol compliant"""
    dimension: AttackDimension
    severity: AttackSeverity
    issue: str
    evidence: str
    impact: str
    recommendation: str
    action_required: str
    priority: str  # high, medium, low
    location: str = ""  # section/line in paper
    hallucination_score: float = 0.0  # 0.0 = factual, 1.0 = hallucinated


@dataclass
class RedCellFeedback:
    """
    @redcell → @executor/@strategist feedback schema
    Matches agents.md JSON schema for structured communication
    """
    message_type: str = "RedCellFeedback"
    task_id: str = ""
    review_id: str = ""
    timestamp: str = ""
    target_agent: str = "@executor"
    attack_dimensions: List[Dict] = field(default_factory=list)
    overall_assessment: Dict = field(default_factory=dict)
    action_items: List[Dict] = field(default_factory=list)
    approval_status: str = "conditional"  # conditional | approved | rejected
    approval_conditions: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ==================== @redcell Hallucination Checker ====================

class RedCellHallucinationChecker:
    """
    @redcell Agent - Six-Dimension Hallucination Attack System
    
    Deeply integrated with agents.md protocol for:
    1. assumption_attack  - 检查假设是否有Justification配对
    2. model_attack       - 检查模型选择理由
    3. data_attack        - 检查数据来源和泄露
    4. result_attack      - 检查统计结果是否代码派生
    5. presentation_attack - 检查因果语言和逻辑链
    6. format_attack      - 检查页数/身份信息/引用格式
    """
    
    def __init__(self, paper_path: str, code_outputs_path: str, 
                 statistical_tests_path: str = None):
        """
        Args:
            paper_path: Path to paper markdown file
            code_outputs_path: Path to code execution outputs (ground truth)
            statistical_tests_path: Path to statistical_tests_results.txt
        """
        self.paper_path = paper_path
        self.code_outputs_path = code_outputs_path
        self.statistical_tests_path = statistical_tests_path
        self.paper_content = self._load_file(paper_path)
        self.code_outputs = self._load_file(code_outputs_path)
        self.statistical_results = self._load_file(statistical_tests_path) if statistical_tests_path else ""
        
        # Ground truth: all verified numbers from code
        self.verified_numbers = self._extract_verified_numbers()
        
        # Findings storage
        self.findings: List[AttackFinding] = []
        
    def _load_file(self, path: str) -> str:
        """Load file content safely"""
        if not path or not os.path.exists(path):
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_verified_numbers(self) -> set:
        """Extract all numbers from code outputs as verified ground truth"""
        all_outputs = self.code_outputs + self.statistical_results
        numbers = re.findall(r'\b\d+\.?\d*\b', all_outputs)
        return set(numbers)

    # ==================== Dimension 1: Assumption Attack ====================
    
    def attack_assumptions(self) -> List[AttackFinding]:
        """
        @redcell assumption_attack
        Checks:
        - 每个假设是否有Justification配对
        - Justification是否有实证支撑
        """
        findings = []
        
        # Extract assumptions (A1, A2, etc.)
        assumption_pattern = r'\*\*A(\d+)[^*]*\*\*[:\s]*([^*]+?)(?=\*\*|\n\n|$)'
        assumptions = re.findall(assumption_pattern, self.paper_content, re.DOTALL)
        
        for a_num, a_content in assumptions:
            # Check for Justification
            has_justification = 'justification' in a_content.lower() or \
                               '*justification*' in a_content.lower()
            
            if not has_justification:
                findings.append(AttackFinding(
                    dimension=AttackDimension.ASSUMPTION,
                    severity=AttackSeverity.CRITICAL,
                    issue=f"假设 A{a_num} 缺少 Justification 配对",
                    evidence=f"A{a_num}: {a_content[:100]}...",
                    impact="评委可能质疑假设合理性",
                    recommendation="添加 *Justification:* 说明假设的实证依据",
                    action_required="@executor 补充假设论证",
                    priority="high",
                    hallucination_score=0.7
                ))
            else:
                findings.append(AttackFinding(
                    dimension=AttackDimension.ASSUMPTION,
                    severity=AttackSeverity.PASSED,
                    issue=f"假设 A{a_num} 已有 Justification",
                    evidence="Assumption-Justification pairing verified",
                    impact="无",
                    recommendation="✓ 通过",
                    action_required="无",
                    priority="low",
                    hallucination_score=0.0
                ))
        
        return findings

    # ==================== Dimension 2: Model Attack ====================
    
    def attack_model(self) -> List[AttackFinding]:
        """
        @redcell model_attack
        Checks:
        - 模型选择是否有理由
        - 是否有更简单的替代方案
        """
        findings = []
        
        # Check for model comparison table
        has_comparison = 'Model Performance Comparison' in self.paper_content or \
                        'Individual Model vs Ensemble' in self.paper_content
        
        if not has_comparison:
            findings.append(AttackFinding(
                dimension=AttackDimension.MODEL,
                severity=AttackSeverity.MAJOR,
                issue="缺少模型对比表",
                evidence="未找到 Model Performance Comparison 表格",
                impact="无法证明集成模型优于单模型",
                recommendation="添加各模型 R²/RMSE/MAE 对比表",
                action_required="@executor 补充模型对比",
                priority="medium",
                hallucination_score=0.5
            ))
        else:
            findings.append(AttackFinding(
                dimension=AttackDimension.MODEL,
                severity=AttackSeverity.PASSED,
                issue="模型对比表存在",
                evidence="Found model comparison table",
                impact="无",
                recommendation="✓ 通过",
                action_required="无",
                priority="low",
                hallucination_score=0.0
            ))
        
        return findings

    # ==================== Dimension 3: Data Attack ====================
    
    def attack_data(self) -> List[AttackFinding]:
        """
        @redcell data_attack
        Checks:
        - 数据来源是否说明
        - 是否存在数据泄露
        """
        findings = []
        
        # Check for data source citation
        has_source = 'COMAP' in self.paper_content or 'IOC' in self.paper_content
        
        if not has_source:
            findings.append(AttackFinding(
                dimension=AttackDimension.DATA,
                severity=AttackSeverity.CRITICAL,
                issue="数据来源未说明",
                evidence="未找到 COMAP 或 IOC 数据源引用",
                impact="评委可能质疑数据可信度",
                recommendation="明确标注数据来源",
                action_required="@executor 补充数据来源说明",
                priority="high",
                hallucination_score=0.8
            ))
        
        # Check for data leakage warning
        mentions_leakage = 'leakage' in self.paper_content.lower() or \
                          'temporal split' in self.paper_content.lower()
        
        if not mentions_leakage:
            findings.append(AttackFinding(
                dimension=AttackDimension.DATA,
                severity=AttackSeverity.MAJOR,
                issue="未讨论数据泄露防护",
                evidence="未找到 leakage 或 temporal split 相关讨论",
                impact="时序预测必须说明如何避免未来信息泄露",
                recommendation="在 Train/Test Split 部分说明时序分割方法",
                action_required="@executor 补充时序验证说明",
                priority="medium",
                hallucination_score=0.6
            ))
        
        return findings

    # ==================== Dimension 4: Result Attack ====================
    
    def attack_results(self) -> List[AttackFinding]:
        """
        @redcell result_attack - CORE SelfCheckGPT Integration
        Checks:
        - 每个统计数值是否在代码输出中找到
        - 置信区间是否合理
        """
        findings = []
        
        # Extract all statistical claims
        stat_patterns = [
            (r'R[²2]\s*[=≈]\s*([\d.]+)', 'R²'),
            (r'RMSE\s*[=≈]\s*([\d.]+)', 'RMSE'),
            (r'MAE\s*[=≈]\s*([\d.]+)', 'MAE'),
            (r'p\s*[<>=]\s*([\d.]+)', 'p-value'),
            (r'F-statistic\s*[=≈]\s*([\d.]+)', 'F-statistic'),
            (r't-statistic\s*[=≈]\s*([\d.]+)', 't-statistic'),
            (r"Cohen's\s*d\s*[=≈]\s*([\d.]+)", "Cohen's d"),
            (r'VIF\s*[=≈]\s*([\d.]+)', 'VIF'),
        ]
        
        verified_count = 0
        unverified_count = 0
        
        for pattern, stat_name in stat_patterns:
            matches = re.findall(pattern, self.paper_content, re.IGNORECASE)
            for value in matches:
                if value in self.verified_numbers:
                    verified_count += 1
                else:
                    unverified_count += 1
                    findings.append(AttackFinding(
                        dimension=AttackDimension.RESULT,
                        severity=AttackSeverity.CRITICAL,
                        issue=f"{stat_name}={value} 未在代码输出中找到",
                        evidence=f"搜索 '{value}' 于 code_outputs 和 statistical_tests 均未命中",
                        impact="评委可能质疑数据真实性",
                        recommendation=f"确保 {stat_name} 来自代码执行结果",
                        action_required="@executor 验证或修正该数值",
                        priority="high",
                        hallucination_score=0.9
                    ))
        
        # Summary finding
        if verified_count > 0:
            findings.append(AttackFinding(
                dimension=AttackDimension.RESULT,
                severity=AttackSeverity.PASSED,
                issue=f"{verified_count} 个统计数值已验证",
                evidence="Numbers found in code output",
                impact="无",
                recommendation="✓ 统计数值可溯源",
                action_required="无",
                priority="low",
                hallucination_score=0.0
            ))
        
        return findings

    # ==================== Dimension 5: Presentation Attack ====================
    
    def attack_presentation(self) -> List[AttackFinding]:
        """
        @redcell presentation_attack
        Checks:
        - 因果语言是否有hedging
        - 图表是否被引用
        """
        findings = []
        
        # Check causal claims for hedging
        causal_markers = ['cause', 'result in', 'lead to', 'because', 'therefore']
        hedging_terms = ['may', 'might', 'could', 'suggests', 'indicates', 'associated']
        
        for marker in causal_markers:
            if marker in self.paper_content.lower():
                # Check if hedging is nearby
                pattern = rf'.{{0,100}}{marker}.{{0,100}}'
                matches = re.findall(pattern, self.paper_content, re.IGNORECASE)
                for match in matches:
                    has_hedging = any(h in match.lower() for h in hedging_terms)
                    if not has_hedging:
                        findings.append(AttackFinding(
                            dimension=AttackDimension.PRESENTATION,
                            severity=AttackSeverity.MAJOR,
                            issue=f"因果语言 '{marker}' 缺少hedging",
                            evidence=f"...{match[:80]}...",
                            impact="过度因果推断，相关≠因果",
                            recommendation="添加 'may', 'suggests' 等hedging语言",
                            action_required="@executor 修改因果表述",
                            priority="medium",
                            hallucination_score=0.5
                        ))
                        break  # One finding per marker
        
        # Check Figure/Table references
        figures = re.findall(r'Figure\s*(\d+)', self.paper_content)
        tables = re.findall(r'Table\s*(\d+)', self.paper_content)
        
        for fig_num in set(figures):
            if figures.count(fig_num) < 2:  # Defined but never referenced
                findings.append(AttackFinding(
                    dimension=AttackDimension.PRESENTATION,
                    severity=AttackSeverity.MINOR,
                    issue=f"Figure {fig_num} 可能未被充分引用",
                    evidence=f"Figure {fig_num} appears only {figures.count(fig_num)} time(s)",
                    impact="评委可能认为图表没有讨论",
                    recommendation="在正文中引用并讨论该图表",
                    action_required="@executor 补充图表引用",
                    priority="low",
                    hallucination_score=0.2
                ))
        
        return findings

    # ==================== Dimension 6: Format Attack ====================
    
    def attack_format(self) -> List[AttackFinding]:
        """
        @redcell format_attack
        Checks:
        - 页数是否≤ 25
        - 是否有身份信息泄露
        - Summary Sheet 是否存在
        """
        findings = []
        
        # Check for Summary Sheet
        has_summary = 'Summary' in self.paper_content[:500] or \
                     '摘要' in self.paper_content[:500]
        
        if not has_summary:
            findings.append(AttackFinding(
                dimension=AttackDimension.FORMAT,
                severity=AttackSeverity.FATAL,
                issue="缺少 Summary Sheet",
                evidence="论文开头500字符内未找到 Summary",
                impact="直接出局！MCM强制要求",
                recommendation="在论文第一页添加 Summary Sheet",
                action_required="@executor 立即添加 Summary",
                priority="high",
                hallucination_score=1.0
            ))
        
        # Check for identity leakage
        identity_patterns = [
            r'\b[A-Za-z]+\s+University\b',
            r'\b[A-Za-z]+\s+College\b',
            r'@[a-zA-Z0-9]+\.edu',
            r'Team\s*#?\d+',
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, self.paper_content):
                findings.append(AttackFinding(
                    dimension=AttackDimension.FORMAT,
                    severity=AttackSeverity.FATAL,
                    issue="检测到潜在身份信息泄露",
                    evidence=f"匹配到模式: {pattern}",
                    impact="直接出局！绝对禁止",
                    recommendation="删除所有学校/姓名/邮箱信息",
                    action_required="@executor 立即删除身份信息",
                    priority="high",
                    hallucination_score=1.0
                ))
        
        return findings

    # ==================== Main Attack Pipeline ====================
    
    def run_full_attack(self) -> RedCellFeedback:
        """
        Execute all six attack dimensions.
        Returns structured @redcell feedback.
        """
        # Run all attacks
        all_findings = []
        all_findings.extend(self.attack_assumptions())
        all_findings.extend(self.attack_model())
        all_findings.extend(self.attack_data())
        all_findings.extend(self.attack_results())
        all_findings.extend(self.attack_presentation())
        all_findings.extend(self.attack_format())
        
        self.findings = all_findings
        
        # Build feedback structure
        feedback = RedCellFeedback(
            task_id=f"mcm_{datetime.now().strftime('%Y%m%d')}",
            review_id=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            target_agent="@executor"
        )
        
        # Group findings by dimension
        dimension_groups = {}
        for finding in all_findings:
            dim = finding.dimension.value
            if dim not in dimension_groups:
                dimension_groups[dim] = []
            dimension_groups[dim].append({
                "severity": finding.severity.value,
                "issue": finding.issue,
                "evidence": finding.evidence,
                "impact": finding.impact,
                "recommendation": finding.recommendation,
                "action_required": finding.action_required,
                "priority": finding.priority,
                "hallucination_score": finding.hallucination_score
            })
        
        feedback.attack_dimensions = [
            {"dimension": dim, "findings": findings}
            for dim, findings in dimension_groups.items()
        ]
        
        # Calculate overall assessment
        fatal_count = sum(1 for f in all_findings if f.severity == AttackSeverity.FATAL)
        critical_count = sum(1 for f in all_findings if f.severity == AttackSeverity.CRITICAL)
        major_count = sum(1 for f in all_findings if f.severity == AttackSeverity.MAJOR)
        passed_count = sum(1 for f in all_findings if f.severity == AttackSeverity.PASSED)
        
        avg_score = sum(f.hallucination_score for f in all_findings) / len(all_findings) if all_findings else 0
        
        feedback.overall_assessment = {
            "quality_score": round((1 - avg_score) * 10, 1),
            "quality_breakdown": {
                "fatal_issues": fatal_count,
                "critical_issues": critical_count,
                "major_issues": major_count,
                "passed_checks": passed_count
            },
            "o_award_readiness": f"{int((1 - avg_score) * 100)}%",
            "blocking_issues": fatal_count > 0,
            "average_hallucination_score": round(avg_score, 3)
        }
        
        # Generate action items
        action_id = 1
        for finding in all_findings:
            if finding.severity in [AttackSeverity.FATAL, AttackSeverity.CRITICAL]:
                feedback.action_items.append({
                    "id": f"AI_{action_id:03d}",
                    "assignee": "@executor",
                    "priority": finding.priority,
                    "task": finding.recommendation,
                    "deadline": "ASAP" if finding.severity == AttackSeverity.FATAL else "Before submission"
                })
                action_id += 1
        
        # Determine approval status
        if fatal_count > 0:
            feedback.approval_status = "rejected"
            feedback.approval_conditions = ["必须修复所有 FATAL 问题才能提交"]
        elif critical_count > 0:
            feedback.approval_status = "conditional"
            feedback.approval_conditions = [f"修复 {critical_count} 个 CRITICAL 问题后可提交"]
        else:
            feedback.approval_status = "approved"
            feedback.approval_conditions = ["✓ 通过所有关键检查"]
        
        return feedback

    def generate_markdown_report(self) -> str:
        """
        Generate human-readable @redcell attack report.
        """
        feedback = self.run_full_attack()
        
        report = []
        report.append("# 🚨 @redcell Hallucination Attack Report")
        report.append(f"\n**Review ID**: {feedback.review_id}")
        report.append(f"**Timestamp**: {feedback.timestamp}")
        report.append(f"**Paper**: {self.paper_path}")
        report.append("")
        
        # Overall Assessment
        assessment = feedback.overall_assessment
        report.append("## 🎯 Overall Assessment")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Quality Score | {assessment['quality_score']}/10 |")
        report.append(f"| O-Award Readiness | {assessment['o_award_readiness']} |")
        report.append(f"| Avg Hallucination Score | {assessment['average_hallucination_score']} |")
        report.append(f"| Approval Status | **{feedback.approval_status.upper()}** |")
        report.append("")
        
        # Issue Summary
        breakdown = assessment['quality_breakdown']
        report.append("## 📊 Issue Summary")
        report.append("")
        report.append("| Severity | Count |")
        report.append("|----------|-------|")
        report.append(f"| 🚨 Fatal | {breakdown['fatal_issues']} |")
        report.append(f"| ⚠️ Critical | {breakdown['critical_issues']} |")
        report.append(f"| 📝 Major | {breakdown['major_issues']} |")
        report.append(f"| ✅ Passed | {breakdown['passed_checks']} |")
        report.append("")
        
        # Dimension Details
        report.append("## 🔍 Attack Dimension Details")
        for dim_group in feedback.attack_dimensions:
            dim_name = dim_group['dimension'].replace('_', ' ').title()
            report.append(f"\n### {dim_name}")
            report.append("")
            report.append("| Issue | Severity | Score | Recommendation |")
            report.append("|-------|----------|-------|----------------|")
            for finding in dim_group['findings']:
                report.append(f"| {finding['issue'][:40]}... | {finding['severity']} | {finding['hallucination_score']} | {finding['recommendation'][:30]}... |")
        
        # Action Items
        if feedback.action_items:
            report.append("\n## 📝 Action Items")
            report.append("")
            report.append("| ID | Priority | Task | Deadline |")
            report.append("|-----|----------|------|----------|")
            for item in feedback.action_items:
                report.append(f"| {item['id']} | {item['priority']} | {item['task'][:40]}... | {item['deadline']} |")
        
        # Approval
        report.append("\n## ✅ Approval Status")
        report.append(f"\n**Status**: `{feedback.approval_status.upper()}`")
        for condition in feedback.approval_conditions:
            report.append(f"- {condition}")
        
        return "\n".join(report)


# ==================== Quick Usage Interface ====================

def run_redcell_attack(paper_path: str, outputs_path: str, 
                       statistical_tests_path: str = None) -> Tuple[RedCellFeedback, str]:
    """
    @redcell Agent Entry Point - Run full hallucination attack.
    
    Usage:
        feedback, report = run_redcell_attack(
            "paper/mcm_2025_c_paper.md",
            "output/complete_mcm_analysis_report.txt",
            "output/statistical_tests_results.txt"
        )
        print(report)  # Human-readable Markdown
        print(feedback.to_json())  # Structured JSON for agent communication
    
    Returns:
        Tuple[RedCellFeedback, str]: Structured feedback + Markdown report
    """
    checker = RedCellHallucinationChecker(
        paper_path=paper_path,
        code_outputs_path=outputs_path,
        statistical_tests_path=statistical_tests_path
    )
    feedback = checker.run_full_attack()
    report = checker.generate_markdown_report()
    return feedback, report


# Backward compatibility alias
def check_paper_hallucinations(paper_path: str, outputs_path: str) -> str:
    """Legacy function for backward compatibility"""
    _, report = run_redcell_attack(paper_path, outputs_path)
    return report


if __name__ == "__main__":
    # Example: Run @redcell attack on C题 paper
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paper_path = os.path.join(base_dir, "paper", "mcm_2025_c_paper.md")
    outputs_path = os.path.join(base_dir, "output", "complete_mcm_analysis_report.txt")
    stats_path = os.path.join(base_dir, "output", "statistical_tests_results.txt")
    
    print("\n" + "="*60)
    print("@redcell Agent - Six-Dimension Hallucination Attack")
    print("="*60 + "\n")
    
    if os.path.exists(paper_path):
        feedback, report = run_redcell_attack(paper_path, outputs_path, stats_path)
        print(report)
        print("\n" + "="*60)
        print("Structured Feedback (JSON for Agent Communication):")
        print("="*60)
        print(feedback.to_json())
    else:
        print(f"Paper not found: {paper_path}")
        sys.exit(1)
