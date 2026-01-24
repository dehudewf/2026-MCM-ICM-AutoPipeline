#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E题O奖论文PDF提取与分析脚本
逐篇提取Abstract和各章节内容
"""

import os
import re
import pdfplumber
from pathlib import Path

# 路径设置
BASE_DIR = Path("/Users/xiaohuiwei/Downloads/肖惠威美赛/E题/MCMICM E题")
OUTPUT_DIR = Path("/Users/xiaohuiwei/Downloads/肖惠威美赛/E题")

# 优先分析2024年E题O奖论文
PAPER_DIRS = [
    BASE_DIR / "2024美赛E题O奖论文",
    BASE_DIR / "2023年美赛O奖论文" / "E",
]


def extract_text_from_pdf(pdf_path, max_pages=10):
    """提取PDF前N页的文本"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n{'='*60}\n[Page {i+1}]\n{'='*60}\n"
                    text += page_text
    except Exception as e:
        text = f"Error reading PDF: {e}"
    return text


def extract_sections(text):
    """尝试提取论文的各个章节"""
    sections = {}
    
    # 常见章节标题模式
    section_patterns = [
        (r'(?i)(?:^|\n)\s*(abstract|summary)\s*(?:\n|:)', 'Abstract'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(introduction)\s*(?:\n|:)', 'Introduction'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(assumptions?(?:\s+and\s+notations?)?)\s*(?:\n|:)', 'Assumptions'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?((?:data|problem)\s*(?:collection|analysis|description)?)\s*(?:\n|:)', 'Data'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(model(?:ing)?(?:\s+development)?)\s*(?:\n|:)', 'Model'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(sensitivity|robustness)\s*(?:\n|:)', 'Sensitivity'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(strength|weakness|limitation)\s*(?:\n|:)', 'Strengths'),
        (r'(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(conclusion)\s*(?:\n|:)', 'Conclusion'),
    ]
    
    # 查找每个章节的位置
    for pattern, section_name in section_patterns:
        match = re.search(pattern, text)
        if match:
            sections[section_name] = match.start()
    
    return sections


def analyze_paper(pdf_path):
    """分析单篇论文"""
    paper_id = pdf_path.stem.split('-')[0].split('【')[0]
    print(f"\n{'#'*70}")
    print(f"# 论文: {paper_id}")
    print(f"# 文件: {pdf_path.name}")
    print(f"{'#'*70}")
    
    # 提取文本
    text = extract_text_from_pdf(pdf_path, max_pages=8)
    
    if "Error" in text:
        print(f"❌ {text}")
        return None
    
    # 输出前几页内容用于分析
    print(text[:15000])  # 打印前15000字符，覆盖Abstract和Introduction
    
    return {
        'paper_id': paper_id,
        'text': text,
        'sections': extract_sections(text)
    }


def main():
    """主函数：提取并分析E题论文"""
    print("="*70)
    print("E题O奖论文提取与分析")
    print("="*70)
    
    all_papers = []
    
    # 遍历论文目录
    for paper_dir in PAPER_DIRS:
        if not paper_dir.exists():
            print(f"⚠️ 目录不存在: {paper_dir}")
            continue
            
        print(f"\n📁 扫描目录: {paper_dir.name}")
        
        pdf_files = sorted(paper_dir.glob("*.pdf"))
        print(f"   找到 {len(pdf_files)} 篇论文")
        
        # 只分析前2篇作为示例
        for pdf_path in pdf_files[:2]:
            result = analyze_paper(pdf_path)
            if result:
                all_papers.append(result)
    
    print(f"\n{'='*70}")
    print(f"✅ 共分析 {len(all_papers)} 篇论文")
    print(f"{'='*70}")
    
    return all_papers


if __name__ == "__main__":
    main()
