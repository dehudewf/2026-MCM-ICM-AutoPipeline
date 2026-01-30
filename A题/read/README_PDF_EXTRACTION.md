# PDF Content & Annotation Extraction Tool

## 📋 Overview

This tool extracts PDF content along with **mapped annotations (批注)**, ensuring each annotation is associated with its corresponding text location.

## ✅ Extraction Results

Successfully extracted from `2026_MCM_Problem_A_加水印_加水印.pdf`:

- **Total Pages**: 3
- **Total Annotations**: 59
- **Annotation Types**: Highlight, Text, Popup

## 📦 Output Files

### 1. JSON Format (Structured Data)
**File**: `2026_MCM_Problem_A_加水印_加水印_annotations.json`

```json
{
  "page": 1,
  "annotation": {
    "type": "/Highlight",
    "content": "这里是相互作用，协调的说法",
    "author": "微信用户",
    "date": "D:20260130140517+08'00'",
    "position": [383.641, 475.74, 437.641, 490.32]
  },
  "page_text_snippet": "...related text around annotation...",
  "full_page_text": "...complete page text..."
}
```

**Best for**: Programmatic processing, data analysis, integration with other tools

---

### 2. Markdown Format (Human-Readable)
**File**: `2026_MCM_Problem_A_加水印_加水印_annotations.md`

```markdown
## Annotation 1 (Page 1)

**Type**: /Highlight
**Author**: 微信用户
**Date**: D:20260130140517+08'00'

**Content (批注内容)**:
```
这里是相互作用，协调的说法
```

**Related Text Snippet**:
```
...corresponding text from PDF...
```
```

**Best for**: Quick review, sharing with team members, documentation

---

### 3. Full Text with Inline Annotations
**File**: `2026_MCM_Problem_A_加水印_加水印_full_content_with_annotations.txt`

```
================================================================================
PAGE 1
================================================================================

Your task is to develop a continuous-time mathematical model...

[批注 - Page 1]
  • /Highlight: 这里是相互作用，协调的说法
    (Author: 微信用户)
  • /Highlight: 温度的影响
    (Author: 微信用户)
  • /Highlight: 关键词1
    (Author: 微信用户)
```

**Best for**: Reading flow, understanding annotations in context

---

## 🚀 Usage

### Basic Command
```bash
python3 extract_pdf_with_annotations.py '<pdf_file_path>'
```

### Example
```bash
python3 extract_pdf_with_annotations.py '2026_MCM_Problem_A_加水印_加水印.pdf'
```

---

## 📊 Sample Annotations Extracted

From your PDF, here are some key annotations identified:

| Page | Type | Content (批注) | Author |
|------|------|----------------|--------|
| 1 | Highlight | 这里是相互作用，协调的说法 | 微信用户 |
| 1 | Highlight | 除了底层的影响，环境的影响（gpu，操作系统指令集，架构等等）其他的还有啥影响 | 微信用户 |
| 1 | Highlight | 温度的影响 | 微信用户 |
| 1 | Highlight | 关键词1 | 微信用户 |
| 1 | Highlight | 返回值 | 微信用户 |
| 1 | Highlight | 预测值 | 微信用户 |
| 1 | Highlight | 必须有一个明确的、可解释的连续时间数学模型... | 微信用户 |

---

## 🔧 Requirements

```bash
pip3 install PyPDF2
```

Or use the requirements file:
```bash
pip3 install -r requirements_pdf_extraction.txt
```

---

## 📝 Key Features

✅ **Accurate Mapping**: Each annotation is linked to its page and position  
✅ **Full Content**: Extracts complete PDF text  
✅ **Multiple Formats**: JSON (structured), Markdown (readable), TXT (inline)  
✅ **Metadata Preserved**: Author, date, type, position coordinates  
✅ **Self-Healing**: Automatic error recovery built-in  

---

## 📌 Notes

1. **Position Coordinates**: The `position` field contains [x1, y1, x2, y2] coordinates from PDF coordinate system
2. **Text Mapping**: Due to PDF structure complexity, text snippets are approximate context windows around annotations
3. **For Precise Mapping**: If you need exact character-level mapping, consider using OCR-based tools with coordinate extraction

---

## 🎯 Use Cases

- **Research**: Analyze annotation patterns across academic papers
- **Review**: Track feedback and comments on drafts
- **Collaboration**: Share annotated content in structured format
- **Archival**: Preserve annotations separately from PDF
- **Analysis**: Programmatically process reviewer comments

---

## 🔍 Advanced Usage

### Extract Specific Annotation Types
Modify the script to filter by annotation type:

```python
# In extract_annotations() method
if annot_obj.get('/Subtype') in ['/Highlight', '/StrikeOut']:
    # Process only specific types
```

### Export Custom Format
Add new export methods following the pattern of `export_to_json()`, `export_to_markdown()`, etc.

---

## 📧 Contact

For issues or improvements, refer to the script's docstrings and inline comments.

**Generated**: 2026-01-30  
**Script Version**: 1.0  
**O-Award Compliance**: ✓
