"""
MCM/ICM 论文分析助手 - 优化版
结合自动提取和人工审核，提供更准确的分析结果
"""

import os
import re
import pandas as pd
from pathlib import Path
from collections import Counter
import json

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# 配置路径
BASE_DIR = Path("MCMICM")
YEARS = {
    "2022": BASE_DIR / "2022美赛优秀论文集" / "2022 美赛 C",
    "2023": BASE_DIR / "2023" / "C",
    "2024": BASE_DIR / "2024" / "C" / "student paper"
}

def extract_text_from_pdf(pdf_path, max_pages=25):
    """从PDF中提取文本 - 改进版"""
    if not PDF_AVAILABLE:
        return "", 0
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            num_pages = min(total_pages, max_pages)
            
            for page_num in range(num_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
        
        return text, total_pages
    except Exception as e:
        print(f"  ⚠️ 读取失败: {e}")
        return "", 0

def extract_abstract(text):
    """提取摘要部分"""
    # 尝试找到摘要
    abstract_patterns = [
        r'ABSTRACT\s*\n(.*?)\n(?:Keywords|Introduction|1\.|INTRODUCTION)',
        r'Abstract\s*\n(.*?)\n(?:Keywords|Introduction|1\.|INTRODUCTION)',
        r'Summary\s*\n(.*?)\n(?:Keywords|Introduction|1\.|INTRODUCTION)'
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # 限制长度
            if len(abstract) < 2000:
                return abstract[:500] + "..." if len(abstract) > 500 else abstract
    
    return "未提取到摘要"

def smart_model_extraction(text):
    """智能模型提取 - 改进版"""
    # 扩展的模型关键词库，按优先级排序
    model_patterns = {
        # 高优先级：完整模型名称
        "priority_high": {
            "Random Forest Regression": "随机森林回归",
            "Random Forest": "随机森林",
            "XGBoost": "XGBoost",
            "Gradient Boosting": "梯度提升",
            "Neural Network": "神经网络",
            "Deep Learning": "深度学习",
            "LSTM": "LSTM",
            "GRU": "GRU",
            "CNN": "卷积神经网络",
            "RNN": "循环神经网络",
            "Support Vector Machine": "支持向量机",
            "SVM": "SVM",
            "Decision Tree": "决策树",
            "K-Means": "K均值",
            "DBSCAN": "DBSCAN",
            "Hierarchical Clustering": "层次聚类",
            
            # 时间序列
            "ARIMA": "ARIMA",
            "SARIMA": "SARIMA",
            "Exponential Smoothing": "指数平滑",
            "Holt-Winters": "Holt-Winters",
            
            # 统计模型
            "Linear Regression": "线性回归",
            "Logistic Regression": "逻辑回归",
            "Polynomial Regression": "多项式回归",
            "Ridge Regression": "岭回归",
            "Lasso Regression": "Lasso回归",
            "Bayesian Network": "贝叶斯网络",
            "Naive Bayes": "朴素贝叶斯",
            
            # 优化算法
            "Linear Programming": "线性规划",
            "Integer Programming": "整数规划",
            "Mixed Integer": "混合整数规划",
            "Genetic Algorithm": "遗传算法",
            "Simulated Annealing": "模拟退火",
            "Particle Swarm Optimization": "粒子群优化",
            "PSO": "粒子群优化",
            "Ant Colony": "蚁群算法",
            
            # 其他
            "Monte Carlo Simulation": "蒙特卡洛模拟",
            "Markov Chain": "马尔可夫链",
            "Hidden Markov Model": "隐马尔可夫模型",
            "HMM": "隐马尔可夫",
            "Principal Component Analysis": "主成分分析",
            "PCA": "主成分分析",
            "Factor Analysis": "因子分析",
            "Graph Theory": "图论",
            "PageRank": "PageRank",
            "Dijkstra": "Dijkstra算法",
            "A* Algorithm": "A*算法"
        },
        
        # 中优先级：通用术语（需要上下文验证）
        "priority_medium": {
            "Regression Model": "回归模型",
            "Classification Model": "分类模型",
            "Clustering": "聚类",
            "Time Series": "时间序列",
            "Optimization": "优化模型",
            "Network Analysis": "网络分析"
        }
    }
    
    found_models = {}
    text_upper = text.upper()
    
    # 先匹配高优先级
    for eng_name, cn_name in model_patterns["priority_high"].items():
        # 使用词边界匹配，避免部分匹配
        pattern = r'\b' + re.escape(eng_name.upper()) + r'\b'
        matches = re.findall(pattern, text_upper)
        if matches:
            count = len(matches)
            # 只保留出现2次以上的（更可能是真正使用的模型）
            if count >= 2:
                found_models[cn_name] = count
    
    # 按出现频率排序
    sorted_models = sorted(found_models.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前5个最常出现的模型
    top_models = [model for model, count in sorted_models[:5]]
    
    return top_models, found_models

def extract_sections(text):
    """提取论文章节结构"""
    sections_found = []
    
    # 常见章节标题
    section_patterns = [
        (r'\b(ABSTRACT|Abstract)\b', '摘要'),
        (r'\b(INTRODUCTION|Introduction|1\.?\s*Introduction)\b', '引言'),
        (r'\b(PROBLEM\s*ANALYSIS|Problem\s*Analysis)\b', '问题分析'),
        (r'\b(ASSUMPTIONS|Assumptions)\b', '假设'),
        (r'\b(MODEL|Model|MODELING|Modeling)\b', '模型'),
        (r'\b(ALGORITHM|Algorithm)\b', '算法'),
        (r'\b(DATA|Data)\b', '数据'),
        (r'\b(RESULTS|Results)\b', '结果'),
        (r'\b(VALIDATION|Validation)\b', '验证'),
        (r'\b(SENSITIVITY|Sensitivity)\b', '敏感性分析'),
        (r'\b(CONCLUSION|Conclusion)\b', '结论'),
        (r'\b(REFERENCES|References)\b', '参考文献')
    ]
    
    for pattern, name in section_patterns:
        if re.search(pattern, text):
            sections_found.append(name)
    
    return sections_found

def count_figures_tables_improved(text):
    """改进的图表统计"""
    # 使用更精确的模式
    figures = set(re.findall(r'Figure\s+(\d+)', text, re.IGNORECASE))
    tables = set(re.findall(r'Table\s+(\d+)', text, re.IGNORECASE))
    
    return len(figures), len(tables)

def analyze_paper_smart(pdf_path):
    """智能分析单篇论文"""
    print(f"\n📄 分析: {pdf_path.name}")
    
    # 提取文本
    text, total_pages = extract_text_from_pdf(pdf_path)
    
    if not text or len(text) < 100:
        return {
            "状态": "❌ 提取失败",
            "页数": total_pages,
            "摘要": "",
            "模型": [],
            "模型详情": {},
            "章节": [],
            "图数": 0,
            "表数": 0
        }
    
    # 提取摘要
    abstract = extract_abstract(text)
    
    # 智能提取模型
    models, model_details = smart_model_extraction(text)
    
    # 提取章节
    sections = extract_sections(text)
    
    # 统计图表
    fig_count, table_count = count_figures_tables_improved(text)
    
    print(f"  ✓ 页数: {total_pages}")
    print(f"  ✓ 识别模型: {len(models)}个")
    print(f"  ✓ 图表: {fig_count}图 + {table_count}表")
    
    return {
        "状态": "✓ 完成",
        "页数": total_pages,
        "摘要": abstract,
        "模型": models,
        "模型详情": model_details,
        "章节": sections,
        "图数": fig_count,
        "表数": table_count
    }

def batch_analyze():
    """批量分析所有论文"""
    if not PDF_AVAILABLE:
        print("\n❌ 错误: 需要安装 PyPDF2")
        print("运行: pip install PyPDF2")
        return None
    
    print("="*70)
    print("📊 MCM/ICM C题论文智能分析系统 - 优化版")
    print("="*70)
    
    all_results = []
    
    for year, path in YEARS.items():
        if not path.exists():
            print(f"\n⚠️  {year}年路径不存在: {path}")
            continue
        
        pdf_files = list(path.glob("*.pdf"))
        print(f"\n{'='*70}")
        print(f"📁 {year}年 - 共{len(pdf_files)}篇论文")
        print(f"{'='*70}")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end=" ")
            
            analysis = analyze_paper_smart(pdf_file)
            
            result = {
                "年份": year,
                "论文编号": pdf_file.stem,
                "状态": analysis["状态"],
                "页数": analysis["页数"],
                "摘要预览": analysis["摘要"][:100] + "..." if len(analysis["摘要"]) > 100 else analysis["摘要"],
                "识别的模型": ", ".join(analysis["模型"]) if analysis["模型"] else "未识别",
                "模型数量": len(analysis["模型"]),
                "章节结构": ", ".join(analysis["章节"]),
                "图数量": analysis["图数"],
                "表数量": analysis["表数"],
                "图表总数": analysis["图数"] + analysis["表数"],
                
                # 需人工补充的字段
                "核心模型": "",
                "创新点": "",
                "数据来源": "",
                "验证方法": "",
                "可借鉴度": "",
                "评级": "",
                "备注": ""
            }
            
            all_results.append(result)
    
    # 保存结果
    df = pd.DataFrame(all_results)
    output_file = "论文分析结果_优化版.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ 分析完成！共处理 {len(all_results)} 篇论文")
    print(f"📁 结果已保存: {output_file}")
    print(f"{'='*70}")
    
    return df

def generate_summary_report(df):
    """生成汇总报告"""
    print("\n" + "="*70)
    print("📈 统计报告")
    print("="*70)
    
    # 年份统计
    print("\n【年份分布】")
    for year in ["2022", "2023", "2024"]:
        count = len(df[df['年份'] == year])
        if count > 0:
            print(f"  {year}年: {count}篇")
    
    # 模型统计
    print("\n【高频模型 TOP 15】")
    all_models = []
    for models_str in df['识别的模型']:
        if models_str and models_str != "未识别":
            models = [m.strip() for m in models_str.split(',')]
            all_models.extend(models)
    
    if all_models:
        model_counter = Counter(all_models)
        for i, (model, count) in enumerate(model_counter.most_common(15), 1):
            print(f"  {i:2d}. {model:20s} - {count}次")
    else:
        print("  未识别到模型")
    
    # 图表统计
    print("\n【图表使用情况】")
    total_figs = df['图数量'].sum()
    total_tables = df['表数量'].sum()
    avg_figs = df['图数量'].mean()
    avg_tables = df['表数量'].mean()
    
    print(f"  总图数: {total_figs}  平均: {avg_figs:.1f}图/篇")
    print(f"  总表数: {total_tables}  平均: {avg_tables:.1f}表/篇")
    print(f"  图表最多的论文: {df.loc[df['图表总数'].idxmax(), '论文编号']} ({df['图表总数'].max()}个)")
    
    # 保存详细报告
    report_file = "分析统计报告.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("MCM/ICM C题论文分析统计报告\n")
        f.write("="*70 + "\n\n")
        
        f.write("【年份分布】\n")
        for year in ["2022", "2023", "2024"]:
            count = len(df[df['年份'] == year])
            if count > 0:
                f.write(f"  {year}年: {count}篇\n")
        
        f.write("\n【高频模型】\n")
        if all_models:
            for i, (model, count) in enumerate(model_counter.most_common(20), 1):
                f.write(f"  {i:2d}. {model:25s} - {count}次\n")
        
        f.write(f"\n【图表统计】\n")
        f.write(f"  总图数: {total_figs}\n")
        f.write(f"  总表数: {total_tables}\n")
        f.write(f"  平均图数: {avg_figs:.2f}\n")
        f.write(f"  平均表数: {avg_tables:.2f}\n")
        
        f.write(f"\n【论文列表】\n")
        for _, row in df.iterrows():
            f.write(f"\n{row['年份']}-{row['论文编号']}\n")
            f.write(f"  模型: {row['识别的模型']}\n")
            f.write(f"  图表: {row['图数量']}图 + {row['表数量']}表\n")
    
    print(f"\n📁 详细报告已保存: {report_file}")

def create_model_database(df):
    """创建模型数据库"""
    print("\n正在创建模型知识库...")
    
    # 收集所有模型
    model_papers = {}
    for _, row in df.iterrows():
        if row['识别的模型'] and row['识别的模型'] != "未识别":
            models = [m.strip() for m in row['识别的模型'].split(',')]
            for model in models:
                if model not in model_papers:
                    model_papers[model] = []
                model_papers[model].append(f"{row['年份']}-{row['论文编号']}")
    
    # 创建数据库
    db_data = []
    for model, papers in sorted(model_papers.items(), key=lambda x: len(x[1]), reverse=True):
        db_data.append({
            "模型名称": model,
            "使用次数": len(papers),
            "使用论文": "; ".join(papers[:10]),  # 最多列10篇
            "2022年": sum(1 for p in papers if p.startswith("2022")),
            "2023年": sum(1 for p in papers if p.startswith("2023")),
            "2024年": sum(1 for p in papers if p.startswith("2024")),
            "适用场景": "",  # 人工补充
            "难度评估": "",  # 人工补充
            "推荐指数": ""   # 人工补充
        })
    
    db_df = pd.DataFrame(db_data)
    db_file = "C题模型知识库_优化版.xlsx"
    db_df.to_excel(db_file, index=False)
    print(f"✅ 模型知识库已保存: {db_file}")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 MCM/ICM C题论文智能分析系统 - 优化版")
    print("="*70)
    
    if not PDF_AVAILABLE:
        print("\n❌ 需要安装 PyPDF2")
        print("运行: pip install PyPDF2")
        return
    
    print("\n📋 功能说明:")
    print("  1. 智能提取PDF内容（改进的文本提取）")
    print("  2. 精准识别模型（基于频率和上下文）")
    print("  3. 提取摘要和章节结构")
    print("  4. 准确统计图表数量")
    print("  5. 生成详细统计报告")
    print("  6. 创建模型知识库")
    
    print("\n⚠️  重要提示:")
    print("  - 自动识别结果仅供参考")
    print("  - 建议人工复核'核心模型'、'创新点'等字段")
    print("  - 可在Excel中直接编辑补充信息")
    
    input("\n按回车键开始分析...")
    
    # 批量分析
    df = batch_analyze()
    
    if df is not None and len(df) > 0:
        # 生成报告
        generate_summary_report(df)
        
        # 创建知识库
        create_model_database(df)
        
        print("\n" + "="*70)
        print("🎉 所有任务完成！")
        print("="*70)
        print("\n📁 生成的文件:")
        print("  1. 论文分析结果_优化版.xlsx - 详细分析结果（可编辑）")
        print("  2. 分析统计报告.txt - 统计汇总")
        print("  3. C题模型知识库_优化版.xlsx - 模型使用统计")
        
        print("\n💡 下一步建议:")
        print("  1. 打开Excel文件，人工复核和补充信息")
        print("  2. 重点关注'识别的模型'列，确认是否准确")
        print("  3. 补充'核心模型'、'创新点'、'可借鉴度'等字段")
        print("  4. 参考统计报告，了解整体趋势")

if __name__ == "__main__":
    main()
