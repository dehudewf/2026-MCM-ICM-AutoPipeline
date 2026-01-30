#!/usr/bin/env python3
"""
A题论文分析脚本 - OCR版本
使用OCR处理扫描PDF，提取文本进行分析
"""

import os
import re
import subprocess
from collections import Counter
import csv
import tempfile
from concurrent.futures import ThreadPoolExecutor
import json

# 目录
PAPER_DIR = "/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/pre-theis"
OUTPUT_DIR = "/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/知识库"

# A题模型关键词 (中英文混合，适应翻译版PDF)
MODEL_KEYWORDS = {
    # 微分方程类
    "常微分方程": ["ODE", "ordinary differential equation", "常微分", "微分方程", "初值问题"],
    "偏微分方程": ["PDE", "partial differential equation", "偏微分", "热传导", "波动方程", "扩散方程", "拉普拉斯"],
    "Navier-Stokes": ["Navier-Stokes", "NS方程", "纳维", "斯托克斯", "流体动力学"],
    "反应扩散方程": ["reaction-diffusion", "反应扩散", "Fisher方程", "Gray-Scott"],
    "Lotka-Volterra": ["Lotka-Volterra", "捕食者", "猎物", "predator-prey", "种群动力学"],
    "SIR/SEIR模型": ["SIR", "SEIR", "SIS", "传染病", "epidemic", "仓室模型", "基本再生数", "R0"],
    
    # 优化类
    "线性规划": ["linear programming", "线性规划", "LP", "单纯形", "目标函数", "约束条件"],
    "整数规划": ["integer programming", "整数规划", "IP", "MIP", "混合整数", "0-1规划"],
    "非线性规划": ["nonlinear programming", "非线性规划", "NLP", "KKT条件"],
    "动态规划": ["dynamic programming", "动态规划", "DP", "Bellman", "最优子结构", "状态转移"],
    "多目标优化": ["multi-objective", "多目标", "Pareto", "帕累托", "权重法", "加权和"],
    "遗传算法": ["genetic algorithm", "遗传算法", "GA", "进化算法", "染色体", "交叉", "变异", "选择"],
    "粒子群优化": ["particle swarm", "粒子群", "PSO", "群体智能"],
    "模拟退火": ["simulated annealing", "模拟退火", "SA", "Metropolis"],
    "蚁群算法": ["ant colony", "蚁群", "ACO", "信息素"],
    "梯度下降": ["gradient descent", "梯度下降", "最速下降", "牛顿法"],
    
    # 统计/回归类
    "线性回归": ["linear regression", "线性回归", "OLS", "最小二乘", "回归分析"],
    "多元回归": ["multiple regression", "多元回归", "multivariate", "多变量"],
    "岭回归": ["ridge regression", "岭回归", "Ridge", "正则化"],
    "Lasso回归": ["Lasso", "套索回归", "L1正则"],
    "逻辑回归": ["logistic regression", "逻辑回归", "Logistic", "二分类"],
    "主成分分析": ["PCA", "principal component", "主成分", "降维", "特征值分解"],
    "因子分析": ["factor analysis", "因子分析", "潜变量"],
    "方差分析": ["ANOVA", "方差分析", "F检验"],
    
    # 时间序列类
    "ARIMA": ["ARIMA", "自回归", "移动平均", "ARMA", "差分"],
    "时间序列分析": ["time series", "时间序列", "平稳性"],
    "傅里叶分析": ["Fourier", "傅里叶", "频谱", "spectral", "频域"],
    "小波分析": ["wavelet", "小波", "多分辨率"],
    
    # 机器学习类
    "神经网络": ["neural network", "神经网络", "ANN", "MLP", "反向传播", "激活函数"],
    "深度学习": ["deep learning", "深度学习", "CNN", "卷积", "RNN"],
    "LSTM": ["LSTM", "长短期记忆", "门控"],
    "随机森林": ["random forest", "随机森林", "RF", "袋装"],
    "支持向量机": ["SVM", "support vector", "支持向量", "核函数"],
    "K均值": ["K-means", "K均值", "聚类", "clustering", "质心"],
    "层次聚类": ["hierarchical clustering", "层次聚类", "树状图"],
    "决策树": ["decision tree", "决策树", "信息增益", "基尼"],
    "XGBoost": ["XGBoost", "梯度提升", "Boosting"],
    "贝叶斯方法": ["Bayesian", "贝叶斯", "后验", "先验", "MCMC"],
    
    # 仿真/数值方法类
    "蒙特卡洛模拟": ["Monte Carlo", "蒙特卡洛", "MC模拟", "随机模拟", "抽样"],
    "有限元分析": ["finite element", "有限元", "FEM", "FEA", "单元", "刚度矩阵"],
    "有限差分法": ["finite difference", "有限差分", "FDM", "差分格式"],
    "龙格-库塔法": ["Runge-Kutta", "龙格库塔", "RK4", "RK45"],
    "欧拉法": ["Euler method", "欧拉法", "欧拉方法", "显式", "隐式"],
    "元胞自动机": ["cellular automaton", "元胞自动机", "CA", "邻域规则"],
    "Agent-Based": ["agent-based", "基于代理", "ABM", "多智能体", "个体模型"],
    "离散事件仿真": ["discrete event", "离散事件", "DES", "事件调度"],
    
    # 图论/网络类
    "图论": ["graph theory", "图论", "网络分析", "节点", "边"],
    "最短路径": ["shortest path", "最短路径", "Dijkstra", "Floyd", "A*"],
    "网络流": ["network flow", "网络流", "最大流", "最小割", "Ford-Fulkerson"],
    "复杂网络": ["complex network", "复杂网络", "scale-free", "无标度", "小世界"],
    "最小生成树": ["minimum spanning tree", "最小生成树", "MST", "Kruskal", "Prim"],
    
    # 博弈论/决策类
    "博弈论": ["game theory", "博弈论", "Nash", "纳什均衡", "策略"],
    "马尔可夫链": ["Markov chain", "马尔可夫", "转移矩阵", "状态转移"],
    "马尔可夫决策过程": ["MDP", "Markov decision", "马尔可夫决策", "贝尔曼方程"],
    "层次分析法": ["AHP", "层次分析", "analytic hierarchy", "判断矩阵"],
    "TOPSIS": ["TOPSIS", "逼近理想解", "正负理想解"],
    "熵权法": ["entropy weight", "熵权", "信息熵"],
    "灰色预测": ["grey prediction", "灰色预测", "GM(1,1)", "灰色系统"],
    
    # 物理模型类
    "流体力学": ["fluid dynamics", "流体力学", "CFD", "计算流体", "雷诺数"],
    "传热学": ["heat transfer", "传热", "热传导", "thermal", "傅里叶定律"],
    "扩散模型": ["diffusion model", "扩散模型", "Fick", "布朗运动"],
    "弹性力学": ["elasticity", "弹性", "应力", "应变", "胡克定律"],
    
    # 验证方法
    "敏感性分析": ["sensitivity analysis", "敏感性分析", "敏感度", "参数扰动"],
    "不确定性分析": ["uncertainty", "不确定性", "误差分析", "置信区间"],
    "稳定性分析": ["stability analysis", "稳定性", "Lyapunov", "特征值"],
    "交叉验证": ["cross-validation", "交叉验证", "k-fold", "留一法"],
}

# 可视化类型
VISUALIZATION_TYPES = {
    "相图/相平面图": ["phase", "相图", "相平面", "轨线"],
    "向量场图": ["vector field", "向量场", "流场", "方向场"],
    "轨迹图": ["trajectory", "轨迹", "orbit", "演化轨迹"],
    "等高线图": ["contour", "等高线", "等值线"],
    "热力图": ["heatmap", "热力图", "热图", "热分布"],
    "3D曲面图": ["3D surface", "三维曲面", "surface plot", "曲面图"],
    "时间演化图": ["time evolution", "时间演化", "演化过程", "动态变化"],
    "分岔图": ["bifurcation", "分岔", "分叉图"],
    "稳定性图": ["stability", "稳定性", "稳定区域"],
    "Pareto前沿图": ["Pareto", "帕累托前沿", "最优解"],
    "收敛曲线": ["convergence", "收敛", "迭代曲线"],
    "误差分析图": ["error", "误差", "精度分析"],
    "残差图": ["residual", "残差", "拟合残差"],
    "散点图": ["scatter", "散点", "相关性"],
    "折线图": ["line chart", "折线", "趋势图"],
    "柱状图": ["bar chart", "柱状", "直方图"],
    "饼图": ["pie chart", "饼图", "占比"],
    "箱线图": ["box plot", "箱线", "分位数"],
    "流程图": ["flowchart", "流程图", "框图", "技术路线"],
    "模型架构图": ["architecture", "架构图", "结构图", "系统图"],
    "地图": ["map", "地图", "地理", "空间分布"],
    "网络图": ["network", "网络图", "拓扑"],
    "树状图": ["tree", "树状", "层次图", "dendrogram"],
    "敏感性分析图": ["sensitivity", "敏感性", "龙卷风图"],
    "网格划分图": ["mesh", "网格", "有限元网格"],
    "温度场图": ["temperature field", "温度场", "温度分布"],
    "浓度分布图": ["concentration", "浓度分布", "物质分布"],
    "速度场图": ["velocity field", "速度场", "流速"],
}


def extract_text_ocr(pdf_path, max_pages=30):
    """使用OCR提取PDF文本 (仅处理前N页节省时间)"""
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 转换PDF为图片
            result = subprocess.run([
                "pdftoppm", "-png", "-r", "150",  # 150 DPI for speed
                "-l", str(max_pages),  # 只处理前max_pages页
                pdf_path, os.path.join(temp_dir, "page")
            ], capture_output=True, timeout=60)
            
            if result.returncode != 0:
                return ""
            
            # OCR每张图片
            text = ""
            png_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
            
            for png_file in png_files[:max_pages]:
                png_path = os.path.join(temp_dir, png_file)
                ocr_result = subprocess.run([
                    "tesseract", png_path, "-", "-l", "chi_sim+eng"
                ], capture_output=True, text=True, timeout=30)
                
                if ocr_result.returncode == 0:
                    text += ocr_result.stdout + "\n"
            
            return text
    except Exception as e:
        print(f"    OCR失败: {e}")
        return ""


def extract_text_pypdf(pdf_path):
    """使用pypdf提取文本"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except:
        return ""


def extract_text(pdf_path):
    """尝试多种方法提取文本"""
    # 先尝试pypdf (最快)
    text = extract_text_pypdf(pdf_path)
    if len(text.strip()) > 500:  # 有效内容足够多
        return text, "pypdf"
    
    # 再尝试pdftotext
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 500:
            return result.stdout, "pdftotext"
    except:
        pass
    
    # 最后使用OCR
    text = extract_text_ocr(pdf_path)
    if text:
        return text, "ocr"
    
    return "", "failed"


def identify_models(text):
    """识别模型"""
    found_models = []
    text_lower = text.lower()
    
    for model_name, keywords in MODEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_models.append(model_name)
                break
    
    return list(set(found_models))


def identify_visualizations(text):
    """识别可视化类型"""
    found_viz = []
    text_lower = text.lower()
    
    for viz_type, keywords in VISUALIZATION_TYPES.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_viz.append(viz_type)
                break
    
    return list(set(found_viz))


def count_figures_tables(text):
    """统计图表数量"""
    fig_patterns = [r'Figure\s*\d+', r'Fig\.\s*\d+', r'图\s*\d+', r'FIGURE\s*\d+']
    table_patterns = [r'Table\s*\d+', r'表\s*\d+', r'TABLE\s*\d+']
    
    figures = set()
    tables = set()
    
    for pattern in fig_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        figures.update([m.lower() for m in matches])
    
    for pattern in table_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tables.update([m.lower() for m in matches])
    
    return len(figures), len(tables)


def get_paper_info(filename):
    """从文件名获取论文信息"""
    match = re.search(r'(\d{7})', filename)
    paper_id = match.group(1) if match else filename.replace(".pdf", "")
    
    year_prefix = paper_id[:2] if len(paper_id) >= 2 else "??"
    year_map = {"20": "2020", "21": "2021", "22": "2022", "23": "2023", "24": "2024", "25": "2025"}
    year = year_map.get(year_prefix, "未知")
    
    return paper_id, year


def main():
    print("=" * 70)
    print("MCM/ICM A题论文全面分析系统 (OCR增强版)")
    print("=" * 70)
    
    pdf_files = sorted([f for f in os.listdir(PAPER_DIR) if f.endswith('.pdf')])
    print(f"\n找到 {len(pdf_files)} 篇论文待分析...")
    print("注: OCR处理较慢，请耐心等待...\n")
    
    papers = []
    all_models = Counter()
    all_visualizations = Counter()
    year_counts = Counter()
    total_figures = 0
    total_tables = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(PAPER_DIR, pdf_file)
        paper_id, year = get_paper_info(pdf_file)
        
        print(f"[{i}/{len(pdf_files)}] 分析: {paper_id} ({year}年) ", end="", flush=True)
        
        text, method = extract_text(pdf_path)
        
        if not text or len(text.strip()) < 100:
            print("⚠️ 文本提取失败")
            continue
        
        print(f"({method}) ", end="", flush=True)
        
        models = identify_models(text)
        figures, tables = count_figures_tables(text)
        visualizations = identify_visualizations(text)
        
        for m in models:
            all_models[m] += 1
        for v in visualizations:
            all_visualizations[v] += 1
        year_counts[year] += 1
        total_figures += figures
        total_tables += tables
        
        papers.append({
            "id": paper_id,
            "year": year,
            "models": models,
            "figures": figures,
            "tables": tables,
            "visualizations": visualizations,
            "extraction_method": method
        })
        
        print(f"✓ 模型:{len(models)} 图:{figures} 表:{tables}")
    
    print("\n" + "=" * 70)
    print("生成知识库文件...")
    
    # 生成报告
    generate_report(papers, all_models, all_visualizations, year_counts, total_figures, total_tables)
    generate_model_csv(all_models)
    generate_viz_csv(all_visualizations)
    
    print("\n" + "=" * 70)
    print(f"分析完成! 成功分析: {len(papers)}/{len(pdf_files)} 篇")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


def generate_report(papers, all_models, all_viz, year_counts, total_figures, total_tables):
    """生成分析报告"""
    output_path = os.path.join(OUTPUT_DIR, "A题分析统计报告.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MCM/ICM A题论文分析统计报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【年份分布】\n")
        for year in sorted(year_counts.keys()):
            f.write(f"  {year}年: {year_counts[year]}篇\n")
        f.write(f"\n  总计: {len(papers)}篇\n\n")
        
        f.write("【高频模型】（A题特色: 连续建模/微分方程/优化）\n")
        for i, (model, count) in enumerate(all_models.most_common(35), 1):
            f.write(f"  {i:2d}. {model:24s} - {count}次\n")
        f.write("\n")
        
        f.write("【A题模型分类统计】\n")
        categories = {
            "微分方程类": ["常微分方程", "偏微分方程", "Navier-Stokes", "反应扩散方程", "Lotka-Volterra", "SIR/SEIR模型"],
            "优化算法类": ["线性规划", "整数规划", "非线性规划", "动态规划", "多目标优化", "遗传算法", "粒子群优化", "模拟退火", "蚁群算法", "梯度下降"],
            "数值仿真类": ["蒙特卡洛模拟", "有限元分析", "有限差分法", "龙格-库塔法", "欧拉法", "元胞自动机", "Agent-Based", "离散事件仿真"],
            "统计分析类": ["线性回归", "多元回归", "主成分分析", "因子分析", "方差分析", "贝叶斯方法"],
            "机器学习类": ["神经网络", "深度学习", "LSTM", "随机森林", "支持向量机", "K均值", "决策树", "XGBoost"],
            "图论网络类": ["图论", "最短路径", "网络流", "复杂网络", "最小生成树"],
            "决策评估类": ["博弈论", "马尔可夫链", "层次分析法", "TOPSIS", "熵权法", "灰色预测"],
            "物理模型类": ["流体力学", "传热学", "扩散模型", "弹性力学"],
            "验证分析类": ["敏感性分析", "不确定性分析", "稳定性分析", "交叉验证"],
        }
        
        for cat_name, models in categories.items():
            cat_count = sum(all_models.get(m, 0) for m in models)
            if cat_count > 0:
                f.write(f"\n  {cat_name}: {cat_count}次\n")
                for m in models:
                    if all_models.get(m, 0) > 0:
                        f.write(f"    └─ {m}: {all_models[m]}次\n")
        f.write("\n")
        
        f.write("【高频可视化类型】\n")
        for i, (viz, count) in enumerate(all_viz.most_common(25), 1):
            f.write(f"  {i:2d}. {viz:20s} - {count}次\n")
        f.write("\n")
        
        f.write("【图表统计】\n")
        f.write(f"  总图数: {total_figures}\n")
        f.write(f"  总表数: {total_tables}\n")
        if papers:
            f.write(f"  平均图数: {total_figures/len(papers):.2f}\n")
            f.write(f"  平均表数: {total_tables/len(papers):.2f}\n")
        f.write("\n")
        
        f.write("【论文详情】\n\n")
        for p in sorted(papers, key=lambda x: x['id']):
            f.write(f"{p['year']}-{p['id']}\n")
            f.write(f"  提取方法: {p['extraction_method']}\n")
            if p['models']:
                f.write(f"  模型: {', '.join(p['models'])}\n")
            else:
                f.write(f"  模型: 未识别\n")
            f.write(f"  图表: {p['figures']}图 + {p['tables']}表\n")
            if p['visualizations']:
                viz_str = ', '.join(p['visualizations'][:6])
                if len(p['visualizations']) > 6:
                    viz_str += "..."
                f.write(f"  可视化: {viz_str}\n")
            f.write("\n")
    
    print(f"✓ {output_path}")


def generate_model_csv(all_models):
    """生成模型知识库CSV"""
    output_path = os.path.join(OUTPUT_DIR, "A题模型知识库.csv")
    
    model_info = {
        "常微分方程": ("微分方程", "描述变量随时间变化的方程", "人口动态/化学反应/传染病传播/机械系统", "物理意义明确;数学严谨;可解析求解", "复杂系统难以求解;需要初值条件", "初始条件;系统参数;时间范围"),
        "偏微分方程": ("微分方程", "描述变量随时空变化的方程", "传热/扩散/流体/波动问题", "描述连续介质;空间分布建模", "求解复杂;需数值方法", "边界条件;初始条件;域几何"),
        "SIR/SEIR模型": ("传染病模型", "仓室模型描述疾病传播", "传染病/信息扩散/谣言传播", "结构简单;参数可解释;经典验证", "均匀混合假设;忽略空间结构", "传染率β;恢复率γ;基本再生数R0"),
        "Lotka-Volterra": ("生态模型", "捕食者-猎物相互作用模型", "生态系统/竞争动态/资源消耗", "周期行为;生态学经典", "简化假设;稳定性敏感", "增长率;捕食率;死亡率"),
        "线性规划": ("数学优化", "线性约束下优化线性目标", "资源分配/生产计划/运输问题", "最优解保证;计算高效", "仅线性关系;连续变量", "目标系数;约束矩阵;边界"),
        "遗传算法": ("元启发式", "模拟生物进化的优化算法", "复杂优化/参数标定/路径规划", "全局搜索;并行;非线性", "不保证最优;参数敏感", "种群大小;交叉概率;变异概率"),
        "蒙特卡洛模拟": ("随机模拟", "随机抽样估计数学期望", "不确定性量化/风险评估/积分估计", "处理复杂分布;灵活;无需解析解", "计算量大;收敛慢", "抽样次数;分布参数;种子"),
        "有限元分析": ("数值方法", "离散为有限单元求解PDE", "结构分析/传热/流体/电磁", "复杂几何;成熟软件;工程标准", "网格依赖;计算量大", "网格尺寸;单元类型;边界条件"),
        "有限差分法": ("数值方法", "差分近似导数求解微分方程", "规则网格PDE/时间演化问题", "实现简单;理论清晰", "复杂边界困难;稳定性条件", "空间步长;时间步长;差分格式"),
        "元胞自动机": ("复杂系统", "基于局部规则的离散演化系统", "火灾蔓延/城市扩张/交通流", "并行计算;直观;涌现行为", "规则设计主观;参数标定", "邻域定义;转移规则;边界条件"),
        "Agent-Based": ("复杂系统", "模拟个体代理交互产生涌现", "社会系统/生态系统/疏散模拟", "个体异质性;涌现行为;直观", "参数标定困难;验证难", "代理规则;交互范围;环境设置"),
        "马尔可夫链": ("随机过程", "状态转移只依赖当前状态", "状态预测/排队系统/随机游走", "数学可处理;稳态分析;概率解释", "无记忆假设;状态离散化", "状态空间;转移概率矩阵"),
        "敏感性分析": ("模型验证", "研究参数变化对输出影响", "参数重要性/不确定性传播/验证", "识别关键参数;增强可信度", "计算量大;局部vs全局", "参数范围;变化幅度;输出指标"),
        "主成分分析": ("降维方法", "正交变换将相关变量转为主成分", "降维/特征提取/数据可视化", "方差最大化;去相关;降噪", "线性假设;解释性降低", "保留主成分数;方差贡献阈值"),
        "神经网络": ("机器学习", "受生物神经网络启发的计算模型", "模式识别/非线性建模/预测", "非线性拟合;自学习;通用近似", "黑箱;需大量数据;过拟合", "层数;神经元数;学习率;激活函数"),
        "随机森林": ("集成学习", "多决策树的集成方法", "分类/回归/特征重要性", "抗过拟合;处理高维;特征选择", "计算量大;解释性降低", "树数量;最大深度;特征数"),
        "贝叶斯方法": ("统计推断", "基于贝叶斯定理更新概率", "参数估计/预测/不确定性量化", "先验知识融合;不确定性;小样本", "先验主观;计算复杂", "先验分布;似然函数;后验采样"),
        "动态规划": ("数学优化", "通过子问题求解最优决策", "序贯决策/路径规划/资源分配", "最优性保证;递归结构", "维度灾难;状态离散化", "状态空间;决策集;转移函数;目标函数"),
        "流体力学": ("物理模型", "研究流体运动规律", "气流/水流/交通流/血流", "物理准确;广泛验证", "复杂流动难以求解", "雷诺数;边界条件;初始场"),
        "传热学": ("物理模型", "研究热量传递规律", "温度分布/热设计/能源系统", "物理原理清晰;工程成熟", "辐射复杂;相变处理", "热导率;边界条件;热源"),
    }
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["模型名称", "类别", "描述", "适用场景", "优点", "缺点", "关键参数", "出现次数"])
        
        for model, count in all_models.most_common():
            if model in model_info:
                info = model_info[model]
                writer.writerow([model, info[0], info[1], info[2], info[3], info[4], info[5], count])
            else:
                writer.writerow([model, "待分类", "待补充", "待补充", "待补充", "待补充", "待补充", count])
    
    print(f"✓ {output_path}")


def generate_viz_csv(all_viz):
    """生成可视化知识库CSV"""
    output_path = os.path.join(OUTPUT_DIR, "A题可视化知识库.csv")
    
    viz_info = {
        "相图/相平面图": ("展示动力系统状态空间轨迹", "直观展示系统行为;识别稳定性", "仅2-3维系统;需微分方程背景", "微分方程/SIR模型/生态系统", "微分方程解/状态变量"),
        "向量场图": ("展示微分方程定义的流向", "直观展示系统流向;预测轨迹", "箭头密度需调节;计算量大", "ODE分析/流场/方向场", "微分方程定义/网格点"),
        "等高线图": ("展示二维函数值的等值线", "空间分布清晰;梯度明确", "等级数选择影响", "温度场/浓度分布/优化目标", "二维网格/函数值矩阵"),
        "热力图": ("颜色深浅展示数值空间分布", "直观;便于识别热点", "色彩映射影响解读", "传热/密度分布/相关性矩阵", "二维数值矩阵"),
        "3D曲面图": ("展示三维空间曲面形态", "立体直观;展示复杂关系", "视角影响理解", "地形建模/响应面/优化景观", "X-Y网格/Z高度值"),
        "时间演化图": ("展示变量随时间变化", "动态过程清晰;趋势明显", "长时间序列可能拥挤", "动态模拟/状态变化/预测", "时间点/状态值"),
        "分岔图": ("展示参数变化导致的系统分岔", "复杂性可视;临界点识别", "参数扫描计算量大", "非线性系统/混沌分析", "参数范围/系统稳态值"),
        "Pareto前沿图": ("展示多目标优化非支配解", "权衡关系清晰;决策支持", "高维难展示", "多目标优化/权衡分析", "目标函数值/非支配解"),
        "收敛曲线": ("展示迭代算法收敛过程", "收敛速度直观;终止验证", "需足够迭代次数", "数值方法/优化算法", "迭代次数/目标值"),
        "流程图": ("展示算法/模型逻辑流程", "逻辑清晰;易于理解", "复杂系统难简化", "算法描述/技术路线/模型框架", "步骤定义/连接关系"),
        "散点图": ("展示两变量关系分布", "相关性直观;异常值识别", "点过多时重叠", "相关性分析/回归拟合", "变量X/变量Y"),
        "折线图": ("展示数据随时间变化趋势", "趋势清晰;便于对比", "数据点多时拥挤", "趋势分析/时间序列/模型对比", "自变量/因变量"),
        "敏感性分析图": ("展示参数对结果的影响", "关键参数识别;鲁棒性评估", "参数多时复杂", "模型验证/参数重要性", "参数范围/响应变量"),
        "网格划分图": ("展示有限元网格结构", "网格质量可视;离散化验证", "复杂网格难展示", "有限元分析/CFD", "节点坐标/单元连接"),
        "地图": ("地理空间上展示数据分布", "空间关系直观;地理背景", "需地理数据;投影影响", "地理分布/传播模拟", "坐标/属性值"),
        "网络图": ("展示节点连接网络结构", "关系可视化;结构分析", "大网络难展示", "网络分析/传播路径", "节点列表/边列表"),
        "误差分析图": ("展示数值方法误差和收敛性", "精度评估;方法验证", "需真解对比", "数值验证/方法对比", "网格尺寸/误差值"),
        "温度场图": ("展示温度空间分布", "热分布直观;设计指导", "需热分析结果", "传热问题/热设计", "空间坐标/温度值"),
        "速度场图": ("展示流体速度空间分布", "流动特性清晰;流态识别", "复杂流动难展示", "流体力学/CFD", "空间坐标/速度向量"),
    }
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["图表类型", "作用/功能", "优点", "缺点", "适用场景/触发关键词", "输入数据要求", "出现次数"])
        
        for viz, count in all_viz.most_common():
            if viz in viz_info:
                info = viz_info[viz]
                writer.writerow([viz, info[0], info[1], info[2], info[3], info[4], count])
            else:
                writer.writerow([viz, "待补充", "待补充", "待补充", viz, "待补充", count])
    
    print(f"✓ {output_path}")


if __name__ == "__main__":
    main()
