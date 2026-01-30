#!/usr/bin/env python3
"""
A题论文分析脚本 - 全面分析MCM/ICM A题O奖论文
生成: 分析统计报告 + 模型库 + 可视化知识库
"""

import os
import re
from collections import Counter, defaultdict
from pypdf import PdfReader
import csv

# A题论文目录
PAPER_DIR = "/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/pre-theis"
OUTPUT_DIR = "/Users/xiaohuiwei/Downloads/肖惠威美赛/A题/知识库"

# A题常用模型关键词映射 (针对连续建模问题)
MODEL_KEYWORDS = {
    # 微分方程类
    "常微分方程": ["ODE", "ordinary differential equation", "常微分", "微分方程"],
    "偏微分方程": ["PDE", "partial differential equation", "偏微分", "热传导方程", "波动方程", "扩散方程"],
    "Navier-Stokes": ["Navier-Stokes", "NS方程", "纳维-斯托克斯"],
    "反应扩散方程": ["reaction-diffusion", "反应扩散", "Fisher方程"],
    "Lotka-Volterra": ["Lotka-Volterra", "捕食者-猎物", "predator-prey"],
    "SIR/SEIR模型": ["SIR", "SEIR", "SIS", "传染病模型", "epidemic", "compartmental"],
    
    # 优化类
    "线性规划": ["linear programming", "线性规划", "LP", "单纯形"],
    "整数规划": ["integer programming", "整数规划", "IP", "MIP", "混合整数"],
    "非线性规划": ["nonlinear programming", "非线性规划", "NLP"],
    "动态规划": ["dynamic programming", "动态规划", "DP", "Bellman"],
    "多目标优化": ["multi-objective", "多目标", "Pareto", "帕累托"],
    "遗传算法": ["genetic algorithm", "遗传算法", "GA", "进化算法"],
    "粒子群优化": ["particle swarm", "粒子群", "PSO"],
    "模拟退火": ["simulated annealing", "模拟退火", "SA"],
    "蚁群算法": ["ant colony", "蚁群", "ACO"],
    
    # 统计/回归类
    "线性回归": ["linear regression", "线性回归", "OLS", "最小二乘"],
    "多元回归": ["multiple regression", "多元回归", "multivariate"],
    "岭回归": ["ridge regression", "岭回归", "Ridge"],
    "Lasso回归": ["Lasso", "套索回归"],
    "逻辑回归": ["logistic regression", "逻辑回归", "Logistic"],
    "主成分分析": ["PCA", "principal component", "主成分分析"],
    "因子分析": ["factor analysis", "因子分析"],
    
    # 时间序列类
    "ARIMA": ["ARIMA", "自回归", "移动平均", "ARMA"],
    "时间序列分析": ["time series", "时间序列"],
    "傅里叶分析": ["Fourier", "傅里叶", "频谱分析", "spectral"],
    "小波分析": ["wavelet", "小波"],
    
    # 机器学习类
    "神经网络": ["neural network", "神经网络", "ANN", "MLP"],
    "深度学习": ["deep learning", "深度学习", "CNN", "卷积神经网络"],
    "LSTM": ["LSTM", "长短期记忆"],
    "随机森林": ["random forest", "随机森林", "RF"],
    "支持向量机": ["SVM", "support vector", "支持向量机"],
    "K均值": ["K-means", "K均值", "聚类", "clustering"],
    "决策树": ["decision tree", "决策树"],
    "XGBoost": ["XGBoost", "梯度提升"],
    "贝叶斯方法": ["Bayesian", "贝叶斯", "后验", "先验"],
    
    # 仿真/数值方法类
    "蒙特卡洛模拟": ["Monte Carlo", "蒙特卡洛", "MC模拟", "随机模拟"],
    "有限元分析": ["finite element", "有限元", "FEM", "FEA"],
    "有限差分法": ["finite difference", "有限差分", "FDM"],
    "龙格-库塔法": ["Runge-Kutta", "龙格库塔", "RK4"],
    "欧拉法": ["Euler method", "欧拉法", "欧拉方法"],
    "元胞自动机": ["cellular automaton", "元胞自动机", "CA"],
    "Agent-Based": ["agent-based", "基于代理", "ABM", "多智能体"],
    
    # 图论/网络类
    "图论": ["graph theory", "图论", "网络分析"],
    "最短路径": ["shortest path", "最短路径", "Dijkstra", "Floyd"],
    "网络流": ["network flow", "网络流", "最大流", "最小割"],
    "复杂网络": ["complex network", "复杂网络", "scale-free", "无标度"],
    
    # 博弈论/决策类
    "博弈论": ["game theory", "博弈论", "Nash", "纳什均衡"],
    "马尔可夫链": ["Markov chain", "马尔可夫", "转移矩阵"],
    "马尔可夫决策过程": ["MDP", "Markov decision", "马尔可夫决策"],
    "层次分析法": ["AHP", "层次分析", "analytic hierarchy"],
    "TOPSIS": ["TOPSIS", "逼近理想解"],
    
    # 物理模型类
    "流体力学": ["fluid dynamics", "流体力学", "CFD", "计算流体"],
    "传热学": ["heat transfer", "传热", "热传导", "thermal"],
    "扩散模型": ["diffusion model", "扩散模型", "Fick"],
    "弹性力学": ["elasticity", "弹性", "应力", "应变"],
    
    # 其他
    "敏感性分析": ["sensitivity analysis", "敏感性分析", "敏感度"],
    "不确定性分析": ["uncertainty", "不确定性", "误差分析"],
    "稳定性分析": ["stability analysis", "稳定性分析", "Lyapunov"],
}

# A题可视化类型 (针对连续建模特点)
VISUALIZATION_TYPES = {
    # 微分方程相关
    "相图/相平面图": ["phase", "相图", "相平面", "phase plane", "相空间"],
    "向量场图": ["vector field", "向量场", "流场", "方向场"],
    "轨迹图": ["trajectory", "轨迹", "orbit", "路径"],
    "稳定性图": ["stability", "稳定性", "平衡点", "equilibrium"],
    "分岔图": ["bifurcation", "分岔", "分叉"],
    
    # 空间分布相关
    "等高线图": ["contour", "等高线", "等值线"],
    "热力图/温度场": ["heatmap", "热力图", "温度场", "temperature field"],
    "密度分布图": ["density", "密度", "分布图"],
    "3D曲面图": ["3D surface", "三维曲面", "surface plot", "曲面"],
    "空间分布图": ["spatial distribution", "空间分布"],
    
    # 时间演化相关
    "时间演化图": ["time evolution", "时间演化", "temporal"],
    "动态过程图": ["dynamic", "动态", "演化过程"],
    "稳态解图": ["steady state", "稳态", "平衡态"],
    "瞬态响应图": ["transient", "瞬态", "响应"],
    
    # 数值方法相关
    "网格划分图": ["mesh", "网格", "grid"],
    "收敛性图": ["convergence", "收敛", "迭代"],
    "误差分析图": ["error", "误差", "精度"],
    "残差图": ["residual", "残差"],
    
    # 优化相关
    "Pareto前沿图": ["Pareto", "帕累托前沿", "非支配解"],
    "适应度曲线": ["fitness", "适应度", "收敛曲线"],
    "搜索空间图": ["search space", "搜索空间"],
    
    # 统计分析相关
    "散点图": ["scatter", "散点", "相关性"],
    "折线图": ["line chart", "折线", "趋势"],
    "柱状图": ["bar chart", "柱状", "对比"],
    "箱线图": ["box plot", "箱线", "分位数"],
    "直方图": ["histogram", "直方", "频率分布"],
    
    # 模型结构相关
    "流程图": ["flowchart", "流程图", "框图", "block diagram"],
    "模型架构图": ["architecture", "架构", "结构图"],
    "因果关系图": ["causal", "因果", "关系图"],
    
    # 地理/空间相关
    "地图": ["map", "地图", "地理", "GIS"],
    "地形图": ["terrain", "地形", "高程"],
    
    # 网络相关
    "网络图": ["network", "网络", "节点", "连接"],
    "树状图": ["tree", "树状", "层次"],
    
    # 其他
    "敏感性分析图": ["sensitivity", "敏感性", "参数影响"],
    "不确定性量化图": ["uncertainty quantification", "不确定性", "置信区间"],
}


def extract_pdf_text(pdf_path):
    """提取PDF文本"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"  警告: 无法读取 {pdf_path}: {e}")
        return ""


def identify_models(text):
    """识别论文中使用的模型"""
    found_models = []
    text_lower = text.lower()
    
    for model_name, keywords in MODEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_models.append(model_name)
                break
    
    return list(set(found_models))


def count_figures_tables(text):
    """统计图表数量"""
    # 匹配Figure/图
    fig_patterns = [
        r'Figure\s*\d+', r'Fig\.\s*\d+', r'图\s*\d+',
        r'FIGURE\s*\d+', r'FIG\.\s*\d+'
    ]
    
    # 匹配Table/表
    table_patterns = [
        r'Table\s*\d+', r'表\s*\d+', r'TABLE\s*\d+'
    ]
    
    figures = set()
    tables = set()
    
    for pattern in fig_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        figures.update([m.lower() for m in matches])
    
    for pattern in table_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tables.update([m.lower() for m in matches])
    
    return len(figures), len(tables)


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


def get_year_from_filename(filename):
    """从文件名提取年份"""
    # 匹配论文编号前两位
    match = re.search(r'(\d{2})\d{5}', filename)
    if match:
        year_prefix = match.group(1)
        if year_prefix == "20":
            return "2020"
        elif year_prefix == "21":
            return "2021"
        elif year_prefix == "22":
            return "2022"
        elif year_prefix == "23":
            return "2023"
        elif year_prefix == "24":
            return "2024"
        elif year_prefix == "25":
            return "2025"
    return "未知"


def get_paper_id(filename):
    """从文件名提取论文编号"""
    match = re.search(r'(\d{7})', filename)
    if match:
        return match.group(1)
    return filename.replace(".pdf", "").replace("_翻译版", "").replace("翻译版", "")


def main():
    print("=" * 70)
    print("MCM/ICM A题论文分析系统")
    print("=" * 70)
    
    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(PAPER_DIR) if f.endswith('.pdf')]
    pdf_files.sort()
    
    print(f"\n找到 {len(pdf_files)} 篇论文待分析...\n")
    
    # 存储分析结果
    papers = []
    all_models = Counter()
    all_visualizations = Counter()
    year_counts = Counter()
    total_figures = 0
    total_tables = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(PAPER_DIR, pdf_file)
        paper_id = get_paper_id(pdf_file)
        year = get_year_from_filename(pdf_file)
        
        print(f"[{i}/{len(pdf_files)}] 分析: {paper_id} ({year}年)")
        
        # 提取文本
        text = extract_pdf_text(pdf_path)
        
        if not text:
            print(f"  跳过: 无法提取文本")
            continue
        
        # 分析
        models = identify_models(text)
        figures, tables = count_figures_tables(text)
        visualizations = identify_visualizations(text)
        
        # 统计
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
            "visualizations": visualizations
        })
        
        print(f"  模型: {len(models)}个 | 图: {figures}张 | 表: {tables}个")
    
    # 生成输出
    print("\n" + "=" * 70)
    print("生成知识库文件...")
    print("=" * 70)
    
    # 1. 生成分析统计报告
    generate_statistics_report(papers, all_models, all_visualizations, 
                               year_counts, total_figures, total_tables)
    
    # 2. 生成可视化知识库CSV
    generate_visualization_csv(all_visualizations, papers)
    
    # 3. 生成模型库CSV
    generate_model_csv(all_models, papers)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)


def generate_statistics_report(papers, all_models, all_visualizations, 
                               year_counts, total_figures, total_tables):
    """生成分析统计报告"""
    output_path = os.path.join(OUTPUT_DIR, "A题分析统计报告.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MCM/ICM A题论文分析统计报告\n")
        f.write("=" * 70 + "\n\n")
        
        # 年份分布
        f.write("【年份分布】\n")
        for year in sorted(year_counts.keys()):
            f.write(f"  {year}年: {year_counts[year]}篇\n")
        f.write(f"\n  总计: {len(papers)}篇\n\n")
        
        # 高频模型
        f.write("【高频模型】（A题特色：连续建模/微分方程/优化）\n")
        for i, (model, count) in enumerate(all_models.most_common(30), 1):
            f.write(f"  {i:2d}. {model:24s} - {count}次\n")
        f.write("\n")
        
        # 高频可视化
        f.write("【高频可视化类型】\n")
        for i, (viz, count) in enumerate(all_visualizations.most_common(25), 1):
            f.write(f"  {i:2d}. {viz:20s} - {count}次\n")
        f.write("\n")
        
        # 图表统计
        f.write("【图表统计】\n")
        f.write(f"  总图数: {total_figures}\n")
        f.write(f"  总表数: {total_tables}\n")
        if papers:
            f.write(f"  平均图数: {total_figures/len(papers):.2f}\n")
            f.write(f"  平均表数: {total_tables/len(papers):.2f}\n")
        f.write("\n")
        
        # A题特色分析
        f.write("【A题特色模型分布】\n")
        categories = {
            "微分方程类": ["常微分方程", "偏微分方程", "Navier-Stokes", "反应扩散方程", 
                        "Lotka-Volterra", "SIR/SEIR模型"],
            "优化算法类": ["线性规划", "整数规划", "非线性规划", "动态规划", "多目标优化",
                        "遗传算法", "粒子群优化", "模拟退火"],
            "数值仿真类": ["蒙特卡洛模拟", "有限元分析", "有限差分法", "元胞自动机", "Agent-Based"],
            "统计分析类": ["线性回归", "主成分分析", "贝叶斯方法", "敏感性分析"],
            "物理模型类": ["流体力学", "传热学", "扩散模型", "弹性力学"],
        }
        
        for cat_name, models in categories.items():
            cat_count = sum(all_models.get(m, 0) for m in models)
            f.write(f"  {cat_name}: {cat_count}次\n")
            for m in models:
                if all_models.get(m, 0) > 0:
                    f.write(f"    └─ {m}: {all_models[m]}次\n")
        f.write("\n")
        
        # 论文列表
        f.write("【论文列表】\n\n")
        for p in sorted(papers, key=lambda x: x['id']):
            f.write(f"{p['year']}-{p['id']}\n")
            if p['models']:
                f.write(f"  模型: {', '.join(p['models'])}\n")
            else:
                f.write(f"  模型: 未识别\n")
            f.write(f"  图表: {p['figures']}图 + {p['tables']}表\n")
            if p['visualizations']:
                f.write(f"  可视化: {', '.join(p['visualizations'][:5])}{'...' if len(p['visualizations']) > 5 else ''}\n")
            f.write("\n")
    
    print(f"✓ 生成: {output_path}")


def generate_visualization_csv(all_visualizations, papers):
    """生成可视化知识库CSV"""
    output_path = os.path.join(OUTPUT_DIR, "A题可视化知识库.csv")
    
    # A题可视化详细定义
    viz_definitions = {
        "相图/相平面图": {
            "作用": "展示动力系统的状态空间轨迹和平衡点",
            "优点": "直观展示系统行为;识别稳定性;理解动力学特性",
            "缺点": "仅适用于2-3维系统;需要微分方程背景",
            "适用场景": "微分方程分析、捕食者-猎物模型、传染病模型、SIR",
            "输入数据": "微分方程解、状态变量时间序列",
        },
        "向量场图": {
            "作用": "展示微分方程定义的向量场方向",
            "优点": "直观展示系统流向;预测轨迹方向",
            "缺点": "箭头密度需调节;计算量大",
            "适用场景": "ODE分析、流场、方向场、微分方程可视化",
            "输入数据": "微分方程定义、网格点",
        },
        "轨迹图": {
            "作用": "展示系统状态随时间的演化路径",
            "优点": "动态过程清晰;便于追踪",
            "缺点": "轨迹过多时混乱",
            "适用场景": "动力系统、粒子运动、传播路径",
            "输入数据": "时间序列、位置/状态数据",
        },
        "等高线图": {
            "作用": "展示二维平面上函数值的等值线",
            "优点": "空间分布清晰;梯度方向明确",
            "缺点": "需要选择合适的等级数",
            "适用场景": "温度场、浓度分布、地形、优化目标函数",
            "输入数据": "二维网格数据、函数值矩阵",
        },
        "热力图/温度场": {
            "作用": "用颜色深浅展示数值大小的空间分布",
            "优点": "直观;便于识别热点区域",
            "缺点": "色彩映射影响解读",
            "适用场景": "传热问题、温度分布、密度分布、相关性矩阵",
            "输入数据": "二维数值矩阵",
        },
        "3D曲面图": {
            "作用": "展示三维空间中的曲面形态",
            "优点": "立体直观;展示复杂关系",
            "缺点": "视角影响理解;交互性需求高",
            "适用场景": "地形建模、响应面、优化景观",
            "输入数据": "X-Y网格、Z高度值",
        },
        "时间演化图": {
            "作用": "展示变量随时间的变化过程",
            "优点": "动态过程清晰;趋势明显",
            "缺点": "长时间序列可能拥挤",
            "适用场景": "时间序列、动态模拟、状态变化",
            "输入数据": "时间点、状态值",
        },
        "Pareto前沿图": {
            "作用": "展示多目标优化的非支配解集",
            "优点": "权衡关系清晰;决策支持",
            "缺点": "高维时难以展示",
            "适用场景": "多目标优化、权衡分析、决策支持",
            "输入数据": "目标函数值、非支配解",
        },
        "网格划分图": {
            "作用": "展示有限元/有限差分的网格结构",
            "优点": "网格质量可视化;验证离散化",
            "缺点": "复杂网格难以展示",
            "适用场景": "有限元分析、CFD、数值模拟",
            "输入数据": "节点坐标、单元连接",
        },
        "收敛性图": {
            "作用": "展示迭代算法的收敛过程",
            "优点": "收敛速度直观;终止条件验证",
            "缺点": "需要足够迭代次数",
            "适用场景": "数值方法、优化算法、迭代求解",
            "输入数据": "迭代次数、目标值/误差",
        },
        "分岔图": {
            "作用": "展示参数变化导致的系统行为分岔",
            "优点": "系统复杂性可视;临界点识别",
            "缺点": "需要参数扫描;计算量大",
            "适用场景": "非线性系统、混沌分析、参数敏感性",
            "输入数据": "参数范围、系统稳态值",
        },
        "流程图": {
            "作用": "展示算法/模型/系统的逻辑流程",
            "优点": "逻辑清晰;易于理解",
            "缺点": "复杂系统难以简化",
            "适用场景": "算法描述、模型框架、技术路线",
            "输入数据": "步骤定义、连接关系",
        },
        "散点图": {
            "作用": "展示两变量间的关系分布",
            "优点": "相关性直观;异常值识别",
            "缺点": "点过多时重叠",
            "适用场景": "相关性分析、数据探索、回归拟合",
            "输入数据": "变量X、变量Y",
        },
        "敏感性分析图": {
            "作用": "展示参数变化对结果的影响程度",
            "优点": "关键参数识别;不确定性评估",
            "缺点": "参数多时复杂",
            "适用场景": "模型验证、参数重要性、鲁棒性分析",
            "输入数据": "参数范围、响应变量",
        },
        "地图": {
            "作用": "在地理空间上展示数据分布",
            "优点": "空间关系直观;地理背景清晰",
            "缺点": "需要地理数据;投影影响",
            "适用场景": "地理分布、传播模拟、资源配置",
            "输入数据": "坐标、属性值",
        },
        "网络图": {
            "作用": "展示节点和连接的网络结构",
            "优点": "关系可视化;结构分析",
            "缺点": "大网络难以展示",
            "适用场景": "网络分析、传播路径、社交网络",
            "输入数据": "节点列表、边列表",
        },
        "稳定性图": {
            "作用": "展示系统平衡点的稳定性区域",
            "优点": "稳定条件明确;设计指导",
            "缺点": "需要理论分析",
            "适用场景": "控制系统、动力学分析、临界条件",
            "输入数据": "参数空间、稳定性判据",
        },
        "折线图": {
            "作用": "展示数据随时间/序列的变化趋势",
            "优点": "趋势清晰;便于对比",
            "缺点": "数据点过多时拥挤",
            "适用场景": "趋势分析、时间序列、模型对比",
            "输入数据": "自变量、因变量序列",
        },
        "柱状图": {
            "作用": "对比不同类别的数值大小",
            "优点": "对比直观;易于理解",
            "缺点": "类别多时拥挤",
            "适用场景": "分类对比、方案比较、统计结果",
            "输入数据": "类别、数值",
        },
        "误差分析图": {
            "作用": "展示数值方法的误差分布和收敛性",
            "优点": "精度评估;方法验证",
            "缺点": "需要真解对比",
            "适用场景": "数值验证、方法对比、精度分析",
            "输入数据": "网格尺寸、误差值",
        },
    }
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["图表类型", "作用/功能", "优点", "缺点", "适用场景/触发关键词", "输入数据要求", "出现次数"])
        
        for viz, count in all_visualizations.most_common():
            if viz in viz_definitions:
                d = viz_definitions[viz]
                writer.writerow([viz, d["作用"], d["优点"], d["缺点"], d["适用场景"], d["输入数据"], count])
            else:
                writer.writerow([viz, "待补充", "待补充", "待补充", viz, "待补充", count])
    
    print(f"✓ 生成: {output_path}")


def generate_model_csv(all_models, papers):
    """生成模型库CSV"""
    output_path = os.path.join(OUTPUT_DIR, "A题模型知识库.csv")
    
    # A题模型详细定义
    model_definitions = {
        "常微分方程": {
            "类别": "微分方程",
            "描述": "描述变量随单一自变量(通常为时间)变化的方程",
            "适用场景": "人口动态、化学反应、机械系统、传染病传播",
            "优点": "数学严谨;解析解可能;物理意义明确",
            "缺点": "复杂系统难以解析求解",
            "关键参数": "初始条件、系统参数",
        },
        "偏微分方程": {
            "类别": "微分方程",
            "描述": "描述变量随多个自变量(时空)变化的方程",
            "适用场景": "传热、扩散、波动、流体",
            "优点": "描述连续介质;空间分布建模",
            "缺点": "求解复杂;需要数值方法",
            "关键参数": "边界条件、初始条件、域几何",
        },
        "Navier-Stokes": {
            "类别": "流体力学",
            "描述": "描述粘性不可压缩流体运动的偏微分方程组",
            "适用场景": "流体流动、气流、水流、交通流",
            "优点": "流体物理准确;广泛验证",
            "缺点": "高Reynolds数难以求解;计算量大",
            "关键参数": "Reynolds数、边界条件、初始速度场",
        },
        "SIR/SEIR模型": {
            "类别": "传染病模型",
            "描述": "仓室模型描述易感-感染-恢复的人群动态",
            "适用场景": "传染病传播、信息扩散、谣言传播",
            "优点": "结构简单;参数可解释;经典验证",
            "缺点": "均匀混合假设;忽略空间结构",
            "关键参数": "传染率β、恢复率γ、基本再生数R0",
        },
        "Lotka-Volterra": {
            "类别": "生态模型",
            "描述": "捕食者-猎物相互作用的微分方程模型",
            "适用场景": "生态系统、竞争动态、资源消耗",
            "优点": "周期行为解释;生态学经典",
            "缺点": "简化假设;稳定性敏感",
            "关键参数": "增长率、捕食率、死亡率",
        },
        "线性规划": {
            "类别": "数学优化",
            "描述": "在线性约束下最大化/最小化线性目标函数",
            "适用场景": "资源分配、生产计划、运输问题",
            "优点": "最优解保证;计算高效;成熟软件",
            "缺点": "仅适用线性关系;离散变量需整数规划",
            "关键参数": "目标系数、约束矩阵、边界",
        },
        "遗传算法": {
            "类别": "元启发式优化",
            "描述": "模拟生物进化的全局优化算法",
            "适用场景": "复杂优化、参数标定、路径规划",
            "优点": "全局搜索;处理非线性;并行",
            "缺点": "不保证最优;参数敏感;计算量大",
            "关键参数": "种群大小、交叉概率、变异概率、代数",
        },
        "蒙特卡洛模拟": {
            "类别": "随机模拟",
            "描述": "通过随机抽样估计数学期望和分布",
            "适用场景": "不确定性量化、风险评估、积分估计",
            "优点": "处理复杂分布;无需解析解;灵活",
            "缺点": "计算量大;收敛慢",
            "关键参数": "抽样次数、分布参数、种子",
        },
        "有限元分析": {
            "类别": "数值方法",
            "描述": "将连续域离散为有限单元求解PDE",
            "适用场景": "结构分析、传热、流体、电磁",
            "优点": "复杂几何;成熟软件;工程标准",
            "缺点": "网格依赖;计算量大;学习曲线陡",
            "关键参数": "网格尺寸、单元类型、边界条件",
        },
        "有限差分法": {
            "类别": "数值方法",
            "描述": "用差分近似导数求解微分方程",
            "适用场景": "规则网格PDE、时间演化问题",
            "优点": "实现简单;理论清晰",
            "缺点": "复杂边界困难;稳定性条件",
            "关键参数": "空间步长、时间步长、差分格式",
        },
        "Agent-Based": {
            "类别": "复杂系统仿真",
            "描述": "模拟个体代理的交互产生涌现行为",
            "适用场景": "社会系统、生态系统、交通、疏散",
            "优点": "个体异质性;涌现行为;直观",
            "缺点": "参数标定困难;计算量大;验证难",
            "关键参数": "代理规则、交互范围、环境设置",
        },
        "元胞自动机": {
            "类别": "离散动力系统",
            "描述": "离散时空上基于局部规则的演化系统",
            "适用场景": "火灾蔓延、城市扩张、交通流",
            "优点": "并行计算;复杂行为从简单规则;直观",
            "缺点": "规则设计主观;参数标定",
            "关键参数": "邻域定义、状态转移规则、边界条件",
        },
        "马尔可夫链": {
            "类别": "随机过程",
            "描述": "状态转移只依赖当前状态的随机过程",
            "适用场景": "状态预测、排队系统、随机游走",
            "优点": "数学可处理;稳态分析;概率解释",
            "缺点": "无记忆假设;状态离散化",
            "关键参数": "状态空间、转移概率矩阵",
        },
        "敏感性分析": {
            "类别": "模型验证",
            "描述": "研究输入参数变化对输出的影响",
            "适用场景": "参数重要性、不确定性传播、模型验证",
            "优点": "识别关键参数;增强可信度",
            "缺点": "计算量随参数数增加;局部vs全局",
            "关键参数": "参数范围、变化幅度、输出指标",
        },
        "主成分分析": {
            "类别": "降维方法",
            "描述": "通过正交变换将相关变量转为不相关主成分",
            "适用场景": "降维、特征提取、数据可视化",
            "优点": "方差最大化;去相关;降噪",
            "缺点": "线性假设;解释性降低",
            "关键参数": "保留主成分数、方差贡献率阈值",
        },
        "贝叶斯方法": {
            "类别": "统计推断",
            "描述": "基于贝叶斯定理更新概率分布的方法",
            "适用场景": "参数估计、预测、不确定性量化",
            "优点": "先验知识融合;不确定性量化;小样本",
            "缺点": "先验选择主观;计算复杂",
            "关键参数": "先验分布、似然函数、后验采样方法",
        },
    }
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["模型名称", "类别", "描述", "适用场景", "优点", "缺点", "关键参数", "出现次数"])
        
        for model, count in all_models.most_common():
            if model in model_definitions:
                d = model_definitions[model]
                writer.writerow([model, d["类别"], d["描述"], d["适用场景"], 
                               d["优点"], d["缺点"], d["关键参数"], count])
            else:
                writer.writerow([model, "待分类", "待补充", "待补充", 
                               "待补充", "待补充", "待补充", count])
    
    print(f"✓ 生成: {output_path}")


if __name__ == "__main__":
    main()
