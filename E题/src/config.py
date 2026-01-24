"""
Configuration file for MCM E-Problem Evaluation System
E题专用配置文件：综合评价、决策支持、风险评估

Core Models: AHP, EWM, TOPSIS, CRITIC, DEA, Grey Relational Analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


@dataclass
class DataConfig:
    """Data file paths and loading configuration"""
    # E题常见数据文件
    indicators_file: str = 'indicators.csv'
    evaluation_matrix_file: str = 'evaluation_matrix.csv'
    decision_units_file: str = 'decision_units.csv'
    
    # Data validation
    required_columns: List[str] = field(default_factory=lambda: [
        'id', 'name', 'category'
    ])
    
    # Missing value handling
    imputation_strategy: str = 'mean'  # 'mean', 'median', 'mode'
    
    # Normalization method for evaluation matrix
    normalization_method: str = 'minmax'  # 'minmax', 'zscore', 'vector'


@dataclass
class AHPConfig:
    """AHP (Analytic Hierarchy Process) 层次分析法配置"""
    # 一致性检验阈值
    consistency_threshold: float = 0.1  # CR < 0.1 通过一致性检验
    
    # 1-9标度法
    scale: int = 9  # Saaty 1-9 scale
    
    # 随机一致性指数 RI (1-15阶)
    random_consistency_index: Dict[int, float] = field(default_factory=lambda: {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    })
    
    # 计算方法
    calculation_method: str = 'eigenvalue'  # 'eigenvalue', 'geometric_mean', 'arithmetic_mean'
    
    # 最大迭代次数（特征值法）
    max_iterations: int = 100
    tolerance: float = 1e-8


@dataclass
class EWMConfig:
    """Entropy Weight Method 熵权法配置"""
    # 数据标准化方法
    normalization_method: str = 'minmax'  # 'minmax', 'zscore'
    
    # 处理零值的小常数
    epsilon: float = 1e-10
    
    # 是否对效益型和成本型指标分别处理
    handle_indicator_types: bool = True
    
    # 指标类型定义 (True=效益型/越大越好, False=成本型/越小越好)
    # 需在运行时设置
    indicator_types: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CRITICConfig:
    """CRITIC (Criteria Importance Through Intercriteria Correlation) 配置"""
    # 标准化方法
    normalization_method: str = 'minmax'
    
    # 相关性计算方法
    correlation_method: str = 'pearson'  # 'pearson', 'spearman', 'kendall'
    
    # 冲突性权重
    conflict_weight: float = 1.0


@dataclass
class TOPSISConfig:
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) 配置"""
    # 标准化方法
    normalization_method: str = 'vector'  # 'vector', 'minmax', 'max'
    
    # 距离计算方法
    distance_method: str = 'euclidean'  # 'euclidean', 'manhattan', 'chebyshev'
    
    # 是否使用加权距离
    use_weighted_distance: bool = True
    
    # 理想解类型
    ideal_solution_type: str = 'max'  # 'max' for positive ideal, 'min' for negative ideal


@dataclass
class GREYConfig:
    """Grey Relational Analysis 灰色关联分析配置"""
    # 分辨系数 ρ (通常取0.5)
    resolution_coefficient: float = 0.5
    
    # 标准化方法
    normalization_method: str = 'minmax'  # 'minmax', 'initial', 'average'
    
    # 参考序列选择方法
    reference_method: str = 'optimal'  # 'optimal', 'average', 'custom'


@dataclass
class DEAConfig:
    """DEA (Data Envelopment Analysis) 数据包络分析配置"""
    # 模型类型
    model_type: str = 'CCR'  # 'CCR', 'BCC', 'SBM'
    
    # 规模报酬假设
    returns_to_scale: str = 'CRS'  # 'CRS' (constant), 'VRS' (variable)
    
    # 导向
    orientation: str = 'input'  # 'input', 'output'
    
    # 是否计算超效率
    super_efficiency: bool = False
    
    # 线性规划求解器
    solver: str = 'scipy'  # 'scipy', 'pulp', 'cvxpy'


@dataclass
class FuzzyConfig:
    """Fuzzy Comprehensive Evaluation 模糊综合评价配置"""
    # 评价等级数
    evaluation_levels: int = 5  # 优、良、中、较差、差
    
    # 隶属函数类型
    membership_function: str = 'trapezoidal'  # 'triangular', 'trapezoidal', 'gaussian'
    
    # 模糊合成算子
    composition_operator: str = 'weighted_average'  # 'min_max', 'weighted_average', 'bounded_sum'
    
    # 等级阈值 (默认5等级)
    level_thresholds: List[float] = field(default_factory=lambda: [0.9, 0.7, 0.5, 0.3, 0.0])
    
    # 等级标签
    level_labels: List[str] = field(default_factory=lambda: ['优秀', '良好', '中等', '较差', '差'])


@dataclass
class GEMatrixConfig:
    """GE Matrix (General Electric Matrix) 配置"""
    # 矩阵维度 (通常3x3)
    matrix_size: int = 3
    
    # 业务强度阈值
    business_strength_thresholds: List[float] = field(default_factory=lambda: [0.33, 0.67, 1.0])
    
    # 行业吸引力阈值
    industry_attractiveness_thresholds: List[float] = field(default_factory=lambda: [0.33, 0.67, 1.0])
    
    # 九宫格策略建议
    strategy_recommendations: Dict[Tuple[int, int], str] = field(default_factory=lambda: {
        (0, 0): 'Invest/Grow', (0, 1): 'Invest/Grow', (0, 2): 'Selectivity',
        (1, 0): 'Invest/Grow', (1, 1): 'Selectivity', (1, 2): 'Harvest/Divest',
        (2, 0): 'Selectivity', (2, 1): 'Harvest/Divest', (2, 2): 'Harvest/Divest'
    })


@dataclass
class RiskAssessmentConfig:
    """Risk Assessment 风险评估配置"""
    # 风险等级数
    risk_levels: int = 5  # I, II, III, IV, V
    
    # 风险等级阈值
    risk_thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 风险等级标签
    risk_labels: List[str] = field(default_factory=lambda: ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # 信息扩散参数
    diffusion_coefficient: float = 0.1
    
    # 破产理论参数
    initial_surplus: float = 1.0
    safety_loading: float = 0.1


@dataclass
class SensitivityConfig:
    """Sensitivity Analysis 敏感性分析配置"""
    # 参数变化范围
    weight_variation: float = 0.20  # ±20%
    threshold_variation: float = 0.20  # ±20%
    
    # 敏感性判定阈值
    high_sensitivity_threshold: float = 0.10  # 结果变化>10%视为高敏感
    
    # 蒙特卡洛模拟次数
    monte_carlo_iterations: int = 1000
    
    # 是否进行权重敏感性分析
    analyze_weight_sensitivity: bool = True
    
    # 是否进行阈值敏感性分析
    analyze_threshold_sensitivity: bool = True


@dataclass
class UncertaintyConfig:
    """Uncertainty Quantification 不确定性量化配置"""
    # Bootstrap参数
    n_bootstrap_samples: int = 1000
    bootstrap_confidence_level: float = 0.95
    bootstrap_method: str = 'case_resampling'  # 'case_resampling', 'weight_perturbation'
    
    # Monte Carlo参数
    n_monte_carlo_simulations: int = 1000
    monte_carlo_noise_level: float = 0.10  # 10% noise
    
    # 随机种子
    random_seed: int = 42


@dataclass
class InterventionConfig:
    """Intervention Optimization 干预策略优化配置"""
    # 默认预算
    default_budget: float = 500_000.0
    
    # 优化目标
    optimization_objective: str = 'minimize_pollution'  # 'minimize_pollution', 'maximize_cost_effectiveness'
    
    # 干预选项 (将在运行时从数据加载)
    available_interventions: List[str] = field(default_factory=lambda: [
        'Full Cutoff Shielding',
        'LED Conversion (Warm CCT)',
        'Adaptive Dimming Controls',
        'Midnight Curfew Policy',
        'Vegetation Screening',
    ])


@dataclass
class OutputConfig:
    """Output and visualization configuration"""
    # 图表配置
    figure_dpi: int = 300
    figure_format: str = 'png'
    figure_size: Tuple[int, int] = (10, 6)
    
    # 字体配置
    font_family: str = 'Times New Roman'
    font_size: int = 12
    
    # 颜色配置
    color_scheme: str = 'academic'  # 'academic', 'colorblind_friendly'
    
    # CSV配置
    csv_encoding: str = 'utf-8'
    
    # 小数精度
    decimal_places: int = 4
    
    # 最小图表数量要求
    min_charts: int = 10


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = field(default_factory=DataConfig)
    ahp: AHPConfig = field(default_factory=AHPConfig)
    ewm: EWMConfig = field(default_factory=EWMConfig)
    critic: CRITICConfig = field(default_factory=CRITICConfig)
    topsis: TOPSISConfig = field(default_factory=TOPSISConfig)
    grey: GREYConfig = field(default_factory=GREYConfig)
    dea: DEAConfig = field(default_factory=DEAConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    ge_matrix: GEMatrixConfig = field(default_factory=GEMatrixConfig)
    risk: RiskAssessmentConfig = field(default_factory=RiskAssessmentConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Combined weighting parameter (α for AHP, 1-α for EWM)
    alpha_combined_weight: float = 0.5


# Global configuration instance
config = Config()


# E题常见指标体系模板
INDICATOR_TEMPLATES = {
    'forest_value': {
        'name': 'Forest Value Evaluation',
        'dimensions': ['Carbon Sequestration', 'Biodiversity', 'Cultural Value', 'Economic Value'],
        'indicators_per_dimension': 3
    },
    'risk_assessment': {
        'name': 'Risk Assessment',
        'dimensions': ['Hazard', 'Exposure', 'Vulnerability', 'Adaptability'],
        'indicators_per_dimension': 4
    },
    'sustainability': {
        'name': 'Sustainability Evaluation',
        'dimensions': ['Economic', 'Social', 'Environmental'],
        'indicators_per_dimension': 5
    },
    'insurance_decision': {
        'name': 'Insurance Decision',
        'dimensions': ['Risk Level', 'Premium Adequacy', 'Market Potential', 'Operational Cost'],
        'indicators_per_dimension': 3
    }
}
