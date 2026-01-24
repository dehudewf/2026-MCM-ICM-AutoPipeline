"""
E题核心模型库

包含决策/评估类问题的三大核心模型：
- AHP (层次分析法) - 主观权重
- EWM (熵权法) - 客观权重  
- TOPSIS (逼近理想解排序法) - 综合评价

Author: MCM Team 2026
"""

from .ahp_model import AHPModel, AHPResult
from .ewm_model import EWMModel, EWMResult, ewm_weights, combine_ahp_ewm
from .topsis_model import TOPSISModel, TOPSISResult

__all__ = [
    # AHP
    'AHPModel',
    'AHPResult',
    
    # EWM
    'EWMModel', 
    'EWMResult',
    'ewm_weights',
    'combine_ahp_ewm',
    
    # TOPSIS
    'TOPSISModel',
    'TOPSISResult',
]
