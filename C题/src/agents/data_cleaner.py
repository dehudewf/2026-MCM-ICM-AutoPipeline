"""
数据清洗 Agent
功能：缺失值处理、异常值检测、数据类型转换
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class DataCleanerAgent:
    """数据清洗自动化工具"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}
    
    def analyze_quality(self) -> Dict:
        """分析数据质量"""
        report = {
            'shape': self.df.shape,
            'missing': self.df.isnull().sum().to_dict(),
            'missing_pct': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        self.report['quality'] = report
        return report
    
    def detect_outliers(self, columns: List[str] = None, method: str = 'iqr') -> Dict:
        """检测异常值"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (self.df[col] < lower) | (self.df[col] > upper)
            elif method == 'zscore':
                z = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                mask = z > 3
            
            outliers[col] = {
                'count': mask.sum(),
                'pct': mask.sum() / len(self.df) * 100,
                'indices': self.df[mask].index.tolist()[:10]  # 只返回前10个
            }
        
        self.report['outliers'] = outliers
        return outliers
