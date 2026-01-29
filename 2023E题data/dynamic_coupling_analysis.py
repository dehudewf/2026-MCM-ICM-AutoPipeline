import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


class DynamicCouplingAnalysis:
    """动态演化与耦合协调度分析"""
    
    def __init__(self):
        self.category_scores_file = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\wsm_results_us\category_scores_us.csv'
        self.regions_of_interest = ['美国大峡谷国家公园', '美国纽约曼哈顿']
        self.years = 5
        
    def load_category_scores(self) -> pd.DataFrame:
        """加载一级指标得分数据"""
        df = pd.read_csv(self.category_scores_file)
        return df
    
    def convert_to_100_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """将一级指标得分转换为100分制原始分"""
        df_100 = df.copy()
        
        for col in ['human_health', 'social_impact', 'ecological_impact', 'energy_factor']:
            df_100[col] = df_100[col] * 100
            
        return df_100
    
    def apply_dynamic_evolution_model(self, current_score: float, intervention_rate: float, 
                                       growth_rate: float, years: int) -> List[float]:
        """
        应用动态演化模型（模拟策略干预）
        
        参数：
        - current_score: 当前得分
        - intervention_rate: 干预强度（策略实施力度）
        - growth_rate: 自然增长率
        - years: 演化年数
        
        返回：每年的得分列表
        """
        scores = [current_score]
        
        for year in range(1, years + 1):
            previous_score = scores[-1]
            
            new_score = previous_score * (1 + growth_rate) + intervention_rate * (100 - previous_score)
            new_score = min(100, max(0, new_score))
            
            scores.append(new_score)
        
        return scores
    
    def simulate_intervention(self, df_100: pd.DataFrame, intervention_params: Dict) -> Dict:
        """
        模拟策略干预
        
        参数：
        - df_100: 100分制的数据
        - intervention_params: 干预参数字典
        """
        results = {}
        
        for region in self.regions_of_interest:
            region_data = df_100[df_100['Region_Name'] == region].iloc[0]
            
            results[region] = {
                'current': {},
                'projected': {}
            }
            
            for indicator in ['human_health', 'social_impact', 'ecological_impact', 'energy_factor']:
                current_score = region_data[indicator]
                params = intervention_params[region][indicator]
                
                results[region]['current'][indicator] = current_score
                
                scores_over_time = self.apply_dynamic_evolution_model(
                    current_score=current_score,
                    intervention_rate=params['intervention_rate'],
                    growth_rate=params['growth_rate'],
                    years=self.years
                )
                
                results[region]['projected'][indicator] = scores_over_time
        
        return results
    
    def calculate_coupling_degree(self, system1: List[float], system2: List[float]) -> float:
        """
        计算两个系统的耦合度
        
        公式：C = 2 * sqrt((U1 * U2) / ((U1 + U2) * (U1 + U2)))
        """
        u1 = np.mean(system1)
        u2 = np.mean(system2)
        
        if u1 + u2 == 0:
            return 0
        
        c = 2 * np.sqrt((u1 * u2) / ((u1 + u2) ** 2))
        return c
    
    def calculate_coupling_coordination_degree(self, system1: List[float], system2: List[float], 
                                                 alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        计算耦合协调度
        
        公式：D = sqrt(C * T)
        其中 T = alpha * U1 + beta * U2
        """
        c = self.calculate_coupling_degree(system1, system2)
        
        u1 = np.mean(system1)
        u2 = np.mean(system2)
        
        t = alpha * u1 + beta * u2
        
        d = np.sqrt(c * t)
        return d
    
    def calculate_multi_system_coupling(self, systems: Dict[str, List[float]], 
                                        weights: Dict[str, float] = None) -> float:
        """
        计算多系统耦合协调度
        
        参数：
        - systems: 各系统的得分列表
        - weights: 各系统的权重
        """
        if weights is None:
            weights = {k: 1.0 / len(systems) for k in systems.keys()}
        
        n = len(systems)
        
        numerator = 1.0
        denominator = 0.0
        
        for key, scores in systems.items():
            u = np.mean(scores)
            numerator *= u
            denominator += u
        
        if denominator == 0:
            return 0
        
        c = n * (numerator ** (1 / n)) / denominator
        
        t = sum(weights[key] * np.mean(scores) for key, scores in systems.items()) / 100
        
        d = np.sqrt(c * t)
        return d
    
    def classify_coordination_level(self, d_value: float) -> Tuple[str, str]:
        """
        划分协调等级
        
        根据耦合协调度D值划分等级：
        - 0.9 < D ≤ 1.0: 优质协调
        - 0.8 < D ≤ 0.9: 良好协调
        - 0.7 < D ≤ 0.8: 中级协调
        - 0.6 < D ≤ 0.7: 初级协调
        - 0.5 < D ≤ 0.6: 勉强协调
        - 0.4 < D ≤ 0.5: 濒临失调
        - 0.0 ≤ D ≤ 0.4: 失调
        """
        if d_value > 0.9:
            level = '优质协调'
            description = '系统间协调性极佳，发展高度均衡'
        elif d_value > 0.8:
            level = '良好协调'
            description = '系统间协调性良好，发展较为均衡'
        elif d_value > 0.7:
            level = '中级协调'
            description = '系统间协调性中等，发展基本均衡'
        elif d_value > 0.6:
            level = '初级协调'
            description = '系统间协调性一般，发展存在一定差异'
        elif d_value > 0.5:
            level = '勉强协调'
            description = '系统间协调性较差，发展差异较大'
        elif d_value > 0.4:
            level = '濒临失调'
            description = '系统间协调性很差，发展严重失衡'
        else:
            level = '失调'
            description = '系统间完全失调，发展极度失衡'
        
        return level, description
    
    def run_analysis(self, intervention_params: Dict):
        """运行完整的分析流程"""
        print("=" * 100)
        print("步骤1: 加载一级指标得分数据")
        print("=" * 100)
        
        df = self.load_category_scores()
        print("\n原始一级指标得分：")
        print(df)
        
        print("\n" + "=" * 100)
        print("步骤2: 转换为100分制原始分")
        print("=" * 100)
        
        df_100 = self.convert_to_100_scale(df)
        print("\n100分制原始分：")
        print(df_100)
        
        print("\n" + "=" * 100)
        print("步骤3: 应用动态演化模型（模拟策略干预）")
        print("=" * 100)
        
        intervention_results = self.simulate_intervention(df_100, intervention_params)
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            print("当前得分:")
            for indicator, score in intervention_results[region]['current'].items():
                print(f"  {indicator}: {score:.2f}")
            
            print(f"\n{self.years}年后预测得分:")
            for indicator, scores in intervention_results[region]['projected'].items():
                print(f"  {indicator}: {scores[-1]:.2f} (从{scores[0]:.2f}变化而来)")
        
        print("\n" + "=" * 100)
        print("步骤4: 计算耦合协调度")
        print("=" * 100)
        
        coupling_results = {}
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            
            projected = intervention_results[region]['projected']
            
            systems = {
                'human_health': projected['human_health'],
                'social_impact': projected['social_impact'],
                'ecological_impact': projected['ecological_impact'],
                'energy_factor': projected['energy_factor']
            }
            
            d_value = self.calculate_multi_system_coupling(systems)
            level, description = self.classify_coordination_level(d_value)
            
            coupling_results[region] = {
                'coupling_degree': d_value,
                'coordination_level': level,
                'description': description,
                'systems': systems
            }
            
            print(f"耦合协调度 D = {d_value:.4f}")
            print(f"协调等级: {level}")
            print(f"描述: {description}")
        
        print("\n" + "=" * 100)
        print("步骤5: 保存结果")
        print("=" * 100)
        
        self.save_results(intervention_results, coupling_results, df_100)
        
        print("\n分析完成！")
        
        return intervention_results, coupling_results
    
    def save_results(self, intervention_results: Dict, coupling_results: Dict, df_100: pd.DataFrame):
        """保存分析结果"""
        import os
        
        output_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\dynamic_coupling_results'
        os.makedirs(output_dir, exist_ok=True)
        
        df_100.to_csv(os.path.join(output_dir, 'category_scores_100_scale.csv'), index=False, encoding='utf-8-sig')
        print(f"\n✅ 100分制原始分已保存: {os.path.join(output_dir, 'category_scores_100_scale.csv')}")
        
        summary_data = []
        for region in self.regions_of_interest:
            current = intervention_results[region]['current']
            projected = {k: v[-1] for k, v in intervention_results[region]['projected'].items()}
            
            summary_data.append({
                'Region': region,
                'Current_Human_Health': current['human_health'],
                'Current_Social_Impact': current['social_impact'],
                'Current_Ecological_Impact': current['ecological_impact'],
                'Current_Energy_Factor': current['energy_factor'],
                'Projected_Human_Health': projected['human_health'],
                'Projected_Social_Impact': projected['social_impact'],
                'Projected_Ecological_Impact': projected['ecological_impact'],
                'Projected_Energy_Factor': projected['energy_factor'],
                'Coupling_Degree': coupling_results[region]['coupling_degree'],
                'Coordination_Level': coupling_results[region]['coordination_level'],
                'Description': coupling_results[region]['description']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'dynamic_coupling_summary.csv'), index=False, encoding='utf-8-sig')
        print(f"✅ 动态演化与耦合协调度分析结果已保存: {os.path.join(output_dir, 'dynamic_coupling_summary.csv')}")
        
        print(f"\n所有结果已保存到目录: {output_dir}")


if __name__ == "__main__":
    analyzer = DynamicCouplingAnalysis()
    
    intervention_params = {
        '美国大峡谷国家公园': {
            'human_health': {'intervention_rate': 0.02, 'growth_rate': 0.01},
            'social_impact': {'intervention_rate': 0.03, 'growth_rate': 0.015},
            'ecological_impact': {'intervention_rate': 0.01, 'growth_rate': 0.005},
            'energy_factor': {'intervention_rate': 0.04, 'growth_rate': 0.02}
        },
        '美国纽约曼哈顿': {
            'human_health': {'intervention_rate': 0.08, 'growth_rate': 0.03},
            'social_impact': {'intervention_rate': 0.05, 'growth_rate': 0.02},
            'ecological_impact': {'intervention_rate': 0.10, 'growth_rate': 0.04},
            'energy_factor': {'intervention_rate': 0.06, 'growth_rate': 0.025}
        }
    }
    
    intervention_results, coupling_results = analyzer.run_analysis(intervention_params)
