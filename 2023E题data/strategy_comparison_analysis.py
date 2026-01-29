import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple


class StrategyComparisonAnalysis:
    """多策略对比分析"""
    
    def __init__(self):
        self.category_scores_file = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\wsm_results_us\category_scores_us.csv'
        self.regions_of_interest = ['美国大峡谷国家公园', '美国纽约曼哈顿']
        self.years = 5
        
        self.strategies = {
            'strategy_1': '策略一：加强光污染治理',
            'strategy_2': '策略二：优化能源系统',
            'strategy_3': '策略三：综合协调发展'
        }
        
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
    
    def apply_strategy(self, current_score: float, intervention_params: Dict, years: int) -> List[float]:
        """
        应用特定策略
        
        参数：
        - current_score: 当前得分
        - intervention_params: 干预参数
        - years: 演化年数
        
        返回：每年的得分列表
        """
        scores = [current_score]
        intervention_rate = intervention_params['intervention_rate']
        growth_rate = intervention_params['growth_rate']
        
        for year in range(1, years + 1):
            previous_score = scores[-1]
            
            new_score = previous_score * (1 + growth_rate) + intervention_rate * (100 - previous_score)
            new_score = min(100, max(0, new_score))
            
            scores.append(new_score)
        
        return scores
    
    def calculate_coupling_degree(self, systems: Dict[str, float]) -> float:
        """
        计算多系统耦合协调度
        
        参数：
        - systems: 各系统的平均得分
        """
        n = len(systems)
        
        numerator = 1.0
        denominator = 0.0
        
        for key, value in systems.items():
            numerator *= value
            denominator += value
        
        if denominator == 0:
            return 0
        
        c = n * (numerator ** (1 / n)) / denominator
        
        t = sum(value for value in systems.values()) / (n * 100)
        
        d = np.sqrt(c * t)
        return d
    
    def classify_coordination_level(self, d_value: float) -> Tuple[str, str]:
        """
        划分协调等级
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
    
    def define_strategy_params(self) -> Dict:
        """
        定义三个独立策略的参数
        
        策略一：加强光污染治理
        - 重点提升人类健康和生态影响
        - 适度提升社会影响
        - 较低提升能源因素
        
        策略二：优化能源系统
        - 重点提升能源因素
        - 适度提升社会影响
        - 较低提升人类健康和生态影响
        
        策略三：综合协调发展
        - 均衡提升四个一级指标
        - 追求系统间的协调发展
        """
        return {
            '美国大峡谷国家公园': {
                'strategy_1': {
                    'human_health': {'intervention_rate': 0.05, 'growth_rate': 0.02},
                    'social_impact': {'intervention_rate': 0.03, 'growth_rate': 0.015},
                    'ecological_impact': {'intervention_rate': 0.05, 'growth_rate': 0.02},
                    'energy_factor': {'intervention_rate': 0.02, 'growth_rate': 0.01}
                },
                'strategy_2': {
                    'human_health': {'intervention_rate': 0.02, 'growth_rate': 0.01},
                    'social_impact': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'ecological_impact': {'intervention_rate': 0.02, 'growth_rate': 0.01},
                    'energy_factor': {'intervention_rate': 0.08, 'growth_rate': 0.04}
                },
                'strategy_3': {
                    'human_health': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'social_impact': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'ecological_impact': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'energy_factor': {'intervention_rate': 0.04, 'growth_rate': 0.02}
                }
            },
            '美国纽约曼哈顿': {
                'strategy_1': {
                    'human_health': {'intervention_rate': 0.12, 'growth_rate': 0.05},
                    'social_impact': {'intervention_rate': 0.06, 'growth_rate': 0.03},
                    'ecological_impact': {'intervention_rate': 0.12, 'growth_rate': 0.05},
                    'energy_factor': {'intervention_rate': 0.04, 'growth_rate': 0.02}
                },
                'strategy_2': {
                    'human_health': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'social_impact': {'intervention_rate': 0.08, 'growth_rate': 0.04},
                    'ecological_impact': {'intervention_rate': 0.04, 'growth_rate': 0.02},
                    'energy_factor': {'intervention_rate': 0.15, 'growth_rate': 0.06}
                },
                'strategy_3': {
                    'human_health': {'intervention_rate': 0.08, 'growth_rate': 0.04},
                    'social_impact': {'intervention_rate': 0.08, 'growth_rate': 0.04},
                    'ecological_impact': {'intervention_rate': 0.08, 'growth_rate': 0.04},
                    'energy_factor': {'intervention_rate': 0.08, 'growth_rate': 0.04}
                }
            }
        }
    
    def run_strategy_comparison(self):
        """运行策略对比分析"""
        print("=" * 100)
        print("多策略对比分析")
        print("=" * 100)
        
        print("\n步骤1: 加载一级指标得分数据")
        print("=" * 100)
        
        df = self.load_category_scores()
        df_100 = self.convert_to_100_scale(df)
        
        print("\n步骤2: 定义策略参数")
        print("=" * 100)
        
        strategy_params = self.define_strategy_params()
        
        print("\n策略说明：")
        for key, name in self.strategies.items():
            print(f"  {key}: {name}")
        
        print("\n步骤3: 应用三个独立策略")
        print("=" * 100)
        
        results = {}
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            
            region_data = df_100[df_100['Region_Name'] == region].iloc[0]
            current_scores = {
                'human_health': region_data['human_health'],
                'social_impact': region_data['social_impact'],
                'ecological_impact': region_data['ecological_impact'],
                'energy_factor': region_data['energy_factor']
            }
            
            print("当前得分（原始数值）:")
            for indicator, score in current_scores.items():
                print(f"  {indicator}: {score:.2f}")
            
            results[region] = {
                'original': current_scores.copy(),
                'strategies': {}
            }
            
            for strategy_key, strategy_name in self.strategies.items():
                print(f"\n{strategy_name}:")
                
                strategy_result = {}
                params = strategy_params[region][strategy_key]
                
                for indicator in ['human_health', 'social_impact', 'ecological_impact', 'energy_factor']:
                    current_score = current_scores[indicator]
                    scores_over_time = self.apply_strategy(
                        current_score=current_score,
                        intervention_params=params[indicator],
                        years=self.years
                    )
                    strategy_result[indicator] = scores_over_time[-1]
                    print(f"  {indicator}: {scores_over_time[-1]:.2f} (从{current_score:.2f}变化而来)")
                
                systems = {
                    'human_health': strategy_result['human_health'],
                    'social_impact': strategy_result['social_impact'],
                    'ecological_impact': strategy_result['ecological_impact'],
                    'energy_factor': strategy_result['energy_factor']
                }
                
                d_value = self.calculate_coupling_degree(systems)
                level, description = self.classify_coordination_level(d_value)
                
                strategy_result['coupling_degree'] = d_value
                strategy_result['coordination_level'] = level
                
                print(f"  耦合协调度 D = {d_value:.4f}")
                print(f"  协调等级: {level}")
                
                results[region]['strategies'][strategy_key] = strategy_result
        
        print("\n步骤4: 计算原始状态的耦合协调度")
        print("=" * 100)
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            
            original_scores = results[region]['original']
            
            d_value = self.calculate_coupling_degree(original_scores)
            level, description = self.classify_coordination_level(d_value)
            
            results[region]['original']['coupling_degree'] = d_value
            results[region]['original']['coordination_level'] = level
            
            print(f"耦合协调度 D = {d_value:.4f}")
            print(f"协调等级: {level}")
        
        print("\n步骤5: 生成对比表格")
        print("=" * 100)
        
        self.generate_comparison_tables(results)
        
        print("\n步骤6: 保存结果")
        print("=" * 100)
        
        self.save_results(results)
        
        print("\n分析完成！")
        
        return results
    
    def generate_comparison_tables(self, results: Dict):
        """生成策略对比表格"""
        
        for region in self.regions_of_interest:
            print(f"\n{'=' * 100}")
            print(f"📍 {region} - 策略对比表格")
            print(f"{'=' * 100}")
            
            original = results[region]['original']
            strategy_1 = results[region]['strategies']['strategy_1']
            strategy_2 = results[region]['strategies']['strategy_2']
            strategy_3 = results[region]['strategies']['strategy_3']
            
            table_data = {
                '指标': [
                    '人类健康影响',
                    '社会影响',
                    '生态影响',
                    '能源因素',
                    '耦合协调度',
                    '协调等级'
                ],
                '原始数值': [
                    f"{original['human_health']:.2f}",
                    f"{original['social_impact']:.2f}",
                    f"{original['ecological_impact']:.2f}",
                    f"{original['energy_factor']:.2f}",
                    f"{original['coupling_degree']:.4f}",
                    original['coordination_level']
                ],
                '策略一实施后': [
                    f"{strategy_1['human_health']:.2f}",
                    f"{strategy_1['social_impact']:.2f}",
                    f"{strategy_1['ecological_impact']:.2f}",
                    f"{strategy_1['energy_factor']:.2f}",
                    f"{strategy_1['coupling_degree']:.4f}",
                    strategy_1['coordination_level']
                ],
                '策略二实施后': [
                    f"{strategy_2['human_health']:.2f}",
                    f"{strategy_2['social_impact']:.2f}",
                    f"{strategy_2['ecological_impact']:.2f}",
                    f"{strategy_2['energy_factor']:.2f}",
                    f"{strategy_2['coupling_degree']:.4f}",
                    strategy_2['coordination_level']
                ],
                '策略三实施后': [
                    f"{strategy_3['human_health']:.2f}",
                    f"{strategy_3['social_impact']:.2f}",
                    f"{strategy_3['ecological_impact']:.2f}",
                    f"{strategy_3['energy_factor']:.2f}",
                    f"{strategy_3['coupling_degree']:.4f}",
                    strategy_3['coordination_level']
                ]
            }
            
            df_table = pd.DataFrame(table_data)
            print(df_table.to_string(index=False))
            
            results[region]['comparison_table'] = df_table
    
    def save_results(self, results: Dict):
        """保存分析结果"""
        output_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\strategy_comparison_results'
        os.makedirs(output_dir, exist_ok=True)
        
        for region in self.regions_of_interest:
            table = results[region]['comparison_table']
            filename = f"{region}_策略对比.csv"
            filepath = os.path.join(output_dir, filename)
            table.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✅ {region}策略对比表格已保存: {filepath}")
        
        print(f"\n所有结果已保存到目录: {output_dir}")


if __name__ == "__main__":
    analyzer = StrategyComparisonAnalysis()
    results = analyzer.run_strategy_comparison()
