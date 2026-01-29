import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple


class SensitivityAnalysis:
    """敏感性分析"""
    
    def __init__(self):
        self.weights_file = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\normalized_results\first_level_weights_ewm.csv'
        self.category_scores_file = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\wsm_results_us\category_scores_us.csv'
        self.regions_of_interest = ['美国大峡谷国家公园', '美国纽约曼哈顿']
        self.years = 5
        self.fluctuation_range = 0.10
        
    def load_first_level_weights(self) -> pd.DataFrame:
        """加载一级指标权重"""
        df = pd.read_csv(self.weights_file)
        return df
    
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
    
    def analyze_weight_sensitivity(self) -> Dict:
        """
        对一级指标权重进行敏感性分析
        
        在±10%范围内波动权重，计算耦合协调度的变化
        """
        print("=" * 100)
        print("一级指标权重敏感性分析")
        print("=" * 100)
        
        weights_df = self.load_first_level_weights()
        category_scores_df = self.load_category_scores()
        category_scores_100 = self.convert_to_100_scale(category_scores_df)
        
        print("\n原始一级指标权重：")
        print(weights_df)
        
        base_weights = {
            'human_health': weights_df[weights_df['category'] == 'human_health']['weight'].values[0],
            'social_impact': weights_df[weights_df['category'] == 'social_impact']['weight'].values[0],
            'ecological_impact': weights_df[weights_df['category'] == 'ecological_impact']['weight'].values[0],
            'energy_factor': weights_df[weights_df['category'] == 'energy_factor']['weight'].values[0]
        }
        
        print("\n基础权重：")
        for key, value in base_weights.items():
            print(f"  {key}: {value:.4f}")
        
        fluctuation_levels = np.linspace(-self.fluctuation_range, self.fluctuation_range, 21)
        
        results = {}
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            
            region_data = category_scores_100[category_scores_100['Region_Name'] == region].iloc[0]
            current_scores = {
                'human_health': region_data['human_health'],
                'social_impact': region_data['social_impact'],
                'ecological_impact': region_data['ecological_impact'],
                'energy_factor': region_data['energy_factor']
            }
            
            results[region] = {
                'human_health': [],
                'social_impact': [],
                'ecological_impact': [],
                'energy_factor': []
            }
            
            for indicator in ['human_health', 'social_impact', 'ecological_impact', 'energy_factor']:
                print(f"\n波动 {indicator} 权重：")
                
                for fluctuation in fluctuation_levels:
                    modified_weights = base_weights.copy()
                    
                    for key in modified_weights.keys():
                        if key == indicator:
                            modified_weights[key] = base_weights[key] * (1 + fluctuation)
                        else:
                            modified_weights[key] = base_weights[key] * (1 - fluctuation * base_weights[key] / (1 - base_weights[key]))
                    
                    total_weight = sum(modified_weights.values())
                    for key in modified_weights.keys():
                        modified_weights[key] = modified_weights[key] / total_weight
                    
                    d_value = self.calculate_weighted_degree(current_scores, modified_weights)
                    
                    results[region][indicator].append({
                        'fluctuation': fluctuation,
                        'weight_percentage': fluctuation * 100,
                        'coupling_degree': d_value
                    })
                
                print(f"  完成：{len(results[region][indicator])} 个数据点")
        
        return results
    
    def analyze_eta_sensitivity(self) -> Dict:
        """
        对节能效率参数η进行敏感性分析
        
        在±10%范围内波动η，计算耦合协调度的变化
        """
        print("\n" + "=" * 100)
        print("节能效率参数η敏感性分析")
        print("=" * 100)
        
        category_scores_df = self.load_category_scores()
        category_scores_100 = self.convert_to_100_scale(category_scores_df)
        
        base_eta = 0.5
        print(f"\n基础节能效率参数 η = {base_eta}")
        
        fluctuation_levels = np.linspace(-self.fluctuation_range, self.fluctuation_range, 21)
        
        results = {}
        
        for region in self.regions_of_interest:
            print(f"\n📍 {region}")
            print("-" * 80)
            
            region_data = category_scores_100[category_scores_100['Region_Name'] == region].iloc[0]
            current_scores = {
                'human_health': region_data['human_health'],
                'social_impact': region_data['social_impact'],
                'ecological_impact': region_data['ecological_impact'],
                'energy_factor': region_data['energy_factor']
            }
            
            results[region] = []
            
            for fluctuation in fluctuation_levels:
                modified_eta = base_eta * (1 + fluctuation)
                
                modified_scores = current_scores.copy()
                modified_scores['energy_factor'] = current_scores['energy_factor'] * (1 + fluctuation)
                
                d_value = self.calculate_coupling_degree(modified_scores)
                
                print(f"    波动: {fluctuation*100:.1f}%, 能源因素: {modified_scores['energy_factor']:.2f}, 耦合度: {d_value:.4f}")
                
                results[region].append({
                    'fluctuation': fluctuation,
                    'eta_value': modified_eta,
                    'coupling_degree': d_value
                })
            
            print(f"  完成：{len(results[region])} 个数据点")
        
        return results
    
    def calculate_weighted_degree(self, scores: Dict, weights: Dict) -> float:
        """
        计算加权耦合协调度
        """
        n = len(scores)
        
        numerator = 1.0
        denominator = 0.0
        
        for key, value in scores.items():
            numerator *= value
            denominator += value
        
        if denominator == 0:
            return 0
        
        c = n * (numerator ** (1 / n)) / denominator
        
        t = sum([weights[key] * value for key, value in scores.items()]) / sum(weights.values()) / 100
        
        d = np.sqrt(c * t)
        return d
    
    def calculate_coupling_degree(self, scores: Dict) -> float:
        """
        计算多系统耦合协调度
        """
        n = len(scores)
        
        numerator = 1.0
        denominator = 0.0
        
        for key, value in scores.items():
            numerator *= value
            denominator += value
        
        if denominator == 0:
            return 0
        
        c = n * (numerator ** (1 / n)) / denominator
        
        t = sum([value for value in scores.values()]) / (n * 100)
        
        d = np.sqrt(c * t)
        return d
    
    def generate_weight_sensitivity_charts(self, results: Dict):
        """生成一级指标权重敏感性分析的HTML图表"""
        print("\n" + "=" * 100)
        print("生成一级指标权重敏感性分析图表")
        print("=" * 100)
        
        for region in self.regions_of_interest:
            html_content = self._create_weight_sensitivity_chart(region, results[region])
            
            output_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\sensitivity_analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{region}_一级指标权重敏感性分析.html"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ {region}一级指标权重敏感性分析图表已保存: {filepath}")
    
    def generate_eta_sensitivity_charts(self, results: Dict):
        """生成所有节能效率参数η敏感性分析的HTML图表"""
        print("\n" + "=" * 100)
        print("生成节能效率参数η敏感性分析图表")
        print("=" * 100)
        
        for region in self.regions_of_interest:
            html_content = self._create_eta_sensitivity_chart(region, results[region])
            
            output_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data\sensitivity_analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{region}_节能效率参数敏感性分析.html"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ {region}节能效率参数η敏感性分析图表已保存: {filepath}")
    
    def _create_weight_sensitivity_chart(self, region: str, data: Dict) -> str:
        """创建一级指标权重敏感性分析图表HTML"""
        
        indicators = {
            'human_health': {'name': '人类健康影响', 'color': '#FF6384'},
            'social_impact': {'name': '社会影响', 'color': '#36A2EB'},
            'ecological_impact': {'name': '生态影响', 'color': '#FFCE56'},
            'energy_factor': {'name': '能源因素', 'color': '#4BC0C0'}
        }
        
        fluctuation_labels = [f"{x:.1%}" for x in np.linspace(-self.fluctuation_range, self.fluctuation_range, 21)]
        
        datasets = []
        all_values = []
        for key, info in indicators.items():
            values = [d['coupling_degree'] for d in data[key]]
            all_values.extend(values)
            datasets.append({
                'label': info['name'],
                'data': values,
                'borderColor': info['color'],
                'backgroundColor': info['color'] + '33',
                'fill': False,
                'tension': 0.1
            })
        
        min_value = min(all_values)
        max_value = max(all_values)
        range_value = max_value - min_value
        y_min = max(0, min_value - range_value * 0.2)
        y_max = min(1, max_value + range_value * 0.2)
        
        if y_max - y_min < 0.02:
            y_min = max(0, min_value - 0.01)
            y_max = min(1, max_value + 0.01)
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{region} - 一级指标权重敏感性分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .chart-container {{
            position: relative;
            height: 500px;
            margin: 30px 0;
        }}
        .info {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .info h3 {{
            margin-top: 0;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{region} - 一级指标权重敏感性分析</h1>
        
        <div class="info">
            <h3>分析说明</h3>
            <p>本图表展示了在±10%范围内波动各一级指标权重时，耦合协调度的变化情况。</p>
            <p><strong>横轴：</strong>权重波动百分比（-10% 到 +10%）</p>
            <p><strong>纵轴：</strong>耦合协调度 D 值</p>
        </div>
        
        <div class="chart-container">
            <canvas id="sensitivityChart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('sensitivityChart').getContext('2d');
        const sensitivityChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(fluctuation_labels, ensure_ascii=False)},
                datasets: {json.dumps(datasets, ensure_ascii=False)}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: '一级指标权重波动对耦合协调度的影响',
                        font: {{
                            size: 18,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            font: {{
                                size: 12
                            }}
                        }}
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        callbacks: {{
                            title: function(context) {{
                                return '权重波动: ' + context[0].label;
                            }},
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '权重波动百分比',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            display: true
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '耦合协调度 D 值',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            display: true
                        }},
                        min: {y_min:.4f},
                        max: {y_max:.4f},
                        ticks: {{
                            stepSize: {(y_max - y_min) / 5:.4f}
                        }}
                    }}
                }},
                interaction: {{
                    mode: 'index',
                    intersect: false
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def _create_eta_sensitivity_chart(self, region: str, data: List) -> str:
        """创建节能效率参数η敏感性分析图表HTML"""
        
        fluctuation_labels = [f"{x:.1%}" for x in np.linspace(-self.fluctuation_range, self.fluctuation_range, 21)]
        eta_values = [f"{d['eta_value']:.4f}" for d in data]
        coupling_values = [d['coupling_degree'] for d in data]
        
        all_zero = all(v == 0 for v in coupling_values)
        warning_message = ""
        if all_zero:
            warning_message = """
        <div class="warning" style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin: 20px 0;">
            <h3 style="margin-top: 0; color: #856404;">⚠️ 注意</h3>
            <p style="color: #856404;">该地区在某些一级指标上的得分为0，导致无论节能效率参数如何变化，耦合协调度始终为0。</p>
            <p style="color: #856404;">这反映了该地区在这些方面表现较差，需要重点改善。</p>
        </div>
        """
        
        min_value = min(coupling_values)
        max_value = max(coupling_values)
        
        if all_zero:
            y_min = 0
            y_max = 0.1
        else:
            range_value = max_value - min_value
            y_min = max(0, min_value - range_value * 0.2)
            y_max = min(1, max_value + range_value * 0.2)
            
            if y_max - y_min < 0.02:
                y_min = max(0, min_value - 0.01)
                y_max = min(1, max_value + 0.01)
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{region} - 节能效率参数η敏感性分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .chart-container {{
            position: relative;
            height: 500px;
            margin: 30px 0;
        }}
        .info {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .info h3 {{
            margin-top: 0;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{region} - 节能效率参数η敏感性分析</h1>
        
        <div class="info">
            <h3>分析说明</h3>
            <p>本图表展示了在±10%范围内波动节能效率参数η时，耦合协调度的变化情况。</p>
            <p><strong>横轴：</strong>η参数波动百分比（-10% 到 +10%）</p>
            <p><strong>纵轴：</strong>耦合协调度 D 值</p>
        </div>
        
        {warning_message}
        
        <div class="chart-container">
            <canvas id="sensitivityChart"></canvas>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('sensitivityChart').getContext('2d');
        const sensitivityChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(fluctuation_labels, ensure_ascii=False)},
                datasets: [{{
                    label: '耦合协调度 D',
                    data: {json.dumps(coupling_values, ensure_ascii=False)},
                    borderColor: '#36A2EB',
                    backgroundColor: '#36A2EB33',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: '节能效率参数η波动对耦合协调度的影响',
                        font: {{
                            size: 18,
                            weight: 'bold'
                        }}
                    }},
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        callbacks: {{
                            title: function(context) {{
                                return 'η波动: ' + context[0].label;
                            }},
                            label: function(context) {{
                                const etaValues = {json.dumps(eta_values, ensure_ascii=False)};
                                return 'η值: ' + etaValues[context.dataIndex] + ' | 耦合协调度: ' + context.parsed.y.toFixed(4);
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'η参数波动百分比',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            display: true
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '耦合协调度 D 值',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }},
                        grid: {{
                            display: true
                        }},
                        min: {y_min:.4f},
                        max: {y_max:.4f},
                        ticks: {{
                            stepSize: {(y_max - y_min) / 5:.4f}
                        }}
                    }}
                }},
                interaction: {{
                    mode: 'index',
                    intersect: false
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def run_full_analysis(self):
        """运行完整的敏感性分析"""
        print("=" * 100)
        print("敏感性分析")
        print("=" * 100)
        
        print("\n步骤1: 一级指标权重敏感性分析")
        print("=" * 100)
        
        weight_results = self.analyze_weight_sensitivity()
        
        print("\n步骤2: 节能效率参数η敏感性分析")
        print("=" * 100)
        
        eta_results = self.analyze_eta_sensitivity()
        
        print("\n步骤3: 生成HTML图表")
        print("=" * 100)
        
        self.generate_weight_sensitivity_charts(weight_results)
        self.generate_eta_sensitivity_charts(eta_results)
        
        print("\n" + "=" * 100)
        print("敏感性分析完成！")
        print("=" * 100)
        
        return weight_results, eta_results


if __name__ == "__main__":    
    analyzer = SensitivityAnalysis()
    weight_results, eta_results = analyzer.run_full_analysis()
