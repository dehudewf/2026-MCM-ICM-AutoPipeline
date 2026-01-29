import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class LightPollutionData:
    """
    光污染影响评估原始数据类
    包含所有一级和二级指标数据，用于EWM分析和归一化处理
    """

    def __init__(self):
        self.regions = [
            'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5',
            'Region_6', 'Region_7', 'Region_8', 'Region_9', 'Region_10',
            'Region_11', 'Region_12', 'Region_13', 'Region_14', 'Region_15'
        ]

        self.data = self._generate_sample_data()

    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        生成示例数据
        实际使用时，应替换为从图片或数据源提取的真实数据
        """
        np.random.seed(42)

        data = {}

        data['human_health'] = pd.DataFrame({
            'Region': self.regions,
            'H1_circadian_rhythm_disorder': np.random.uniform(0.15, 0.85, 15),
            'H2_metabolic_obesity_index': np.random.uniform(0.20, 0.90, 15),
            'H3_visual_function_impairment': np.random.uniform(0.10, 0.75, 15),
            'H4_psychological_emotional': np.random.uniform(0.18, 0.82, 15)
        })

        data['social_impact'] = pd.DataFrame({
            'Region': self.regions,
            'S1_population_density': np.random.uniform(100, 5000, 15),
            'S2_social_development_level': np.random.uniform(0.3, 0.95, 15),
            'S3_traffic_accident_rate': np.random.uniform(2.5, 15.0, 15),
            'S4_crime_rate': np.random.uniform(150, 800, 15)
        })

        data['ecological_impact'] = pd.DataFrame({
            'Region': self.regions,
            'N1_vegetation_coverage': np.random.uniform(0.15, 0.85, 15),
            'N2_biodiversity_index': np.random.uniform(0.2, 0.9, 15),
            'N3_climate_humidity_transparency': np.random.uniform(0.4, 0.95, 15),
            'N4_geographic_sunlight_elevation': np.random.uniform(0.25, 0.88, 15)
        })

        data['energy_factor'] = pd.DataFrame({
            'Region': self.regions,
            'E1_lighting_energy_per_area': np.random.uniform(50, 300, 15),
            'E2_lighting_system_efficiency': np.random.uniform(0.4, 0.92, 15)
        })

        return data

    def get_combined_data(self) -> pd.DataFrame:
        """
        获取合并后的完整数据集
        """
        combined = self.data['human_health'].copy()
        combined = combined.merge(self.data['social_impact'], on='Region')
        combined = combined.merge(self.data['ecological_impact'], on='Region')
        combined = combined.merge(self.data['energy_factor'], on='Region')

        return combined

    def get_indicator_hierarchy(self) -> Dict[str, Dict[str, List[str]]]:
        """
        返回指标层次结构
        """
        return {
            'human_health': {
                'description': '人类健康影响',
                'indicators': ['H1_circadian_rhythm_disorder', 'H2_metabolic_obesity_index',
                              'H3_visual_function_impairment', 'H4_psychological_emotional']
            },
            'social_impact': {
                'description': '社会影响',
                'indicators': ['S1_population_density', 'S2_social_development_level',
                              'S3_traffic_accident_rate', 'S4_crime_rate']
            },
            'ecological_impact': {
                'description': '生态影响',
                'indicators': ['N1_vegetation_coverage', 'N2_biodiversity_index',
                              'N3_climate_humidity_transparency', 'N4_geographic_sunlight_elevation']
            },
            'energy_factor': {
                'description': '能源因素',
                'indicators': ['E1_lighting_energy_per_area', 'E2_lighting_system_efficiency']
            }
        }

    def get_indicator_descriptions(self) -> Dict[str, str]:
        """
        返回各指标的详细描述
        """
        return {
            'H1_circadian_rhythm_disorder': '昼夜节律失调 - 基于NHANES睡眠数据',
            'H2_metabolic_obesity_index': '代谢与肥胖指标 - BMI、腰围、血糖等',
            'H3_visual_function_impairment': '视功能损害 - 视力障碍与眼疾统计',
            'H4_psychological_emotional': '心理与情绪 - 精神压力、抑郁和焦虑',
            'S1_population_density': '人口密度 - 每平方公里人口数',
            'S2_social_development_level': '社会发展水平 - 人均收入、受教育程度',
            'S3_traffic_accident_rate': '交通事故率 - 每万人事故数',
            'S4_crime_rate': '犯罪率 - 每万人犯罪数',
            'N1_vegetation_coverage': '植被覆盖率 - 森林、草地比例',
            'N2_biodiversity_index': '生物多样性 - 物种分布和丰度',
            'N3_climate_humidity_transparency': '气候(湿度/透明度) - 大气透明度和湿度',
            'N4_geographic_sunlight_elevation': '地理(日照/海拔) - 日照时数和海拔',
            'E1_lighting_energy_per_area': '单位面积照明能耗 - kWh/m²',
            'E2_lighting_system_efficiency': '照明系统效率 - LED普及率和效率'
        }

    def get_data_sources(self) -> Dict[str, str]:
        """
        返回数据来源信息
        """
        return {
            'H1': 'CDC - NHANES (国家健康与营养调查)',
            'H2': 'CDC - NHANES',
            'H3': 'CDC - Vision Health Data',
            'H4': 'SAMHSA / BRFSS',
            'S1': 'U.S. Census Bureau (美国普查局)',
            'S2': 'BEA / ACS',
            'S3': 'NHTSA - FARS (致命事故分析系统)',
            'S4': 'FBI - Crime Data Explorer',
            'N1': 'USGS - NLCD (国家土地覆盖数据库)',
            'N2': 'GBIF / NatureServe',
            'N3': 'NOAA - NCEI',
            'N4': 'USGS / NOAA',
            'E1': 'U.S. EIA (能源信息署)',
            'E2': 'DOE - SSL Program'
        }

    def save_to_csv(self, output_dir: str = '.'):
        """
        将数据保存为CSV文件
        """
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for category, df in self.data.items():
            filepath = os.path.join(output_dir, f'{category}.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

        combined = self.get_combined_data()
        combined_filepath = os.path.join(output_dir, 'combined_data.csv')
        combined.to_csv(combined_filepath, index=False, encoding='utf-8-sig')

        print(f"数据已保存到目录: {output_dir}")

    def print_data_summary(self):
        """
        打印数据摘要信息
        """
        print("=" * 80)
        print("光污染影响评估原始数据摘要")
        print("=" * 80)
        print(f"\n地区数量: {len(self.regions)}")
        print(f"指标类别数: {len(self.data)}")

        total_indicators = sum(len(df.columns) - 1 for df in self.data.values())
        print(f"总指标数: {total_indicators}")

        print("\n" + "=" * 80)
        print("指标类别详情:")
        print("=" * 80)

        hierarchy = self.get_indicator_hierarchy()
        for category, info in hierarchy.items():
            print(f"\n{info['description']} ({category}):")
            for indicator in info['indicators']:
                desc = self.get_indicator_descriptions().get(indicator, '')
                print(f"  - {indicator}: {desc}")

        print("\n" + "=" * 80)
        print("数据来源:")
        print("=" * 80)

        sources = self.get_data_sources()
        for key, source in sources.items():
            print(f"{key}: {source}")


def main():
    """
    主函数：创建数据对象并显示摘要
    """
    lp_data = LightPollutionData()

    lp_data.print_data_summary()

    print("\n" + "=" * 80)
    print("合并数据预览:")
    print("=" * 80)
    combined = lp_data.get_combined_data()
    print(combined.head())

    print("\n" + "=" * 80)
    print("数据统计信息:")
    print("=" * 80)
    print(combined.describe())

    output_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data'
    lp_data.save_to_csv(output_dir)

    return lp_data


if __name__ == '__main__':
    data = main()
