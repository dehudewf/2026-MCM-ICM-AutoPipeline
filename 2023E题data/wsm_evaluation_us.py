import numpy as np
import pandas as pd
import os
from typing import Dict, List


class WSMEvaluatorator:
    """
    线性加权综合评价模型（WSM - Weighted Sum Model）
    用于计算美国境内四个地点的综合得分
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data'

        self.data_dir = data_dir
        self.indicator_direction = self._define_indicator_direction()

    def _define_indicator_direction(self) -> Dict[str, str]:
        """
        定义指标方向
        'positive': 越大越好（正向指标）
        'negative': 越小越好（负向指标）
        """
        return {
            'H1_circadian_rhythm_disorder': 'negative',
            'H2_metabolic_obesity_index': 'negative',
            'H3_visual_function_impairment': 'negative',
            'H4_psychological_emotional': 'negative',
            'S1_population_density': 'positive',
            'S2_social_development_level': 'positive',
            'S3_traffic_accident_rate': 'negative',
            'S4_crime_rate': 'negative',
            'N1_vegetation_coverage': 'positive',
            'N2_biodiversity_index': 'positive',
            'N3_climate_humidity_transparency': 'positive',
            'N4_geographic_sunlight_elevation': 'positive',
            'E1_lighting_energy_per_area': 'negative',
            'E2_lighting_system_efficiency': 'positive'
        }

    def load_real_locations_data(self) -> pd.DataFrame:
        """
        加载美国境内真实地点数据
        """
        filepath = os.path.join(self.data_dir, 'real_locations_data_us.csv')
        data = pd.read_csv(filepath)
        print("已加载美国境内真实地点数据")
        print(f"数据形状: {data.shape}")
        return data

    def load_weights(self) -> pd.DataFrame:
        """
        加载二级指标权重
        """
        filepath = os.path.join(self.data_dir, 'normalized_results', 'weights.csv')
        weights = pd.read_csv(filepath)
        print("\n已加载二级指标权重")
        print(weights)
        return weights

    def min_max_normalize(self, data: pd.DataFrame,
                         exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        最大最小归一化
        对于正向指标：x' = (x - min) / (max - min)
        对于负向指标：x' = (max - x) / (max - min)
        """
        if exclude_cols is None:
            exclude_cols = ['Region', 'Region_Name', 'Region_Type', 'State']

        normalized = data.copy()

        for col in data.columns:
            if col in exclude_cols:
                continue

            col_data = data[col].values
            min_val = np.min(col_data)
            max_val = np.max(col_data)

            if max_val == min_val:
                normalized[col] = 0.5
            else:
                direction = self.indicator_direction.get(col, 'positive')

                if direction == 'positive':
                    normalized[col] = (col_data - min_val) / (max_val - min_val)
                else:
                    normalized[col] = (max_val - col_data) / (max_val - min_val)

        return normalized

    def calculate_wsm_scores(self, normalized_data: pd.DataFrame,
                           weights: pd.DataFrame) -> pd.DataFrame:
        """
        计算WSM综合得分
        Score = Σ(weight_i * normalized_value_i)
        """
        exclude_cols = ['Region', 'Region_Name', 'Region_Type', 'State']
        indicators = [col for col in normalized_data.columns if col not in exclude_cols]

        weight_dict = dict(zip(weights['indicator'], weights['weight']))

        scores = []
        for idx in range(len(normalized_data)):
            score = 0
            for indicator in indicators:
                weight = weight_dict.get(indicator, 0)
                value = normalized_data.loc[idx, indicator]
                score = score + weight * value
            scores.append(score)

        result = normalized_data.copy()
        result['WSM_Score'] = scores
        result['WSM_Rank'] = result['WSM_Score'].rank(ascending=False).astype(int)

        return result

    def calculate_category_scores(self, normalized_data: pd.DataFrame,
                               weights: pd.DataFrame) -> Dict[str, List[float]]:
        """
        计算各一级指标类别的得分
        """
        category_mapping = {
            'human_health': {
                'indicators': ['H1_circadian_rhythm_disorder', 'H2_metabolic_obesity_index',
                              'H3_visual_function_impairment', 'H4_psychological_emotional'],
                'name': '人类健康影响'
            },
            'social_impact': {
                'indicators': ['S1_population_density', 'S2_social_development_level',
                              'S3_traffic_accident_rate', 'S4_crime_rate'],
                'name': '社会影响'
            },
            'ecological_impact': {
                'indicators': ['N1_vegetation_coverage', 'N2_biodiversity_index',
                              'N3_climate_humidity_transparency', 'N4_geographic_sunlight_elevation'],
                'name': '生态影响'
            },
            'energy_factor': {
                'indicators': ['E1_lighting_energy_per_area', 'E2_lighting_system_efficiency'],
                'name': '能源因素'
            }
        }

        weight_dict = dict(zip(weights['indicator'], weights['weight']))

        category_scores = {}

        for category, info in category_mapping.items():
            indicators = info['indicators']

            scores = []
            for idx in range(len(normalized_data)):
                score = 0
                for indicator in indicators:
                    weight = weight_dict.get(indicator, 0)
                    value = normalized_data.loc[idx, indicator]
                    score = score + weight * value
                scores.append(score)

            category_scores[category] = scores

        return category_scores

    def print_comparison(self, original_data: pd.DataFrame,
                       normalized_data: pd.DataFrame,
                       scored_data: pd.DataFrame):
        """
        打印原始数据、归一化数据和得分的对比
        """
        print("\n" + "=" * 100)
        print("数据对比：原始值 vs 归一化值 vs WSM得分")
        print("=" * 100)

        indicators = [col for col in original_data.columns 
                    if col not in ['Region', 'Region_Name', 'Region_Type', 'State']]

        for idx, row in original_data.iterrows():
            region = row['Region']
            region_name = row['Region_Name']
            region_type = row['Region_Type']
            state = row['State']
            wsm_score = scored_data.loc[idx, 'WSM_Score']
            wsm_rank = scored_data.loc[idx, 'WSM_Rank']

            print(f"\n{'=' * 100}")
            print(f"地区: {region_name} ({region_type}, {state})")
            print(f"{'=' * 100}")
            print(f"\nWSM综合得分: {wsm_score:.4f}")
            print(f"排名: {wsm_rank}")
            print(f"\n{'指标':<40} {'原始值':<15} {'归一化值':<15} {'权重':<10} {'贡献':<10}")
            print("-" * 100)

            for indicator in indicators:
                original_val = row[indicator]
                normalized_val = normalized_data.loc[idx, indicator]
                weight = self.weights_df[
                    self.weights_df['indicator'] == indicator
                ]['weight'].values[0]
                contribution = weight * normalized_val

                print(f"{indicator:<40} {original_val:<15.4f} {normalized_val:<15.4f} {weight:<10.4f} {contribution:<10.4f}")

    def print_category_summary(self, category_scores: Dict[str, List[float]]):
        """
        打印各一级指标类别得分摘要
        """
        category_names = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        print("\n" + "=" * 100)
        print("各一级指标类别得分")
        print("=" * 100)

        for category, scores in category_scores.items():
            name = category_names.get(category, category)
            print(f"\n{name}:")
            for idx, score in enumerate(scores):
                region_name = self.original_data.loc[idx, 'Region_Name']
                print(f"  {region_name}: {score:.4f}")

    def save_results(self, normalized_data: pd.DataFrame,
                   scored_data: pd.DataFrame,
                   category_scores: Dict[str, List[float]]):
        """
        保存结果
        """
        output_dir = os.path.join(self.data_dir, 'wsm_results_us')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        normalized_file = os.path.join(output_dir, 'normalized_real_locations_us.csv')
        normalized_data.to_csv(normalized_file, index=False, encoding='utf-8-sig')
        print(f"\n归一化数据已保存: {normalized_file}")

        scored_file = os.path.join(output_dir, 'wsm_scores_us.csv')
        scored_data.to_csv(scored_file, index=False, encoding='utf-8-sig')
        print(f"WSM得分数据已保存: {scored_file}")

        category_df = pd.DataFrame(category_scores)
        category_df.index = self.original_data['Region_Name']
        category_file = os.path.join(output_dir, 'category_scores_us.csv')
        category_df.to_csv(category_file, encoding='utf-8-sig')
        print(f"类别得分数据已保存: {category_file}")

        print(f"\n所有结果已保存到目录: {output_dir}")

    def run_full_evaluation(self):
        """
        运行完整的WSM评价流程
        """
        print("=" * 100)
        print("线性加权综合评价模型（WSM）- 美国境内四个地点")
        print("=" * 100)

        print("\n步骤1: 加载美国境内真实地点数据")
        print("=" * 100)
        self.original_data = self.load_real_locations_data()

        print("\n步骤2: 加载二级指标权重")
        print("=" * 100)
        self.weights_df = self.load_weights()

        print("\n步骤3: 数据归一化处理")
        print("=" * 100)
        normalized_data = self.min_max_normalize(self.original_data)
        print("归一化完成！")
        print("\n归一化数据预览:")
        print(normalized_data[['Region', 'Region_Name', 'State',
                           'H1_circadian_rhythm_disorder', 'S1_population_density',
                           'N1_vegetation_coverage', 'E1_lighting_energy_per_area']].head())

        print("\n步骤4: 计算WSM综合得分")
        print("=" * 100)
        scored_data = self.calculate_wsm_scores(normalized_data, self.weights_df)

        print("\nWSM得分排名:")
        ranked = scored_data.sort_values('WSM_Score', ascending=False)
        print(ranked[['Region_Name', 'State', 'WSM_Score', 'WSM_Rank']])

        print("\n步骤5: 计算各一级指标类别得分")
        print("=" * 100)
        category_scores = self.calculate_category_scores(normalized_data, self.weights_df)
        self.print_category_summary(category_scores)

        print("\n步骤6: 数据对比分析")
        print("=" * 100)
        self.print_comparison(self.original_data, normalized_data, scored_data)

        print("\n步骤7: 保存结果")
        print("=" * 100)
        self.save_results(normalized_data, scored_data, category_scores)

        print("\n" + "=" * 100)
        print("WSM评价完成！")
        print("=" * 100)

        results = {
            'original_data': self.original_data,
            'normalized_data': normalized_data,
            'scored_data': scored_data,
            'category_scores': category_scores
        }

        return results


def main():
    """
    主函数
    """
    evaluator = WSMEvaluatorator()

    results = evaluator.run_full_evaluation()

    return evaluator, results


if __name__ == '__main__':
    evaluator, results = main()
