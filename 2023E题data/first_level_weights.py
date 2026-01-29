import numpy as np
import pandas as pd
import os
from typing import Dict, List


class FirstLevelWeightsCalculator:
    """
    一级指标权重计算器
    基于二级指标的权重和类别综合得分计算一级指标权重
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data'

        self.data_dir = data_dir
        self.normalized_results_dir = os.path.join(data_dir, 'normalized_results')
        self.second_level_weights = None
        self.first_level_weights = None

    def load_second_level_weights(self) -> pd.DataFrame:
        """
        加载二级指标权重
        """
        weights_file = os.path.join(self.normalized_results_dir, 'weights.csv')
        self.second_level_weights = pd.read_csv(weights_file)
        print("已加载二级指标权重")
        print(self.second_level_weights)
        return self.second_level_weights

    def load_category_scores(self) -> Dict[str, pd.DataFrame]:
        """
        加载各一级指标类别的得分数据
        """
        categories = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        category_scores = {}

        for category, description in categories.items():
            score_file = os.path.join(self.normalized_results_dir, f'{category}_scored.csv')
            if os.path.exists(score_file):
                category_scores[category] = pd.read_csv(score_file)
                print(f"\n已加载 {description} 得分数据")
                print(f"  文件: {category}_scored.csv")
                print(f"  形状: {category_scores[category].shape}")

        return category_scores

    def calculate_first_level_weights_by_ewm(self, category_scores: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        使用EWM方法计算一级指标权重
        基于各一级指标类别的综合得分
        """
        categories = list(category_scores.keys())
        regions = category_scores[categories[0]]['Region'].values
        n_regions = len(regions)

        score_matrix = []
        for category in categories:
            scores = category_scores[category]['composite_score'].values
            score_matrix.append(scores)

        score_matrix = np.array(score_matrix)

        print("\n" + "=" * 80)
        print("一级指标综合得分矩阵")
        print("=" * 80)
        print(f"矩阵形状: {score_matrix.shape}")
        print("\n各一级指标的平均得分:")
        for idx, category in enumerate(categories):
            mean_score = np.mean(score_matrix[idx])
            print(f"  {category}: {mean_score:.4f}")

        entropies = {}
        diversities = {}
        weights = {}

        total_diversity = 0

        for idx, category in enumerate(categories):
            category_scores_data = score_matrix[idx]

            entropy = self._calculate_entropy(category_scores_data)
            entropies[category] = entropy

            diversity = 1 - entropy
            diversities[category] = diversity

            total_diversity += diversity

        for category in categories:
            if total_diversity == 0:
                weights[category] = 1.0 / len(categories)
            else:
                weights[category] = diversities[category] / total_diversity

        self.first_level_weights = {
            'entropies': entropies,
            'diversities': diversities,
            'weights': weights
        }

        return weights

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """
        计算信息熵
        H = -sum(p * ln(p)) / ln(n)
        """
        n = len(data)

        if n == 0:
            return 1.0

        data = np.array(data, dtype=float)
        data = np.abs(data)

        data_sum = np.sum(data)
        if data_sum == 0:
            return 1.0

        p = data / data_sum
        p = p[p > 0]

        if len(p) == 0:
            return 1.0

        entropy = -np.sum(p * np.log(p)) / np.log(n)

        return entropy

    def calculate_first_level_weights_by_aggregation(self) -> Dict[str, float]:
        """
        通过二级指标权重聚合计算一级指标权重
        一级指标权重 = 该类别下所有二级指标权重之和
        """
        if self.second_level_weights is None:
            self.load_second_level_weights()

        category_mapping = {
            'human_health': ['H1_circadian_rhythm_disorder', 'H2_metabolic_obesity_index',
                           'H3_visual_function_impairment', 'H4_psychological_emotional'],
            'social_impact': ['S1_population_density', 'S2_social_development_level',
                             'S3_traffic_accident_rate', 'S4_crime_rate'],
            'ecological_impact': ['N1_vegetation_coverage', 'N2_biodiversity_index',
                                 'N3_climate_humidity_transparency', 'N4_geographic_sunlight_elevation'],
            'energy_factor': ['E1_lighting_energy_per_area', 'E2_lighting_system_efficiency']
        }

        weights = {}
        for category, indicators in category_mapping.items():
            category_weight = 0
            for indicator in indicators:
                indicator_weight = self.second_level_weights[
                    self.second_level_weights['indicator'] == indicator
                ]['weight'].values[0]
                category_weight += indicator_weight
            weights[category] = category_weight

        print("\n" + "=" * 80)
        print("通过二级指标权重聚合计算一级指标权重")
        print("=" * 80)

        for category, weight in weights.items():
            print(f"{category}: {weight:.4f}")

        return weights

    def print_first_level_weights_summary(self, weights: Dict[str, float]):
        """
        打印一级指标权重摘要
        """
        category_names = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        print("\n" + "=" * 80)
        print("一级指标权重")
        print("=" * 80)

        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for idx, (category, weight) in enumerate(sorted_weights, 1):
            name = category_names.get(category, category)
            print(f"{idx}. {name} ({category}): {weight:.4f} ({weight*100:.2f}%)")

        total_weight = sum(weights.values())
        print(f"\n权重总和: {total_weight:.4f}")

        print("\n" + "=" * 80)
        print("信息熵和差异系数")
        print("=" * 80)

        if self.first_level_weights is not None:
            for category in weights.keys():
                name = category_names.get(category, category)
                entropy = self.first_level_weights['entropies'][category]
                diversity = self.first_level_weights['diversities'][category]
                weight = self.first_level_weights['weights'][category]
                print(f"\n{name} ({category}):")
                print(f"  信息熵: {entropy:.4f}")
                print(f"  差异系数: {diversity:.4f}")
                print(f"  权重: {weight:.4f}")

    def save_first_level_weights(self, weights: Dict[str, float], method: str = 'ewm'):
        """
        保存一级指标权重
        """
        category_names = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        weights_df = pd.DataFrame({
            'category': list(weights.keys()),
            'category_name': [category_names.get(cat, cat) for cat in weights.keys()],
            'weight': list(weights.values())
        })

        output_file = os.path.join(self.normalized_results_dir, f'first_level_weights_{method}.csv')
        weights_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n一级指标权重已保存: {output_file}")

        if self.first_level_weights is not None:
            detailed_df = pd.DataFrame({
                'category': list(weights.keys()),
                'category_name': [category_names.get(cat, cat) for cat in weights.keys()],
                'entropy': list(self.first_level_weights['entropies'].values()),
                'diversity': list(self.first_level_weights['diversities'].values()),
                'weight': list(weights.values())
            })

            detailed_file = os.path.join(self.normalized_results_dir, f'first_level_weights_{method}_detailed.csv')
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')

            print(f"一级指标权重详细信息已保存: {detailed_file}")

    def run_full_analysis(self):
        """
        运行完整的一级指标权重分析
        """
        print("=" * 80)
        print("一级指标权重计算")
        print("=" * 80)

        print("\n步骤1: 加载二级指标权重")
        print("=" * 80)
        self.load_second_level_weights()

        print("\n步骤2: 加载各一级指标类别得分")
        print("=" * 80)
        category_scores = self.load_category_scores()

        print("\n步骤3: 使用EWM方法计算一级指标权重")
        print("=" * 80)
        weights_ewm = self.calculate_first_level_weights_by_ewm(category_scores)

        print("\n步骤4: 打印权重摘要")
        print("=" * 80)
        self.print_first_level_weights_summary(weights_ewm)

        print("\n步骤5: 保存结果")
        print("=" * 80)
        self.save_first_level_weights(weights_ewm, method='ewm')

        print("\n步骤6: 计算聚合权重（对比）")
        print("=" * 80)
        weights_aggregation = self.calculate_first_level_weights_by_aggregation()

        print("\n" + "=" * 80)
        print("两种方法对比")
        print("=" * 80)

        category_names = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        print(f"\n{'类别':<20} {'EWM权重':<12} {'聚合权重':<12} {'差异':<12}")
        print("-" * 60)
        for category in weights_ewm.keys():
            name = category_names.get(category, category)
            w_ewm = weights_ewm[category]
            w_agg = weights_aggregation[category]
            diff = abs(w_ewm - w_agg)
            print(f"{name:<20} {w_ewm:<12.4f} {w_agg:<12.4f} {diff:<12.4f}")

        print("\n" + "=" * 80)
        print("一级指标权重计算完成！")
        print("=" * 80)

        return {
            'weights_ewm': weights_ewm,
            'weights_aggregation': weights_aggregation,
            'category_scores': category_scores
        }


def main():
    """
    主函数
    """
    calculator = FirstLevelWeightsCalculator()

    results = calculator.run_full_analysis()

    return calculator, results


if __name__ == '__main__':
    calculator, results = main()
