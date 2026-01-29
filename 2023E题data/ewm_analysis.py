import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional


class EWMAnalyzer:
    """
    熵权法（EWM）分析器
    用于数据归一化处理和权重计算
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data'

        self.data_dir = data_dir
        self.raw_data = None
        self.normalized_data = None
        self.weights = {}
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

    def load_data(self, filename: str = 'combined_data.csv') -> pd.DataFrame:
        """
        加载原始数据
        """
        filepath = os.path.join(self.data_dir, filename)
        self.raw_data = pd.read_csv(filepath)
        print(f"已加载数据: {filename}")
        print(f"数据形状: {self.raw_data.shape}")
        return self.raw_data

    def min_max_normalize(self, data: pd.DataFrame, 
                         exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        最大最小归一化
        对于正向指标：x' = (x - min) / (max - min)
        对于负向指标：x' = (max - x) / (max - min)
        """
        if exclude_cols is None:
            exclude_cols = ['Region']

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

    def z_score_normalize(self, data: pd.DataFrame,
                         exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Z-score标准化
        x' = (x - mean) / std
        """
        if exclude_cols is None:
            exclude_cols = ['Region']

        normalized = data.copy()

        for col in data.columns:
            if col in exclude_cols:
                continue

            col_data = data[col].values
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)

            if std_val == 0:
                normalized[col] = 0
            else:
                normalized[col] = (col_data - mean_val) / std_val

        return normalized

    def calculate_entropy(self, data: np.ndarray) -> float:
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

    def calculate_ewm_weights(self, data: pd.DataFrame,
                             exclude_cols: List[str] = None,
                             update_global: bool = True) -> Dict[str, float]:
        """
        计算熵权
        1. 计算每个指标的信息熵
        2. 计算差异系数：d = 1 - H
        3. 计算权重：w = d / sum(d)
        """
        if exclude_cols is None:
            exclude_cols = ['Region']

        indicators = [col for col in data.columns if col not in exclude_cols]

        entropies = {}
        diversities = {}
        weights = {}

        total_diversity = 0

        for indicator in indicators:
            indicator_data = data[indicator].values

            entropy = self.calculate_entropy(indicator_data)
            entropies[indicator] = entropy

            diversity = 1 - entropy
            diversities[indicator] = diversity

            total_diversity += diversity

        for indicator in indicators:
            if total_diversity == 0:
                weights[indicator] = 1.0 / len(indicators)
            else:
                weights[indicator] = diversities[indicator] / total_diversity

        if update_global:
            self.weights = {
                'entropies': entropies,
                'diversities': diversities,
                'weights': weights
            }

        return weights

    def calculate_composite_scores(self, normalized_data: pd.DataFrame,
                                  weights: Dict[str, float],
                                  exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        计算综合得分
        Score = sum(weight * normalized_value)
        """
        if exclude_cols is None:
            exclude_cols = ['Region']

        result = normalized_data.copy()

        indicators = [col for col in normalized_data.columns if col not in exclude_cols]

        scores = []
        for idx in range(len(normalized_data)):
            score = 0
            for indicator in indicators:
                score = score + weights[indicator] * normalized_data.loc[idx, indicator]
            scores.append(score)

        result['composite_score'] = scores

        return result

    def analyze_by_category(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        按类别分析
        """
        categories = {
            'human_health': ['H1_circadian_rhythm_disorder', 'H2_metabolic_obesity_index',
                           'H3_visual_function_impairment', 'H4_psychological_emotional'],
            'social_impact': ['S1_population_density', 'S2_social_development_level',
                             'S3_traffic_accident_rate', 'S4_crime_rate'],
            'ecological_impact': ['N1_vegetation_coverage', 'N2_biodiversity_index',
                                 'N3_climate_humidity_transparency', 'N4_geographic_sunlight_elevation'],
            'energy_factor': ['E1_lighting_energy_per_area', 'E2_lighting_system_efficiency']
        }

        category_results = {}

        for category, indicators in categories.items():
            category_data = data[['Region'] + indicators].copy()

            category_normalized = self.min_max_normalize(category_data)

            category_weights = self.calculate_ewm_weights(category_normalized, update_global=False)

            category_scores = self.calculate_composite_scores(category_normalized, category_weights)

            category_results[category] = {
                'normalized': category_normalized,
                'weights': category_weights,
                'scores': category_scores
            }

        return category_results

    def run_full_analysis(self, save_results: bool = True) -> Dict:
        """
        运行完整的EWM分析流程
        """
        print("=" * 80)
        print("开始EWM熵权法分析")
        print("=" * 80)

        self.load_data('combined_data.csv')

        print("\n" + "=" * 80)
        print("步骤1: 数据归一化处理")
        print("=" * 80)

        self.normalized_data = self.min_max_normalize(self.raw_data)

        print("归一化完成！")
        print("\n归一化数据预览:")
        print(self.normalized_data.head())

        print("\n" + "=" * 80)
        print("步骤2: 计算熵权")
        print("=" * 80)

        weights = self.calculate_ewm_weights(self.normalized_data)

        print("熵权计算完成！")
        self._print_weights_summary()

        print("\n" + "=" * 80)
        print("步骤3: 计算综合得分")
        print("=" * 80)

        scored_data = self.calculate_composite_scores(self.normalized_data, weights)

        print("综合得分计算完成！")
        print("\n综合得分排名:")
        ranked = scored_data.sort_values('composite_score', ascending=False)
        print(ranked[['Region', 'composite_score']].head(10))

        print("\n" + "=" * 80)
        print("步骤4: 按类别分析")
        print("=" * 80)

        category_results = self.analyze_by_category(self.raw_data)

        print("按类别分析完成！")
        self._print_category_summary(category_results)

        if save_results:
            print("\n" + "=" * 80)
            print("步骤5: 保存结果")
            print("=" * 80)

            self._save_results(scored_data, category_results)

        results = {
            'raw_data': self.raw_data,
            'normalized_data': self.normalized_data,
            'weights': weights,
            'scored_data': scored_data,
            'category_results': category_results
        }

        print("\n" + "=" * 80)
        print("EWM分析完成！")
        print("=" * 80)

        return results

    def _print_weights_summary(self):
        """
        打印权重摘要
        """
        weights = self.weights['weights']

        print("\n各指标权重（按权重降序排列）:")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for idx, (indicator, weight) in enumerate(sorted_weights, 1):
            print(f"{idx:2d}. {indicator:40s} : {weight:.4f}")

        print(f"\n权重总和: {sum(weights.values()):.4f}")

    def _print_category_summary(self, category_results: Dict):
        """
        打印类别分析摘要
        """
        category_names = {
            'human_health': '人类健康影响',
            'social_impact': '社会影响',
            'ecological_impact': '生态影响',
            'energy_factor': '能源因素'
        }

        for category, results in category_results.items():
            print(f"\n{category_names.get(category, category)}:")
            weights = results['weights']

            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for indicator, weight in sorted_weights[:3]:
                print(f"  - {indicator}: {weight:.4f}")

            scores = results['scores']
            top_region = scores.loc[scores['composite_score'].idxmax(), 'Region']
            top_score = scores['composite_score'].max()
            print(f"  最优地区: {top_region} (得分: {top_score:.4f})")

    def _save_results(self, scored_data: pd.DataFrame, 
                     category_results: Dict):
        """
        保存分析结果
        """
        output_dir = os.path.join(self.data_dir, 'normalized_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        normalized_file = os.path.join(output_dir, 'normalized_data.csv')
        self.normalized_data.to_csv(normalized_file, index=False, encoding='utf-8-sig')
        print(f"已保存归一化数据: {normalized_file}")

        scored_file = os.path.join(output_dir, 'scored_data.csv')
        scored_data.to_csv(scored_file, index=False, encoding='utf-8-sig')
        print(f"已保存得分数据: {scored_file}")

        weights_df = pd.DataFrame({
            'indicator': list(self.weights['weights'].keys()),
            'entropy': list(self.weights['entropies'].values()),
            'diversity': list(self.weights['diversities'].values()),
            'weight': list(self.weights['weights'].values())
        })
        weights_file = os.path.join(output_dir, 'weights.csv')
        weights_df.to_csv(weights_file, index=False, encoding='utf-8-sig')
        print(f"已保存权重数据: {weights_file}")

        for category, results in category_results.items():
            category_normalized_file = os.path.join(output_dir, f'{category}_normalized.csv')
            results['normalized'].to_csv(category_normalized_file, index=False, encoding='utf-8-sig')

            category_scored_file = os.path.join(output_dir, f'{category}_scored.csv')
            results['scores'].to_csv(category_scored_file, index=False, encoding='utf-8-sig')

        print(f"\n所有结果已保存到目录: {output_dir}")

    def print_statistics(self):
        """
        打印统计信息
        """
        if self.normalized_data is None:
            print("请先运行分析！")
            return

        print("\n" + "=" * 80)
        print("数据统计信息")
        print("=" * 80)

        print("\n原始数据统计:")
        print(self.raw_data.describe())

        print("\n归一化数据统计:")
        print(self.normalized_data.describe())


def main():
    """
    主函数
    """
    analyzer = EWMAnalyzer()

    results = analyzer.run_full_analysis(save_results=True)

    analyzer.print_statistics()

    return analyzer, results


if __name__ == '__main__':
    analyzer, results = main()
