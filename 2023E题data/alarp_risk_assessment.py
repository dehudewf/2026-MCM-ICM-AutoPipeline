import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple


class ALARPRiskAssessment:
    """
    ALARP（As Low As Reasonably Practicable）风险评估模型
    用于评估光污染影响的风险等级
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = r'c:\Users\liuyu\Desktop\数模桌面文件\美赛2023E题复现\2023E题data'

        self.data_dir = data_dir

        self.alarp_thresholds = self._define_alarp_thresholds()

    def _define_alarp_thresholds(self) -> Dict[str, float]:
        """
        定义ALARP模型的临界阈值
        
        ALARP模型将风险分为三个区域：
        1. 不可接受区域（Unacceptable）：风险值 > upper_threshold
        2. ALARP区域：lower_threshold < 风险值 <= upper_threshold
        3. 可接受区域（Broadly Acceptable）：风险值 <= lower_threshold
        
        对于光污染影响评估：
        - WSM得分越高，综合表现越好，风险越低
        - 风险值 = 1 - WSM得分
        
        推荐临界值：
        - 上限阈值（upper_threshold）：0.65（风险值超过65%为不可接受）
        - 下限阈值（lower_threshold）：0.35（风险值低于35%为可接受）
        """
        return {
            'upper_threshold': 0.65,
            'lower_threshold': 0.35,
            'description': {
                'upper_threshold': '不可接受区域上限',
                'lower_threshold': '可接受区域下限'
            }
        }

    def load_wsm_scores(self) -> pd.DataFrame:
        """
        加载WSM得分数据
        """
        filepath = os.path.join(self.data_dir, 'wsm_results', 'wsm_scores.csv')
        data = pd.read_csv(filepath)
        print("已加载WSM得分数据")
        print(f"数据形状: {data.shape}")
        return data

    def calculate_risk_values(self, wsm_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算风险值
        
        风险值 = 1 - WSM得分
        
        WSM得分越高，表示综合表现越好，风险越低
        """
        result = wsm_data.copy()
        result['Risk_Value'] = 1 - result['WSM_Score']
        result['Risk_Percentage'] = result['Risk_Value'] * 100
        return result

    def classify_risk_level(self, risk_value: float) -> Tuple[str, str, str]:
        """
        根据风险值划分风险等级
        
        参数:
            risk_value: 风险值（0-1之间）
            
        返回:
            (level, category, description)
            - level: 风险等级（1-5级）
            - category: 风险类别（不可接受/ALARP/可接受）
            - description: 详细描述
        """
        upper_threshold = self.alarp_thresholds['upper_threshold']
        lower_threshold = self.alarp_thresholds['lower_threshold']

        if risk_value > upper_threshold:
            level = 5
            category = '不可接受区域'
            description = '风险过高，必须立即采取措施降低风险'
        elif risk_value > lower_threshold:
            level = 3
            category = 'ALARP区域'
            description = '风险在可接受范围内，但应尽可能降低'
        else:
            level = 1
            category = '可接受区域'
            description = '风险较低，可以接受'

        return (level, category, description)

    def apply_alarp_classification(self, risk_data: pd.DataFrame) -> pd.DataFrame:
        """
        应用ALARP分类
        """
        risk_levels = []
        risk_categories = []
        risk_descriptions = []

        for idx in range(len(risk_data)):
            risk_value = risk_data.loc[idx, 'Risk_Value']
            level, category, description = self.classify_risk_level(risk_value)
            
            risk_levels.append(level)
            risk_categories.append(category)
            risk_descriptions.append(description)

        risk_data['Risk_Level'] = risk_levels
        risk_data['Risk_Category'] = risk_categories
        risk_data['Risk_Description'] = risk_descriptions

        return risk_data

    def print_alarp_thresholds(self):
        """
        打印ALARP模型阈值说明
        """
        print("\n" + "=" * 100)
        print("ALARP模型风险等级划分")
        print("=" * 100)
        
        print("\n📊 ALARP（As Low As Reasonably Practicable）模型")
        print("-" * 100)
        print("ALARP是风险管理中常用的风险评估框架，将风险分为三个区域：")
        print()
        
        upper_threshold = self.alarp_thresholds['upper_threshold']
        lower_threshold = self.alarp_thresholds['lower_threshold']
        
        print(f"1️⃣ 不可接受区域（Unacceptable Region）")
        print(f"   风险值范围：{upper_threshold:.2f} < Risk ≤ 1.00")
        print(f"   风险等级：5级（极高风险）")
        print(f"   措施：必须立即采取措施降低风险")
        print()
        
        print(f"2️⃣ ALARP区域（As Low As Reasonably Practicable Region）")
        print(f"   风险值范围：{lower_threshold:.2f} < Risk ≤ {upper_threshold:.2f}")
        print(f"   风险等级：3级（中等风险）")
        print(f"   措施：风险在可接受范围内，但应尽可能降低")
        print()
        
        print(f"3️⃣ 可接受区域（Broadly Acceptable Region）")
        print(f"   风险值范围：0.00 ≤ Risk ≤ {lower_threshold:.2f}")
        print(f"   风险等级：1级（低风险）")
        print(f"   措施：风险较低，可以接受")
        print()
        
        print("=" * 100)
        print("推荐临界值（针对光污染影响评估）：")
        print("=" * 100)
        print(f"✅ 上限阈值（Upper Threshold）：{upper_threshold:.2f}")
        print(f"   含义：风险值超过{upper_threshold*100:.0f}%为不可接受")
        print(f"   理由：光污染影响严重，对人类健康和生态环境造成重大威胁")
        print()
        print(f"✅ 下限阈值（Lower Threshold）：{lower_threshold:.2f}")
        print(f"   含义：风险值低于{lower_threshold*100:.0f}%为可接受")
        print(f"   理由：光污染影响较小，在可控范围内")
        print()

    def print_risk_assessment_results(self, risk_data: pd.DataFrame):
        """
        打印风险评估结果
        """
        print("\n" + "=" * 100)
        print("四个地区风险评估结果")
        print("=" * 100)
        
        print(f"\n{'地区':<25} {'WSM得分':<12} {'风险值':<12} {'风险百分比':<12} {'风险等级':<10} {'风险类别':<15}")
        print("-" * 100)
        
        for idx in range(len(risk_data)):
            region_name = risk_data.loc[idx, 'Region_Name']
            wsm_score = risk_data.loc[idx, 'WSM_Score']
            risk_value = risk_data.loc[idx, 'Risk_Value']
            risk_percentage = risk_data.loc[idx, 'Risk_Percentage']
            risk_level = risk_data.loc[idx, 'Risk_Level']
            risk_category = risk_data.loc[idx, 'Risk_Category']
            
            print(f"{region_name:<25} {wsm_score:<12.4f} {risk_value:<12.4f} {risk_percentage:<12.2f}% {risk_level:<10} {risk_category:<15}")
        
        print("\n" + "=" * 100)
        print("详细风险评估")
        print("=" * 100)
        
        for idx in range(len(risk_data)):
            region_name = risk_data.loc[idx, 'Region_Name']
            region_type = risk_data.loc[idx, 'Region_Type']
            wsm_score = risk_data.loc[idx, 'WSM_Score']
            risk_value = risk_data.loc[idx, 'Risk_Value']
            risk_percentage = risk_data.loc[idx, 'Risk_Percentage']
            risk_level = risk_data.loc[idx, 'Risk_Level']
            risk_category = risk_data.loc[idx, 'Risk_Category']
            risk_description = risk_data.loc[idx, 'Risk_Description']
            
            print(f"\n{'=' * 100}")
            print(f"📍 地区：{region_name}（{region_type}）")
            print(f"{'=' * 100}")
            print(f"\n📊 评估指标：")
            print(f"   - WSM综合得分：{wsm_score:.4f}")
            print(f"   - 风险值：{risk_value:.4f}")
            print(f"   - 风险百分比：{risk_percentage:.2f}%")
            print(f"\n🎯 风险等级：{risk_level}级")
            print(f"📋 风险类别：{risk_category}")
            print(f"💡 措施建议：{risk_description}")

    def save_results(self, risk_data: pd.DataFrame):
        """
        保存结果
        """
        output_dir = os.path.join(self.data_dir, 'alarp_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        risk_file = os.path.join(output_dir, 'alarp_risk_assessment.csv')
        risk_data.to_csv(risk_file, index=False, encoding='utf-8-sig')
        print(f"\nALARP风险评估结果已保存: {risk_file}")

        print(f"\n所有结果已保存到目录: {output_dir}")

    def run_full_assessment(self):
        """
        运行完整的ALARP风险评估流程
        """
        print("=" * 100)
        print("ALARP（As Low As Reasonably Practicable）风险评估模型")
        print("=" * 100)

        print("\n步骤1: 加载WSM得分数据")
        print("=" * 100)
        wsm_data = self.load_wsm_scores()

        print("\n步骤2: 计算风险值")
        print("=" * 100)
        print("风险值 = 1 - WSM得分")
        print("WSM得分越高，综合表现越好，风险越低")
        risk_data = self.calculate_risk_values(wsm_data)

        print("\n步骤3: 应用ALARP分类")
        print("=" * 100)
        risk_data = self.apply_alarp_classification(risk_data)

        print("\n步骤4: 打印ALARP模型阈值说明")
        print("=" * 100)
        self.print_alarp_thresholds()

        print("\n步骤5: 打印风险评估结果")
        print("=" * 100)
        self.print_risk_assessment_results(risk_data)

        print("\n步骤6: 保存结果")
        print("=" * 100)
        self.save_results(risk_data)

        print("\n" + "=" * 100)
        print("ALARP风险评估完成！")
        print("=" * 100)

        return risk_data


def main():
    """
    主函数
    """
    assessor = ALARPRiskAssessment()

    risk_data = assessor.run_full_assessment()

    return assessor, risk_data


if __name__ == '__main__':
    assessor, risk_data = main()
