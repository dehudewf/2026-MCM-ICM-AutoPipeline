
from mp_api.client import MPRester

# 替换为你的API Key
API_KEY = "YOUR_API_KEY_HERE"

def query_battery_materials():
    """查询电池材料数据"""
    
    with MPRester(API_KEY) as mpr:
        # 1. 查询锂离子电池正极材料
        cathode_materials = mpr.materials.summary.search(
            elements=["Li", "Co", "O"],
            num_elements=(3, 4),
            fields=["material_id", "formula_pretty", "energy_above_hull", 
                    "band_gap", "formation_energy_per_atom"]
        )
        
        print(f"找到 {len(cathode_materials)} 种正极材料")
        
        # 2. 查询电池电压数据
        battery_data = mpr.battery.search(
            working_ion="Li",
            fields=["battery_id", "average_voltage", "capacity_grav", 
                    "energy_grav", "max_voltage_step"]
        )
        
        print(f"找到 {len(battery_data)} 条电池数据")
        
        return cathode_materials, battery_data

# 运行查询
cathode_materials, battery_data = query_battery_materials()

# 保存数据
import pandas as pd
pd.DataFrame(cathode_materials).to_csv("cathode_materials.csv", index=False)
pd.DataFrame(battery_data).to_csv("battery_data.csv", index=False)
