"""
Materials Project Data Downloader
Downloads battery material properties from Materials Project API
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Check if mp_api is installed
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    print("⚠️ mp-api not installed or incompatible. Install with: pip install mp-api")
except Exception as e:
    MP_API_AVAILABLE = False
    print(f"⚠️ mp-api import error: {e}")

def download_cathode_materials(api_key):
    """
    Download lithium-ion battery cathode materials
    Common chemistries: LCO, NMC, LFP, NCA, LMNO
    """
    
    with MPRester(api_key) as mpr:
        print("\n[1/5] Lithium Cobalt Oxide (LCO: LiCoO2)...")
        lco = mpr.materials.summary.search(
            chemsys="Li-Co-O",
            num_elements=(3, 3),
            fields=[
                "material_id", "formula_pretty", "formula_anonymous", 
                "nelements", "elements", "composition",
                "energy_above_hull", "formation_energy_per_atom",
                "band_gap", "efermi", "is_stable", "is_gap_direct",
                "density", "volume", "nsites",
                "symmetry", "theoretical"
            ]
        )
        print(f"   Found {len(lco)} LCO materials")
        
        print("\n[2/5] Lithium Nickel Manganese Cobalt (NMC: Li-Ni-Mn-Co-O)...")
        nmc = mpr.materials.summary.search(
            chemsys="Li-Ni-Mn-Co-O",
            num_elements=(4, 5),
            fields=[
                "material_id", "formula_pretty", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "density",
                "is_stable", "symmetry"
            ]
        )
        print(f"   Found {len(nmc)} NMC materials")
        
        print("\n[3/5] Lithium Iron Phosphate (LFP: LiFePO4)...")
        lfp = mpr.materials.summary.search(
            chemsys="Li-Fe-P-O",
            num_elements=(4, 4),
            fields=[
                "material_id", "formula_pretty", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "density",
                "is_stable", "symmetry"
            ]
        )
        print(f"   Found {len(lfp)} LFP materials")
        
        print("\n[4/5] Lithium Nickel Cobalt Aluminum (NCA: Li-Ni-Co-Al-O)...")
        nca = mpr.materials.summary.search(
            chemsys="Li-Ni-Co-Al-O",
            num_elements=(4, 5),
            fields=[
                "material_id", "formula_pretty", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "density",
                "is_stable", "symmetry"
            ]
        )
        print(f"   Found {len(nca)} NCA materials")
        
        print("\n[5/5] Lithium Manganese Oxide (LMO: Li-Mn-O)...")
        lmo = mpr.materials.summary.search(
            chemsys="Li-Mn-O",
            num_elements=(3, 3),
            fields=[
                "material_id", "formula_pretty", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "density",
                "is_stable", "symmetry"
            ]
        )
        print(f"   Found {len(lmo)} LMO materials")
        
        # Combine all cathode data
        all_cathodes = []
        for materials, chemistry in [(lco, 'LCO'), (nmc, 'NMC'), (lfp, 'LFP'), 
                                     (nca, 'NCA'), (lmo, 'LMO')]:
            for mat in materials:
                mat_dict = mat.dict() if hasattr(mat, 'dict') else mat
                mat_dict['Chemistry'] = chemistry
                all_cathodes.append(mat_dict)
        
        return pd.DataFrame(all_cathodes)

def download_battery_performance(api_key):
    """
    Download battery performance data (voltage, capacity, energy density)
    """
    
    with MPRester(api_key) as mpr:
        print("\n[Battery Performance Data]")
        battery_data = mpr.battery.search(
            working_ion="Li",
            fields=[
                "battery_id", "framework", "working_ion",
                "average_voltage", "max_voltage", "min_voltage",
                "capacity_grav", "capacity_vol",
                "energy_grav", "energy_vol",
                "max_voltage_step", "fracA_charge", "fracA_discharge"
            ]
        )
        print(f"   Found {len(battery_data)} battery records")
        
        return pd.DataFrame([b.dict() if hasattr(b, 'dict') else b for b in battery_data])

def download_anode_materials(api_key):
    """
    Download anode materials (Graphite, Silicon, Lithium titanate)
    """
    
    with MPRester(api_key) as mpr:
        print("\n[Anode Materials]")
        
        # Graphite
        print("  • Graphite structures...")
        graphite = mpr.materials.summary.search(
            formula="C",
            fields=["material_id", "formula_pretty", "energy_above_hull", 
                   "density", "symmetry", "is_stable"]
        )
        print(f"    Found {len(graphite)} graphite structures")
        
        # Silicon
        print("  • Silicon structures...")
        silicon = mpr.materials.summary.search(
            formula="Si",
            fields=["material_id", "formula_pretty", "energy_above_hull",
                   "density", "symmetry", "is_stable"]
        )
        print(f"    Found {len(silicon)} silicon structures")
        
        # Lithium titanate
        print("  • Lithium titanate (Li-Ti-O)...")
        lto = mpr.materials.summary.search(
            chemsys="Li-Ti-O",
            num_elements=(3, 3),
            fields=["material_id", "formula_pretty", "energy_above_hull",
                   "density", "symmetry", "is_stable"]
        )
        print(f"    Found {len(lto)} LTO structures")
        
        # Combine
        anodes = []
        for materials, anode_type in [(graphite, 'Graphite'), (silicon, 'Silicon'), (lto, 'LTO')]:
            for mat in materials:
                mat_dict = mat.dict() if hasattr(mat, 'dict') else mat
                mat_dict['Anode_Type'] = anode_type
                anodes.append(mat_dict)
        
        return pd.DataFrame(anodes)

def main():
    """
    Main execution - requires API key
    """
    
    print("="*70)
    print("MATERIALS PROJECT BATTERY DATA DOWNLOADER")
    print("="*70)
    
    if not MP_API_AVAILABLE:
        print("\n❌ ERROR: mp-api not installed")
        print("\nInstall with: pip install mp-api")
        print("Then get API key from: https://materialsproject.org/api")
        return
    
    # Check for API key
    api_key = input("\nEnter your Materials Project API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\n⚠️ No API key provided. Using demo mode (limited data)...")
        print("\nTo download real data:")
        print("1. Register at https://materialsproject.org")
        print("2. Get API key from https://materialsproject.org/api")
        print("3. Run: python download_materials_project_data.py")
        
        # Create demo summary file
        output_dir = Path(__file__).parent
        demo_file = output_dir / 'materials_project_data_summary.txt'
        with open(demo_file, 'w') as f:
            f.write("Materials Project Data Summary\n")
            f.write("="*50 + "\n\n")
            f.write("Status: API key required\n")
            f.write("Data available with valid API key:\n")
            f.write("  • Cathode materials: LCO, NMC, LFP, NCA, LMO\n")
            f.write("  • Anode materials: Graphite, Silicon, LTO\n")
            f.write("  • Battery performance: voltage, capacity, energy\n")
            f.write("  • Material properties: band gap, density, stability\n")
            f.write("\nExpected fields: ~500+ materials\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        print(f"\n✅ Created: {demo_file}")
        return
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download cathode materials
        print("\n[TASK 1/3] Cathode materials (LCO, NMC, LFP, NCA, LMO)...")
        df_cathode = download_cathode_materials(api_key)
        cathode_file = output_dir / 'materials_project_cathode_materials.csv'
        df_cathode.to_csv(cathode_file, index=False)
        print(f"\n✅ Saved: {cathode_file}")
        print(f"   Shape: {df_cathode.shape}")
        
        # Download battery performance
        print("\n[TASK 2/3] Battery performance data...")
        df_battery = download_battery_performance(api_key)
        battery_file = output_dir / 'materials_project_battery_performance.csv'
        df_battery.to_csv(battery_file, index=False)
        print(f"\n✅ Saved: {battery_file}")
        print(f"   Shape: {df_battery.shape}")
        
        # Download anode materials
        print("\n[TASK 3/3] Anode materials (Graphite, Silicon, LTO)...")
        df_anode = download_anode_materials(api_key)
        anode_file = output_dir / 'materials_project_anode_materials.csv'
        df_anode.to_csv(anode_file, index=False)
        print(f"\n✅ Saved: {anode_file}")
        print(f"   Shape: {df_anode.shape}")
        
        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Total files: 3")
        print(f"Total materials: {len(df_cathode) + len(df_anode)}")
        print(f"Battery records: {len(df_battery)}")
        print(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check API key is valid")
        print("2. Check internet connection")
        print("3. Try: pip install --upgrade mp-api")

if __name__ == "__main__":
    main()