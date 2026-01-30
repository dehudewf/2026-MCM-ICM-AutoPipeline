#!/usr/bin/env python3
"""
Download EPA Fuel Economy Dataset - EV Battery Data
Supports MCM A-Problem Task 2.7 (Observed Behavior Validation)

Data Source: https://www.fueleconomy.gov/feg/download.shtml
License: Public Domain (US Government)
"""

import pandas as pd
import requests
from pathlib import Path

def download_epa_vehicles():
    """
    Download EPA vehicles dataset and extract EV-specific data.
    
    Expected columns:
    - year: Model year
    - make: Manufacturer
    - model: Model name
    - evMotor: Electric motor info
    - phevBlended: PHEV flag
    - city08: City MPGe (miles per gallon equivalent)
    - highway08: Highway MPGe
    - comb08: Combined MPGe
    - range: Electric range (miles)
    - rangeCity: City range
    - rangeHwy: Highway range
    - charge240: Time to charge on 240V (hours)
    """
    
    url = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
    print(f"📥 Downloading EPA vehicles dataset from: {url}")
    
    try:
        # Download with timeout
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save raw data
        output_dir = Path(__file__).parent
        raw_file = output_dir / "epa_vehicles_raw.csv"
        
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded {len(response.content) / 1024 / 1024:.1f} MB")
        
        # Load and filter
        df = pd.read_csv(raw_file, low_memory=False)
        print(f"📊 Total vehicles in dataset: {len(df):,}")
        
        # Filter for EVs (Battery Electric Vehicles)
        # Key: fuelType1 == 'Electricity' and NOT a PHEV
        evs = df[
            (df['fuelType1'] == 'Electricity') & 
            (df['phevBlended'] == False)
        ].copy()
        
        print(f"🔋 Battery EVs found: {len(evs):,}")
        
        # Select relevant columns
        columns_to_keep = [
            'year', 'make', 'model', 'evMotor',
            'city08', 'highway08', 'comb08',  # MPGe efficiency
            'range', 'rangeCity', 'rangeHwy',  # Electric range
            'charge240',  # Charging time
            'youSaveSpend',  # Cost savings
        ]
        
        # Add columns if they exist
        available_cols = [col for col in columns_to_keep if col in evs.columns]
        evs_clean = evs[available_cols].copy()
        
        # Estimate battery capacity from range and efficiency
        # Assumption: 1 kWh = 33.7 kWh/gallon equivalent
        # Battery_kWh ≈ (Range_miles × 33.7) / (MPGe × 1000) × 1000
        # Simplified: Battery_kWh ≈ Range_miles / (MPGe / 33.7)
        
        if 'range' in evs_clean.columns and 'comb08' in evs_clean.columns:
            evs_clean['Battery_kWh_estimated'] = (
                evs_clean['range'] * 33.7 / evs_clean['comb08']
            ).round(1)
        
        # Sort by year (most recent first)
        evs_clean = evs_clean.sort_values('year', ascending=False)
        
        # Save filtered EV data
        ev_file = output_dir / "epa_ev_battery_data.csv"
        evs_clean.to_csv(ev_file, index=False)
        
        print(f"\n✅ Saved EV data: {ev_file}")
        print(f"📊 Records: {len(evs_clean):,}")
        print(f"📅 Year range: {evs_clean['year'].min()}-{evs_clean['year'].max()}")
        
        # Statistics
        print("\n📈 Dataset Statistics:")
        print(f"   Manufacturers: {evs_clean['make'].nunique()}")
        print(f"   Models: {evs_clean['model'].nunique()}")
        
        if 'Battery_kWh_estimated' in evs_clean.columns:
            valid_battery = evs_clean[evs_clean['Battery_kWh_estimated'] > 0]
            print(f"   Battery capacity range: {valid_battery['Battery_kWh_estimated'].min():.1f} - {valid_battery['Battery_kWh_estimated'].max():.1f} kWh")
            print(f"   Average battery: {valid_battery['Battery_kWh_estimated'].mean():.1f} kWh")
        
        if 'range' in evs_clean.columns:
            print(f"   Range: {evs_clean['range'].min():.0f} - {evs_clean['range'].max():.0f} miles")
            print(f"   Average range: {evs_clean['range'].mean():.0f} miles")
        
        # Sample recent EVs for MCM validation
        print("\n🔍 Sample Recent EVs (for validation):")
        recent = evs_clean[evs_clean['year'] >= 2023].head(10)
        print(recent[['year', 'make', 'model', 'range', 'Battery_kWh_estimated']].to_string(index=False))
        
        return ev_file
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        print("\n📝 Manual download instructions:")
        print("   1. Visit: https://www.fueleconomy.gov/feg/download.shtml")
        print("   2. Download 'vehicles.csv'")
        print("   3. Place in: battery_data/")
        print("   4. Run this script again")
        return None
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("EPA Fuel Economy Dataset Downloader")
    print("MCM A-Problem: Battery Modeling Parameter Validation")
    print("=" * 70)
    
    result = download_epa_vehicles()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ SUCCESS: EPA EV data ready for MCM modeling")
        print("=" * 70)
        print("\n📋 Next Steps:")
        print("   1. Use for Task 2.7: Compare predictions to observed behavior")
        print("   2. Extract battery capacity ranges for parameter bounds")
        print("   3. Validate TTE predictions against EPA range data")
    else:
        print("\n⚠️ Download incomplete - see instructions above")
