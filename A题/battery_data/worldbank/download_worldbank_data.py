"""
World Bank Data Downloader for Battery Industry Analysis
Downloads economic and trade indicators relevant to battery supply chains
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

# World Bank API endpoints
WB_API_BASE = "https://api.worldbank.org/v2"

def fetch_worldbank_indicator(indicator_code, countries, start_year=2000, end_year=2024):
    """
    Fetch World Bank indicator data via API
    
    Args:
        indicator_code: WB indicator code (e.g., 'NY.GDP.MKTP.CD')
        countries: List of country ISO codes or 'all'
        start_year, end_year: Data range
    """
    
    country_string = ';'.join(countries) if isinstance(countries, list) else countries
    
    url = f"{WB_API_BASE}/country/{country_string}/indicator/{indicator_code}"
    params = {
        'date': f'{start_year}:{end_year}',
        'format': 'json',
        'per_page': 10000,
        'page': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if len(data) > 1 and data[1]:  # API returns [metadata, data]
            records = []
            for item in data[1]:
                records.append({
                    'Country': item.get('country', {}).get('value'),
                    'Country_Code': item.get('countryiso3code'),
                    'Year': int(item.get('date', 0)),
                    'Value': item.get('value'),
                    'Indicator_Code': indicator_code
                })
            return pd.DataFrame(records)
        else:
            print(f"⚠️ No data for {indicator_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching {indicator_code}: {e}")
        return pd.DataFrame()

def download_battery_countries_data():
    """
    Download comprehensive data for major battery-producing countries
    """
    
    # Major battery supply chain countries
    countries = [
        'AUS',  # Australia - Lithium
        'CHL',  # Chile - Lithium
        'ARG',  # Argentina - Lithium
        'CHN',  # China - Processing, Graphite
        'COD',  # Congo DRC - Cobalt
        'IDN',  # Indonesia - Nickel
        'PHL',  # Philippines - Nickel
        'USA',  # USA
        'CAN',  # Canada
        'RUS',  # Russia
        'ZAF',  # South Africa - Manganese
        'BRA',  # Brazil
        'IND',  # India
        'JPN',  # Japan - Manufacturing
        'KOR',  # South Korea - Manufacturing
        'DEU',  # Germany
        'FRA',  # France
        'POL',  # Poland
        'NOR',  # Norway - EVs
        'SWE',  # Sweden
    ]
    
    # Key indicators for battery supply chain analysis
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP_Current_USD',
        'NY.GDP.PCAP.CD': 'GDP_per_capita_USD',
        'NY.GDP.MKTP.KD.ZG': 'GDP_Growth_Pct',
        'SP.POP.TOTL': 'Population',
        'NE.EXP.GNFS.ZS': 'Exports_Pct_GDP',
        'NE.IMP.GNFS.ZS': 'Imports_Pct_GDP',
        'NV.IND.MANF.ZS': 'Manufacturing_Pct_GDP',
        'EG.ELC.ACCS.ZS': 'Electricity_Access_Pct',
        'EN.ATM.CO2E.PC': 'CO2_Emissions_per_capita',
        'IC.BUS.EASE.XQ': 'Ease_Doing_Business_Score',
        'TX.VAL.MRCH.CD.WT': 'Merchandise_Exports_USD',
        'TM.VAL.MRCH.CD.WT': 'Merchandise_Imports_USD',
    }
    
    all_data = []
    
    print("Downloading World Bank data for battery supply chain countries...")
    
    for idx, (code, name) in enumerate(indicators.items(), 1):
        print(f"\n[{idx}/{len(indicators)}] {name} ({code})")
        df = fetch_worldbank_indicator(code, countries, start_year=2000, end_year=2024)
        
        if not df.empty:
            df['Indicator_Name'] = name
            all_data.append(df)
            print(f"   ✅ {len(df)} records")
        else:
            print(f"   ⚠️ No data")
        
        time.sleep(0.5)  # Rate limiting
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()

def download_global_ev_data():
    """
    Download global EV adoption and energy transition indicators
    """
    
    indicators_ev = {
        'EG.ELC.COAL.ZS': 'Electricity_from_Coal_Pct',
        'EG.FEC.RNEW.ZS': 'Renewable_Energy_Consumption_Pct',
        'EN.ATM.CO2E.KT': 'CO2_Emissions_Kilotons',
        'IS.VEH.NVEH.P3': 'Vehicles_per_1000_people',
    }
    
    # Top 30 economies
    countries = 'all'
    
    all_data = []
    
    print("\nDownloading global EV and energy indicators...")
    
    for idx, (code, name) in enumerate(indicators_ev.items(), 1):
        print(f"[{idx}/{len(indicators_ev)}] {name} ({code})")
        df = fetch_worldbank_indicator(code, countries, start_year=2010, end_year=2024)
        
        if not df.empty:
            df['Indicator_Name'] = name
            all_data.append(df)
            print(f"   ✅ {len(df)} records")
        
        time.sleep(0.5)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def download_mineral_trade_proxies():
    """
    Download trade statistics as proxies for mineral flows
    (World Bank doesn't have specific mineral trade codes, but we can use these proxies)
    """
    
    # High-tech exports and metals trade
    indicators_trade = {
        'TX.VAL.TECH.CD': 'High_Tech_Exports_USD',
        'TX.VAL.TECH.MF.ZS': 'High_Tech_Pct_Manufactured_Exports',
        'TG.VAL.TOTL.GD.ZS': 'Merchandise_Trade_Pct_GDP',
    }
    
    countries = ['CHN', 'USA', 'JPN', 'KOR', 'DEU', 'AUS', 'CHL', 'COD', 'IDN', 'CAN']
    
    all_data = []
    
    print("\nDownloading mineral trade proxy indicators...")
    
    for code, name in indicators_trade.items():
        print(f"• {name}")
        df = fetch_worldbank_indicator(code, countries, start_year=2010, end_year=2024)
        
        if not df.empty:
            df['Indicator_Name'] = name
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def pivot_wide_format(df):
    """
    Convert long format to wide format for easier analysis
    """
    if df.empty:
        return df
    
    df_wide = df.pivot_table(
        index=['Country', 'Country_Code', 'Year'],
        columns='Indicator_Name',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    return df_wide

def main():
    """Main execution"""
    print("="*70)
    print("WORLD BANK BATTERY SUPPLY CHAIN DATA DOWNLOADER")
    print("="*70)
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download battery countries data
    print("\n[TASK 1/3] Battery supply chain countries (20 countries)...")
    df_battery = download_battery_countries_data()
    
    if not df_battery.empty:
        # Save long format
        long_file = output_dir / 'worldbank_battery_countries_long_format.csv'
        df_battery.to_csv(long_file, index=False)
        print(f"\n✅ Saved: {long_file}")
        print(f"   Shape: {df_battery.shape}")
        
        # Save wide format
        df_battery_wide = pivot_wide_format(df_battery)
        wide_file = output_dir / 'worldbank_battery_countries_wide_format.csv'
        df_battery_wide.to_csv(wide_file, index=False)
        print(f"✅ Saved: {wide_file}")
        print(f"   Shape: {df_battery_wide.shape}")
    
    # Download global EV data
    print("\n[TASK 2/3] Global EV and energy indicators...")
    df_ev = download_global_ev_data()
    
    if not df_ev.empty:
        ev_file = output_dir / 'worldbank_global_ev_energy_indicators.csv'
        df_ev.to_csv(ev_file, index=False)
        print(f"\n✅ Saved: {ev_file}")
        print(f"   Shape: {df_ev.shape}")
    
    # Download trade proxies
    print("\n[TASK 3/3] Mineral trade proxies...")
    df_trade = download_mineral_trade_proxies()
    
    if not df_trade.empty:
        trade_file = output_dir / 'worldbank_mineral_trade_proxies.csv'
        df_trade.to_csv(trade_file, index=False)
        print(f"\n✅ Saved: {trade_file}")
        print(f"   Shape: {df_trade.shape}")
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Files created: 5")
    print(f"Data source: World Bank Open Data API")
    print(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    print("\n📊 Data Coverage:")
    print("  • 20 battery supply chain countries")
    print("  • 12+ economic indicators per country")
    print("  • Time range: 2000-2024")
    print("  • Global EV transition indicators")
    print("  • High-tech export proxies for mineral trade")
    print("="*70)

if __name__ == "__main__":
    main()
