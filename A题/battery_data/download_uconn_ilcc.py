#!/usr/bin/env python3
"""
Download UConn-ILCC Battery Dataset
Critical for MCM A-Problem Task 3.3 (Usage Pattern Fluctuations)

Data Source: https://digitalcommons.lib.uconn.edu/reil_datasets/2/
License: CC-BY 4.0
Citation: University of Connecticut, Renewable Energy & Island Lab (2024)

Dataset: 44 Panasonic NMC cells, 9 SOC levels (10%-90%), 1Hz sampling
Use Case: Fluctuation modeling within scenarios
"""

import requests
from pathlib import Path
import time

def download_uconn_ilcc():
    """
    Download UConn-ILCC battery dataset (impedance + degradation data).
    
    Dataset characteristics:
    - 44 cells tested
    - 9 SOC levels: 10%, 20%, ..., 90%
    - 1Hz sampling rate
    - Voltage, current, temperature measurements
    - Impedance spectroscopy data
    
    Perfect for modeling fluctuations in Task 3.3!
    """
    
    base_url = "https://digitalcommons.lib.uconn.edu"
    dataset_path = "/context/reil_datasets/article/1003/type/native/viewcontent"
    
    output_dir = Path(__file__).parent / "uconn_ilcc"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("UConn-ILCC Battery Dataset Downloader")
    print("=" * 70)
    
    # The dataset is distributed as a single ZIP file
    # URL structure from Digital Commons platform
    zip_url = f"{base_url}{dataset_path}"
    
    print(f"\n📥 Attempting download from: {zip_url}")
    print("⏳ This may take a few minutes (dataset is large)...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(zip_url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        
        # Save ZIP file
        zip_file = output_dir / "uconn_ilcc_dataset.zip"
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end='')
        
        print(f"\n✅ Downloaded: {zip_file}")
        print(f"📦 Size: {downloaded / 1024 / 1024:.1f} MB")
        
        # Extract ZIP
        print("\n📂 Extracting files...")
        import zipfile
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"✅ Extracted to: {output_dir}")
        
        # List extracted files
        files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.txt")) + list(output_dir.glob("*.mat"))
        print(f"\n📁 Extracted {len(files)} files:")
        for f in files[:10]:  # Show first 10
            print(f"   - {f.name}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")
        
        return output_dir
        
    except requests.exceptions.Timeout:
        print("\n❌ Download timed out")
        print("\n📝 MANUAL DOWNLOAD REQUIRED:")
        print("   1. Visit: https://digitalcommons.lib.uconn.edu/reil_datasets/2/")
        print("   2. Click 'Download' button")
        print("   3. Save ZIP to: battery_data/uconn_ilcc/")
        print("   4. Extract manually")
        return None
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📝 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("   1. Visit: https://digitalcommons.lib.uconn.edu/reil_datasets/2/")
        print("   2. Title: 'Panasonic NMC 18650PF Li-ion Battery Impedance Data'")
        print("   3. Click 'Download' to get full dataset")
        print("   4. Extract to: battery_data/uconn_ilcc/")
        print("\n💡 Alternative direct link:")
        print("   https://digitalcommons.lib.uconn.edu/context/reil_datasets/article/1003/type/native/viewcontent")
        return None

def create_readme():
    """Create README for UConn-ILCC dataset usage"""
    
    readme_content = """# UConn-ILCC Battery Dataset

## Overview
- **Source**: University of Connecticut, Renewable Energy & Island Lab
- **License**: CC-BY 4.0
- **URL**: https://digitalcommons.lib.uconn.edu/reil_datasets/2/

## Dataset Characteristics
- **Cells Tested**: 44 Panasonic NMC 18650PF cells
- **SOC Levels**: 9 levels (10%, 20%, ..., 90%)
- **Sampling Rate**: 1 Hz
- **Measurements**: Voltage, Current, Temperature, Impedance

## MCM A-Problem Usage

### Task 3.3: Fluctuations in Usage Patterns
This dataset is PERFECT for modeling within-scenario fluctuations:

1. **SOC-dependent behavior**: 
   - Extract voltage/current curves at different SOC levels
   - Model how power draw varies within each scenario
   
2. **Temperature effects**:
   - Use temperature measurements to validate f_temp(T) model
   - Lines 261, 885 in strategic deployment
   
3. **Stochastic modeling**:
   - Use 1Hz sampling to characterize noise/fluctuation parameters
   - Ornstein-Uhlenbeck process calibration (Line 393-396)

## Data Structure
Typical files:
- `cell_XX_SOC_YY_impedance.csv`: Impedance data
- `cell_XX_SOC_YY_timeseries.csv`: Voltage/current time series
- `metadata.txt`: Experimental conditions

## Processing Steps
1. Load time-series data for each cell
2. Group by SOC level (10%-90%)
3. Extract statistical properties:
   - Mean voltage at each SOC
   - Standard deviation (fluctuation parameter σ)
   - Auto-correlation (θ parameter for OU process)
4. Build lookup table: SOC → (μ, σ, θ)

## Citation
```
University of Connecticut, Renewable Energy & Island Lab. (2024). 
Panasonic NMC 18650PF Li-ion Battery Impedance Data. 
Digital Commons @ UConn. CC-BY 4.0.
```
"""
    
    readme_file = Path(__file__).parent / "uconn_ilcc" / "README.md"
    readme_file.parent.mkdir(exist_ok=True)
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created: {readme_file}")

if __name__ == "__main__":
    result = download_uconn_ilcc()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ SUCCESS: UConn-ILCC dataset ready")
        print("=" * 70)
        print("\n📋 MCM Application:")
        print("   - Task 3.3: Fluctuation modeling (Lines 379-425)")
        print("   - Parameter calibration: σ, θ for OU process")
        print("   - Temperature validation: f_temp(T) model")
        
        create_readme()
    else:
        print("\n⚠️ Manual download required - see instructions above")
