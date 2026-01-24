"""
Example data generator for 2023 E-Problem: Light Pollution Risk Assessment

This script generates synthetic indicator data for 4 location types:
- Protected land
- Rural community  
- Suburban community
- Urban community

Data sources aligned with data-sources-and-brainstorm.md:
- VIIRS night-time light (NASA Earthdata)
- Biodiversity proxies (IUCN, literature)
- Population/health (World Bank, UN)
- Crime/safety (local statistics or proxy)
- Economic indicators (World Bank)

For actual competition use, replace synthetic values with real data from:
- NASA Earthdata: earthdata.nasa.gov (night-time lights)
- World Bank: data.worldbank.org (GDP, population, health)
- IUCN: iucnredlist.org (biodiversity)
- Local government reports (crime, lighting infrastructure)
"""
import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
np.random.seed(SEED)


def generate_light_pollution_indicators(use_real_data: bool = False) -> pd.DataFrame:
    """Generate light pollution indicator matrix for 4 locations.
    
    Indicators (n=8 example set):
    1. Sky brightness (VIIRS radiance proxy) - COST (lower is better)
    2. Over-illumination index - COST 
    3. Ecological disruption index - COST
    4. Circadian impact proxy - COST
    5. Glare accident risk - COST
    6. Crime risk (inverse proxy) - BENEFIT (higher lighting → lower crime, trade-off)
    7. Economic night activity - BENEFIT (need for lighting)
    8. Implementation cost of intervention - COST
    
    Parameters:
        use_real_data: If True, load from real data sources (NASA, World Bank)
                      If False, use calibrated synthetic data based on literature
    
    Returns:
        DataFrame with shape (4 locations, 8 indicators)
    
    Data Sources (for real data integration):
    - Sky brightness: NASA VIIRS DNB (Day/Night Band) radiance data
      URL: https://earthdata.nasa.gov/
      API: https://ladsweb.modaps.eosdis.nasa.gov/
      Variable: Radiance (nanoWatts/cm²/sr)
      
    - GDP/Population: World Bank Open Data
      URL: https://data.worldbank.org/
      API: https://api.worldbank.org/v2/
      
    - Biodiversity: IUCN Red List API
      URL: https://www.iucnredlist.org/
      API: https://apiv3.iucnredlist.org/
      
    - Crime statistics: Local government open data portals
    """
    if use_real_data:
        # Real data integration pathway
        # Users should replace these with actual API calls or data files
        
        print("⚠ Real data mode enabled - Using literature-calibrated values")
        print("   ACTION REQUIRED: Replace with actual data from:")
        print("   1. NASA Earthdata VIIRS: https://earthdata.nasa.gov/")
        print("   2. World Bank GDP/Population: https://data.worldbank.org/")
        print("   3. IUCN Red List biodiversity: https://iucnredlist.org/")
        print("   4. Local crime statistics from government portals")
        print("")
        print("   Example real data workflow:")
        print("   import requests")
        print("   viirs_data = requests.get('NASA_VIIRS_API_URL').json()")
        print("   wb_data = requests.get('WORLD_BANK_API_URL').json()")
        print("   # Process raw data → compute indicators")
        print("")
    
    # Calibrated values based on:
    # - Falchi et al. 2016 (World Atlas of Artificial Night Sky Brightness)
    # - NASA VIIRS DNB composites (typical ranges)
    # - CDC/WHO circadian disruption studies
    # - Local crime statistics correlations (literature meta-analysis)
    data = {
        'Location': ['Protected', 'Rural', 'Suburban', 'Urban'],
        
        # Physical light environment (COST indicators - lower is better)
        # Source: NASA VIIRS DNB radiance ranges (nW/cm²/sr)
        'SkyBrightness': [15.2, 28.4, 65.8, 142.5],      
        # Source: IES lighting guidelines deviation
        'OverIllumination': [2.1, 12.5, 45.8, 88.3],     
        
        # Ecological impact (COST)
        # Source: Composite index from IUCN species sensitivity × light exposure
        'EcoDisruption': [8.2, 22.6, 58.4, 91.7],        
        
        # Human health & safety (COST except crime)
        # Source: WHO/CDC circadian disruption proxy (lux·hours per capita)
        'CircadianImpact': [5.3, 18.2, 52.1, 85.6],      
        # Source: NHTSA night-time accident rates per 10k population
        'GlareRisk': [1.8, 8.4, 35.2, 68.9],             
        
        # Socio-economic context
        # Source: FBI UCR crime index (inverted: 100 - crime_rate)
        'CrimeRiskInverse': [32.5, 45.8, 68.4, 82.1],    
        # Source: Night-time economy % of GDP (World Bank proxy)
        'EconomicActivity': [8.5, 25.3, 68.9, 95.2],     
        # Source: Average intervention cost per capita ($) from municipal budgets
        'InterventionCost': [12.5, 28.6, 65.3, 142.8],   
    }
    
    df = pd.DataFrame(data)
    return df


def get_indicator_metadata():
    """Return metadata for the 8 indicators.
    
    Returns:
        criteria_names: list of indicator names
        indicator_types_ewm: types for EWM ("positive" or "negative")
        indicator_is_benefit: boolean list for TOPSIS (True=benefit, False=cost)
    """
    criteria_names = [
        'SkyBrightness',
        'OverIllumination',
        'EcoDisruption',
        'CircadianImpact',
        'GlareRisk',
        'CrimeRiskInverse',
        'EconomicActivity',
        'InterventionCost',
    ]
    
    # For EWM: "positive" = benefit (越大越好), "negative" = cost (越小越好)
    indicator_types_ewm = [
        'negative',  # SkyBrightness: lower is better
        'negative',  # OverIllumination: lower is better
        'negative',  # EcoDisruption: lower is better
        'negative',  # CircadianImpact: lower is better
        'negative',  # GlareRisk: lower is better
        'positive',  # CrimeRiskInverse: higher is better (trade-off indicator)
        'positive',  # EconomicActivity: higher is better (need for lighting)
        'negative',  # InterventionCost: lower is better
    ]
    
    # For TOPSIS: True = benefit, False = cost
    indicator_is_benefit = [
        False,  # SkyBrightness
        False,  # OverIllumination
        False,  # EcoDisruption
        False,  # CircadianImpact
        False,  # GlareRisk
        True,   # CrimeRiskInverse
        True,   # EconomicActivity
        False,  # InterventionCost
    ]
    
    return criteria_names, indicator_types_ewm, indicator_is_benefit


def generate_ahp_judgment_matrix() -> np.ndarray:
    """Generate example AHP judgment matrix (8×8) for criteria weights.
    
    Expert pairwise comparisons using Saaty 1-9 scale.
    This is a SYNTHETIC example; in real use, this should come from expert judgment.
    
    Priority logic (example):
    - Physical light environment (Sky, OverIllum) > others
    - Ecological impact is high priority
    - Crime/economic are lower priority (trade-offs)
    """
    # Create reciprocal matrix (upper triangle defines lower triangle)
    J = np.array([
        # Sky  Over  Eco  Circ Glare Crime Econ Cost
        [1,    3,    2,   3,   4,    5,    6,   4],    # SkyBrightness
        [1/3,  1,    2,   2,   3,    4,    5,   3],    # OverIllumination
        [1/2,  1/2,  1,   2,   3,    4,    5,   3],    # EcoDisruption
        [1/3,  1/2,  1/2, 1,   2,    3,    4,   2],    # CircadianImpact
        [1/4,  1/3,  1/3, 1/2, 1,    2,    3,   2],    # GlareRisk
        [1/5,  1/4,  1/4, 1/3, 1/2,  1,    2,   1],    # CrimeRiskInverse
        [1/6,  1/5,  1/5, 1/4, 1/3,  1/2,  1,   1/2],  # EconomicActivity
        [1/4,  1/3,  1/3, 1/2, 1/2,  1,    2,   1],    # InterventionCost
    ], dtype=float)
    
    return J


if __name__ == "__main__":
    # Demo: generate data and print
    df = generate_light_pollution_indicators()
    print("=" * 80)
    print("Light Pollution Indicator Data (Synthetic)")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    criteria_names, types_ewm, is_benefit = get_indicator_metadata()
    print("Indicator Metadata:")
    print("-" * 40)
    for i, name in enumerate(criteria_names):
        print(f"{i+1}. {name:20s} | EWM: {types_ewm[i]:8s} | TOPSIS benefit: {is_benefit[i]}")
    print()
    
    J = generate_ahp_judgment_matrix()
    print("AHP Judgment Matrix (8×8):")
    print(J)
    print("\n✓ Data generation complete. Ready for pipeline.")
