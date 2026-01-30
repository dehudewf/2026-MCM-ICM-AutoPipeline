# MCM 2026 Competition-Critical Data Acquisition Plan
# 竞赛关键数据收集计划

> **Created**: 2026-01-30  
> **Purpose**: Map strategic deployment document to actionable data collection  
> **Priority**: COMPETITION-WINNING ONLY (只收集对比赛有直接助力的数据)

---

## 🎯 Strategic Document → Data Mapping

### **Task 1: Continuous-Time Model** 🔴 MUST

| Strategic Need | Required Data | Priority | Collection Method | Status |
|---------------|---------------|----------|-------------------|--------|
| **Power Decomposition (5 factors)** | Screen/CPU/Network/GPS/Background power consumption | 🔴 CRITICAL | GSMArena + Academic papers | ⚠️ Partial |
| **Parameter Validation (Table 8.2)** | P_screen, P_CPU, P_GPU, P_4G, P_GPS ranges | 🔴 CRITICAL | Literature + OEM specs | ⬜ Missing |
| **Temperature Factor f_temp(T)** | Battery efficiency vs temperature data | 🔴 CRITICAL | NASA/CALCE cycling data | ✅ Have |
| **Aging Factor f_aging** | Capacity fade vs cycle count | 🔴 CRITICAL | NASA/Oxford datasets | ✅ Have |
| **Q_nominal values** | Battery capacities for validation | 🔴 CRITICAL | GSMArena + Apple specs | ✅ Have |

### **Task 2: TTE Predictions** 🔴 MUST

| Strategic Need | Required Data | Priority | Collection Method | Status |
|---------------|---------------|----------|-------------------|--------|
| **Observed TTE Benchmarks (T2.7)** | iPhone 15: 20h video, 16h streaming | 🔴 CRITICAL | Apple official specs | ✅ Have |
| **GSMArena Endurance Rating** | 95h composite endurance average | 🔴 CRITICAL | GSMArena database | ⚠️ Only 1 phone |
| **Real-world Usage Patterns** | Actual user behavior distributions | 🟡 IMPORTANT | MAGGIE/Android datasets | ⬜ Missing |
| **Gaming TTE Range** | 2.5-4h typical gaming duration | 🟡 IMPORTANT | Academic papers | ⬜ Missing |

### **Task 3: Sensitivity Analysis** 🔴 MUST

| Strategic Need | Required Data | Priority | Collection Method | Status |
|---------------|---------------|----------|-------------------|--------|
| **Parameter Sensitivity Ranges** | ±10% variations for all P_i | 🔴 CRITICAL | Model + Literature bounds | ⬜ Missing |
| **Usage Fluctuation Data** | Stochastic variation within scenarios | 🔴 CRITICAL | Real device traces | ⬜ Missing |
| **Bluetooth/WiFi Idle Power** | "Surprisingly little" impact validation | 🟡 IMPORTANT | Power measurement papers | ⬜ Missing |

### **Task 4: Recommendations** 🔴 MUST

| Strategic Need | Required Data | Priority | Collection Method | Status |
|---------------|---------------|----------|-------------------|--------|
| **Triple Baseline Comparison** | Default vs Power Saver vs Model-based | 🟢 NICE-TO-HAVE | Simulate from model | N/A |
| **Cross-Device Parameters** | Smartwatch/Tablet/Laptop specs | 🟡 IMPORTANT | OEM specifications | ⬜ Missing |

---

## 📦 Priority 1: CRITICAL MISSING DATA (Must Collect)

### 1. **Power Consumption Literature Database** 🔴

**Why Critical**: Table 8.2 parameter validation requires 2-3 published sources per parameter.

**Target Sources**:
```yaml
search_queries:
  google_scholar:
    - "smartphone power consumption measurement"
    - "mobile device energy profiling"
    - "Android power model validation"
    - "iPhone battery power breakdown"
  
  target_papers:
    - Carroll & Heiser (2010): "An Analysis of Power Consumption in a Smartphone"
    - Perrucci et al. (2011): "Energy Consumption of Mobile Multimedia"
    - Banerjee et al. (2016): "Cross-Layer Power Analysis"
    - Ma et al. (2013): "eDoctor: Automatically Diagnosing Abnormal Battery Drain"

  extraction_targets:
    - P_screen: [0.3W, 5W] range with citations
    - P_CPU: [0.1W, 5W] range at various loads
    - P_GPU: [0.1W, 3W] range idle→gaming
    - P_4G: [0.8W, 1.5W] active transmission
    - P_WiFi: [0.02W, 0.3W] idle→active
    - P_GPS: [0.3W, 0.8W] continuous tracking
    - P_Bluetooth: [0.01W, 0.1W] BLE idle→active
```

**Collection Method**:
```bash
# Use arXiv MCP server
# Search Google Scholar for citations
# Extract power values into structured CSV
```

**Output Format**:
```csv
Component,Power_W,Load_Condition,Device,Source,DOI,Year
Screen,1.2,50% brightness,Generic smartphone,Carroll2010,10.1145/xxxx,2010
CPU,0.4,20% load,Nexus One,Carroll2010,10.1145/xxxx,2010
...
```

---

### 2. **Real-World Usage Traces** 🔴

**Why Critical**: T3.3 requires modeling "fluctuations in usage patterns" within scenarios.

**NEW DATASETS NEEDED**:

| Dataset | URL | Key Features | Competition Use |
|---------|-----|--------------|-----------------|
| **PhoneLab** | http://www.phone-lab.org | Real user traces, 200+ users | Usage fluctuation statistics |
| **Device Analyzer (Cambridge)** | https://deviceanalyzer.cl.cam.ac.uk | 17,000+ devices, multi-year | Background app behavior |
| **LiveLab Dataset** | Rice University archive | Detailed power traces | Power validation |

**Collection Plan**:
```python
# Priority order (try in sequence until success)
datasets_to_collect = [
    {
        "name": "PhoneLab",
        "url": "http://datasets.phone-lab.org",
        "method": "direct_download",
        "size": "~2GB",
        "extraction": "CPU/battery traces"
    },
    {
        "name": "Device Analyzer",
        "url": "https://deviceanalyzer.cl.cam.ac.uk/",
        "method": "registration_required",
        "size": "~100GB (subset needed)",
        "extraction": "App usage patterns"
    }
]
```

---

### 3. **Gaming Power Consumption Benchmark** 🟡

**Why Important**: S3 Gaming scenario + "surprisingly little" validation.

**Target Data**:
- 3D gaming power draw: 5-8W typical
- GPU load: 80-95% sustained
- TTE range: 2.5-4h (validation target)

**Collection Method**:
```yaml
sources:
  gsmarena_gaming_tests:
    method: "MCP Playwright scrape"
    phones_needed: 10
    metrics: ["Gaming endurance (h)", "Active use score"]
  
  academic_papers:
    - "Mobile Gaming Energy Consumption" studies
    - Qualcomm/ARM GPU power specifications
```

---

## 📦 Priority 2: NICE-TO-HAVE DATA (Time Permitting)

### 4. **Cross-Device Generalization Data** 🟢

**Strategic Document Reference**: Section 6.6 - Cross-Device Generalization Framework

**Needed Specs**:
```yaml
devices:
  smartwatch:
    screen_power: 0.1W
    battery_capacity: 300mAh
    source: "Apple Watch specs"
  
  tablet:
    screen_power: 4W
    battery_capacity: 10000mAh
    source: "iPad specs"
  
  laptop:
    screen_power: 8W
    battery_capacity: 60Wh
    source: "MacBook specs"
```

---

## 🚀 Actionable Collection Scripts

### Script 1: Academic Paper Power Database

```python
#!/usr/bin/env python3
"""
Collect power consumption values from academic literature.
Output: power_literature_database.csv
"""

import pandas as pd

# Manual extraction template (to be filled)
power_data = {
    'Component': [],
    'Power_W': [],
    'Load_Condition': [],
    'Device': [],
    'Source': [],
    'DOI': [],
    'Year': []
}

# Target papers to extract
papers_to_review = [
    "Carroll & Heiser 2010 - Power Consumption in Smartphone",
    "Perrucci et al 2011 - Energy Consumption Mobile Multimedia",
    "Ma et al 2013 - eDoctor Battery Drain",
    "Banerjee et al 2016 - Cross-Layer Power Analysis"
]

# TODO: Manual extraction from PDFs
# Save to: battery_data/literature/power_literature_database.csv
```

### Script 2: GSMArena Expansion (MCP)

```python
#!/usr/bin/env python3
"""
Expand GSMArena dataset from 1 phone to 20+ phones.
Priority phones: iPhone 15 series, Galaxy S24, Pixel 8 Pro
"""

target_phones = [
    "iPhone 15 Pro Max",
    "iPhone 15 Pro",
    "iPhone 15",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
    "Google Pixel 8 Pro",
    "OnePlus 12",
    "Xiaomi 14 Pro",
    # ... expand to 20 total
]

# Use MCP Playwright with search-result navigation
# Bypass Cloudflare via search results
# Extract: Battery capacity, Active use score, Gaming time, Video time
```

### Script 3: PhoneLab/Device Analyzer Download

```bash
#!/bin/bash
# Download real-world usage traces

# PhoneLab Dataset
wget http://datasets.phone-lab.org/battery_traces.tar.gz
tar -xzf battery_traces.tar.gz -C battery_data/usage_traces/

# Extract CPU usage, app switching, background activity
python extract_usage_patterns.py
```

---

## 📊 Current Status Summary

### ✅ **COMPLETE** (6/8 datasets)
- NASA Battery (aging, temperature factors)
- Oxford Battery (degradation curves)
- Apple Specs (Q_nominal, design cycles)
- USGS Minerals (context only)
- World Bank (context only)
- Materials Project (API scripts ready)

### ⚠️ **PARTIAL** (1/8 datasets)
- GSMArena: Only 1 phone → Need 20+ phones

### ⬜ **MISSING CRITICAL** (Competition Blockers)
1. **Power Consumption Literature Database** → Parameter validation
2. **Real-World Usage Traces** → Fluctuation modeling
3. **Gaming Benchmark Data** → S3 scenario validation
4. **Parameter Sensitivity Ranges** → Table 8.2 bounds

---

## 🎯 Recommended Collection Order

```
Day 1 (Now):
  1. Expand GSMArena to 20 phones (MCP scraping)
     → Enables TTE validation (T2.7)
  
  2. Download PhoneLab dataset OR Device Analyzer subset
     → Enables fluctuation modeling (T3.3)

Day 2:
  3. Build power literature database (manual PDF extraction)
     → Enables parameter validation (Table 8.2)
  
  4. Collect gaming benchmark data (GSMArena + papers)
     → Enables S3 scenario + "surprisingly little" analysis

Day 3 (if time):
  5. Cross-device specs (smartwatch/tablet/laptop)
     → Enables Section 6.6 generalization framework
```

---

## 🔍 Quality Checklist

Before proceeding with collection, verify:
- [ ] Data is **free and open** (CC-BY/CC0/MIT/Public Domain)
- [ ] Data has **public URL** (no email walls)
- [ ] Data is **competition-relevant** (directly supports strategic doc tasks)
- [ ] Data is **complete** (≥95% non-null, as per user memory)
- [ ] Data has **citation-ready metadata** (author, year, DOI)

---

**END OF PLAN**

> 📝 **Next Action**: Execute Priority 1 collections (GSMArena expansion + Usage traces)  
> 🎯 **Goal**: Enable parameter validation + fluctuation modeling for T1/T3  
> ⏱️ **Estimated Time**: 4-6 hours for Priority 1 completion
