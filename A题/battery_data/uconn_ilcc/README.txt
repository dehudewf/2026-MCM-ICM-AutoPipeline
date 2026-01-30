**********************************************************************
UConn-ILCC NMC/Gr Battery Aging Dataset (c) 2025 by Nowacki et al. 
**********************************************************************

TABLE OF CONTENTS
— OVERVIEW OF CELL TESTING
— CONTENTS OF ZIP FILE(S)
— IMPLEMENTATION DETAILS
— CITATION
— LICENSE AGREEMENT


**********************************************************************
OVERVIEW OF CELL TESTING

This dataset contains battery aging data from 44 nickel-manganese-cobalt/graphite (NMC/Gr) cells. The Panasonic 18650 cells (UR18650AA) have a nominal capacity of 2.25 Ah and are aged until 65% state of health (SOH), approximately 1.46 Ah. 

Aging consists of constant-current constant-voltage (CC-CV) cycling with a reference performance test (RPT) carried out every 3.5 days. The RPT contains a sequence of CC pulses aimed at measuring the internal resistance at different states of charge (SOC). It also performs a low C-rate (C/3) CC-CV charge and discharge cycle to measure the cell's current capacity. All cycling tests and RPTs are performed in a temperature-controlled room at approximately 22 degrees Celsius. Unfortunately, cell surface temperature measurements are not available for this dataset.

Eleven cycling groups were formed (4 cells per group), each with unique aging conditions. The groups vary in depth of discharge (DOD), charging C-rate, and discharging C-rate. The set of applied cycling parameters is provided in the "cycling_parameters.csv" file. Group aging trajectories are shown in Figure 1 of the paper associated with this dataset (see the CITATION section for the paper's DOI). The aging duration varies from 286 cycles (group 9) to 3488 cycles (group 8).



**********************************************************************
CONTENTS OF ZIP FILE(S)

This dataset contains 7 files:
1. 'README.txt' - this file
2. 'rpt_data.zip' — a folder containing all RPT data
3. 'cycling_data.zip' - a folder containing all cycling data
4. 'cycling_parameters.csv' — aging details for each cycling group
5. 'CITATION.bib' - BibTex citation details
6. 'LICENSE.txt' - terms of use license agreement

The ‘rpt_data’ and ‘cycling_data’ folders contain files with the following naming pattern: ‘rpt_cell_XX_partYY.csv’
1. XX refers to the cell number (01 to 44)
2. YY refers to the partition number. Data belonging to each cell is partitioned into ~100MB CSV files. Cells may have a varying number of partitions depending on their aging duration.

Each CSV file under ‘cycling_data’ contains the following column headers:
1. Week Number — The week number at the start of each protocol
2. Life — Indicates first or second life aging. Note that this published version contains only 1st life data.
3. Date (yyyy.mm.dd hh.mm.ss) — a timestamp in the specified format
4. Cycle Number — the cycle number relative to each protocol
5. State — the state of the applied step (CCCV Chg, CCCV DChg, etc)
6. Time (s) — time since the start of each protocol (in seconds) 
7. Voltage (V) — the cell’s terminal voltage
8. Current (A) — the cell’s charge or discharge current (negative during discharge)
9. Capacity (Ah) — the cell’s cumulative capacity (starts at 0 for each protocol)

The CSV files under ‘rpt_data’ contain the same columns header in ‘cycling_data’ in addition to the following:
10. Step Number — the RPT protocol step number
11. Segment Key — ‘ref_chg’, ‘ref_dchg’, ‘slowpulse’, ‘fastpulse’* 
12. Pulse Type — ‘chg’ or ‘dchg’ to indicate the pulse type
13. Pulse SOC — the SOC at which the specified pulse is applied**
14. Num Cycles — the number of cycles performed during cycling since the last RPT***

* This column is used to denote specific regions of the RPT protocol (e.g., the full charge and discharge reference cycles or the applied pulses).
** Note that the intended SOC drifts away from the actual SOC as the cell ages. This SOC drift issue is discussed in Nowacki et al.  
*** Num Cycles is non-cumulative over Week Number


**********************************************************************
IMPLEMENTATION DETAILS

Python scripts for loading and processing this data are provided on GitHub at: https://github.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation


**********************************************************************
CITATION

If you make use of this data, please cite it using the following:
@unpublished{nowacki2025finetuning,
  title = {Fine-tuning for rapid capacity estimation of lithium-ion batteries},
  author = {Benjamin Nowacki and Thomas Schmitt and Phillip Aquino and Chao Hu},
  year = {in prep.},
}
This citation will be updated when the paper is published.


Thank you for your interest in our work.
Benjamin Nowacki
2025


**********************************************************************
LICENSE AGREEMENT

The UConn-ILCC NMC/Gr Battery Aging Dataset is licensed under a
Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.




