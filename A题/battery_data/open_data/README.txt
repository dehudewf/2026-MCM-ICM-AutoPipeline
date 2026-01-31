This archive contain the complementary data for the publication "Power Efficiency Unpacked: Modeling Per-Component Power Consumption in Mobile Devices".

Warning - while the content of this archive has been anonymized to the best of our efforts, informations revealing the identify of authors may remain.


This archive contains: 
- ./material/trace_parser - the tool used to analyse android system traces, with its own readme.
- ./material/application-android-conso-tel - the source code of the android application used to stimulate hardware components
- ./material/res_test/aggregated.csv - the aggregated metrics of each of the 1000 test performed, including the state of the device and power usage
- ./material/analysis.py - the script that analyses aggregated.csv and generates the following files: (requires tunning)
- ./material/grid_search_results.csv - the result of grid search for each assessed model (Table 3 & Figure 1)
- ./material/per_component_accuracy.csv - the accuracy of each component's prediction, for 1000 measures (Table IV, Figure IV)
- ./material/size_comparison.csv - the impact of dataset size on the accuracy of prediction (Table V, Figure 5)
