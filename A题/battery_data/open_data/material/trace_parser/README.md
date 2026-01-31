## Perfetto Trace Script

Can be used to extract specific data from a perfetto trace file.

The most recent version, ScriptV2, uses the perfetto python API to extract the data from the trace.
ScriptV2 is way more compact and efficient than the previous version and should always be used.

**ScriptV1 is deprecated and is just here for references.**

So far the metrics tracked are : 
- Power rails data
- Average frequency for the little CPU group
- Average frequency for the medium CPU group (big in the trace)
- Average frequency for the big CPU group (bigger in the trace)
- Frequency of the first GPU
- Frequency of the second GPU
- Average GPU memory frequency
- Battery discharge total
- Battery discharge rate
- Total Data WIFI IN/OUT
- Average SOC temperature
- Difference SOC temperature
- Battery percent

Supported traces format are :
- perfetto-trace
- proto

The default input directory is './in', the default CSV output is './out/out.csv' and the default slice is [0:100]

### Usage : 

Default usage example
```bash
cd ScriptV2
python3 main.py       
```

Usage example with options
```bash
cd ScriptV2
python3 main.py --in ./my_input_dir --out ./my_output.csv --slice [25:75]   
```

Options :

*--in* : Can be used to specify the input director from where traces will be loaded for analysis

*--out* : Can be used to specify a CSV for the results. If the CSV cannot be found, it will be created.

**WARNING** The CSV you used will be EMPTIED at the beginning of the analysis

*--slice* : Can be used to specify the range of the trace to analyze **FOR THE POWER RAILS ONLY**. 
No matter what the slice given is, other traced metrics will be for the entire trace.

The slice format is as follows 
```bash
[x:y]
```
Where x is the start of the slice and y is the end of the slice.
With x and y being integers and 0 <= x < y <= 100.

References :
- [Perfetto](https://perfetto.dev/)
- [Perfetto Python API](https://perfetto.dev/docs/analysis/trace-processor-python)
- [Protobuf](https://protobuf.dev/getting-started/pythontutorial/)
- [Android Metrics](https://cs.android.com/android/platform/superproject/main/+/main:external/perfetto/protos/perfetto/metrics/metrics.proto;l=163?q=powrails&sq=&ss=android%2Fplatform%2Fsuperproject%2Fmain&hl=fr)


