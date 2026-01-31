# ANONYMOUS, 2024
import argparse
import csv
import os
import sys
import re
from multiprocessing import Pool

from perfetto.trace_processor import TraceProcessor
from google.protobuf.json_format import MessageToDict


cpu_core_speeds_cluster0=[324000.0, 615000.0, 820000.0, 975000.0, 1098000.0, 1197000.0, 1328000.0, 1475000.0, 1548000.0, 1704000.0, 1844000.0, 1950000.0, 2024000.0, 2098000.0, 2147000.0]
cpu_core_speeds_cluster1=[402000.0, 578000.0, 697000.0, 721000.0, 910000.0, 1082000.0, 1221000.0, 1328000.0, 1418000.0, 1549000.0, 1622000.0, 1836000.0, 1999000.0, 2130000.0, 2245000.0, 2352000.0, 2450000.0]
cpu_core_speeds_cluster2=[500000.0, 893000.0, 1164000.0, 1328000.0, 1557000.0, 1745000.0, 1852000.0, 1901000.0, 2049000.0, 2147000.0, 2294000.0, 2409000.0, 2556000.0, 2687000.0, 2802000.0, 2914000.0, 3015000.0]

"""
load_input_filename : Load all the trace files in the in directory

Returns:
    filenames : List of all the trace files
"""
def load_input_filename(directory):
    filenames = ([],[])
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.perfetto-trace') or filename.endswith('.proto'):
                filenames[1].append(filename)
                file_path = os.path.join(root, filename)
                filenames[0].append(file_path)
    return filenames

"""
create_rail_entry : Create a dictionary for a power rail

Parameters:
    name : Name of the power rail
    energy_list : List of energy values
    
Returns:
    Dictionary for the power rail
"""
def create_rail_entry(name, energy_list, time_list):
    return {
        "rail": name,
        "total_energy": 0,
        "total_time_ms": 0,
        "energy_list": energy_list,
        "time_list": time_list,
        "delta_list": [],
        "delta_time": [],
        "battery_poll_s": 0
    }

"""
cpu_freq_compilation : Calculate the average frequency for each CPU group

Parameters:
    freq_tab : Dictionary containing the frequency data for each CPU group
    trace_duration_ns : Duration of the trace
    
Returns:
    little : Average frequency for the little CPU group
    medium : Average frequency for the medium CPU group (big in the trace)
    big : Average frequency for the big CPU group (bigger in the trace)
"""
def cpu_freq_compilation(freq_tab, trace_duration_ns):
    little = 0
    medium = 0
    big = 0
    for key in freq_tab:
        num = 0
        total_time = 0
        for elt in freq_tab[key]:
            total_time += int(elt[1])
            num += int(elt[0])*int(elt[1])
        if key == "Little":
            num += cpu_core_speeds_cluster0[0] * (int(trace_duration_ns) - total_time)
            little = num / int(trace_duration_ns)
        if key == "Medium":
            num += cpu_core_speeds_cluster1[0] * (int(trace_duration_ns) - total_time)
            medium = num / int(trace_duration_ns)
        if key == "Big":
            num += cpu_core_speeds_cluster2[0] * (int(trace_duration_ns) - total_time)
            big = num / int(trace_duration_ns)
    return little, medium, big

"""
energy_delta : Calculate the energy delta for a power rail

Parameters:
    list_energy : List of energy values
    
Returns:
    end - start : Give total energy consumption
    list_delta : List of energy deltas
"""
def energy_delta(list_energy, list_time):
    list_delta = []
    delta_time = []
    size = len(list_energy)
    start_energy = list_energy[0]
    end_energy = list_energy[size - 1]
    j = start_energy
    for i in list_energy:
        v = i - j
        list_delta.append(v)
        j = i

    for i in range(size - 1):
        delta_time.append(int(list_time[i+1]) - int(list_time[i]))

    return end_energy - start_energy, list_delta, delta_time, int(list_time[-1]) - int(list_time[0])


def average_discharge_rate(battery_data):
    """
    average_discharge_rate : Calculate the average discharge rate of the battery per second

    Parameters:
        battery_data : Battery data

    Returns:
        avg_discharge_rate : Average discharge rate of the battery per second
    """
    list_current_ua = []
    deltas = []
    battery_counters = battery_data.get("batteryCounters", [])
    for i in range(1, len(battery_counters)):
        prev_timestamp = int(battery_counters[i - 1]["timestampNs"])
        curr_timestamp = int(battery_counters[i]["timestampNs"])
        list_current_ua.append(- int(battery_counters[i]["currentUa"]))
        delta_s = (curr_timestamp - prev_timestamp) / 1e9
        deltas.append(delta_s)


    num = 0
    den = 0
    for i in range(len(deltas)):
        num += list_current_ua[i] * int(deltas[i])
        den += int(deltas[i])

    battery_avg_discharge_ua_s = num / den

    return battery_avg_discharge_ua_s

"""
parse_file : Parse the trace file and extract the relevant data

Parameters:
    filename : Name of the trace file
    
Returns:
    rails_data : List of power rail data
    cpu_little_freq : Average frequency for the little CPU group
    cpu_medium_freq : Average frequency for the medium CPU group (big in the trace)
    cpu_big_freq : Average frequency for the big CPU group (bigger in the trace)
    gpu0_freq : Frequency of the first GPU
    gpu1_freq : Frequency of the second GPU
    gpu_mem_avg : Average GPU memory frequency
    battery_discharge : Battery discharge
"""
def parse_file(filename):
    freq_tab = {
        "Little" : [],
        "Medium" : [],
        "Big" : []
    }

    gpu0_freq = 0
    gpu1_freq = 0
    gpu_mem_avg = 0

    tp = TraceProcessor(trace=filename)


    #Parse trace metadata
    metadata_metrics = tp.metric(['trace_metadata'])
    metadata_metrics_dict = MessageToDict(metadata_metrics)
    traceConfigPbtxt = metadata_metrics_dict['traceMetadata']['traceConfigPbtxt']
    trace_duration_ns = metadata_metrics_dict['traceMetadata']['traceDurationNs']

    pattern = r"battery_poll_ms:\s*(\d+)"
    result = re.search(pattern, traceConfigPbtxt)

    if result:
        battery_poll_s =  int(result.group(1))/1000
    else:
        battery_poll_s = 1

    # Parse GPU metrics
    ad_gpu_metrics = tp.metric(['android_gpu'])
    gpu_dict = MessageToDict(ad_gpu_metrics)

    for elt in gpu_dict['androidGpu']['freqMetrics']:
        if elt['gpuId'] == 0:
            gpu0_freq = int(float(elt['freqAvg']))

        if elt['gpuId'] == 1:
            gpu1_freq = int(float(elt['freqAvg']))

    gpu_mem_avg = int(gpu_dict['androidGpu']['memAvg'])

    # Parse power rails
    ad_power_rails_metrics = tp.metric(['android_powrails'])
    rails_data = []
    rails_dict = MessageToDict(ad_power_rails_metrics)
    for elt in rails_dict['androidPowrails']['powerRails']:
        list_energy = []
        list_time = []
        name = elt.get('name')
        for i in elt['energyData']:
            list_energy.append(i.get('energyUws'))
            list_time.append(i.get('timestampMs'))
        rails_data.append(create_rail_entry(name, list_energy, list_time))
    for i in rails_data:
        res = energy_delta(i["energy_list"], i["time_list"])
        i["total_energy"] = res[0]
        i["delta_list"] = res[1]
        i["delta_time"] = res[2]
        i["total_time_ms"] = res[3]
        i["delta_list"].pop(0)
        i["battery_poll_s"] = battery_poll_s


    # Parse CPU metrics
    ad_cpu_metrics = tp.metric(['android_cpu'])
    cpu_dict = MessageToDict(ad_cpu_metrics)
    for elt in cpu_dict['androidCpu']['processInfo']:
        for ct in elt.get('coreType', []):
            core_type = ct.get('type')
            metrics = ct.get('metrics', {})

            if core_type == 'little':
                freq_tab["Little"].append([
                    metrics.get('avgFreqKhz', 0),
                    metrics.get('runtimeNs', 0)
                ])
            elif core_type == 'big':
                freq_tab["Medium"].append([
                    metrics.get('avgFreqKhz', 0),
                    metrics.get('runtimeNs', 0)
                ])
            elif core_type == 'bigger':
                freq_tab["Big"].append([
                    metrics.get('avgFreqKhz', 0),
                    metrics.get('runtimeNs', 0)
                ])

    # Calculate average cpu frequency
    res =  cpu_freq_compilation(freq_tab, trace_duration_ns)
    if res[0] <= 0:
        cpu_little_freq = 'err'
    else:
        cpu_little_freq = int(res[0])

    if res[1] <= 0:
        cpu_medium_freq = 'err'
    else:
        cpu_medium_freq = int(res[1])

    if res[2] <= 0:
        cpu_big_freq = 'err'
    else:
        cpu_big_freq = int(res[2])


    # Parse Battery metrics
    ad_battery_metrics = tp.metric(['android_batt'])
    battery_dict = MessageToDict(ad_battery_metrics)
    battery_avg_discharge_rate = average_discharge_rate(battery_dict['androidBatt'])

    battery_start = battery_dict['androidBatt']['batteryCounters'][0]['chargeCounterUah']
    battery_end = battery_dict['androidBatt']['batteryCounters'][-1]['chargeCounterUah']
    battery_discharge_total = battery_end - battery_start

    battery_percent = battery_dict['androidBatt']['batteryCounters'][-1]['capacityPercent']


    # Parse network packets
    total_data = 0
    qr_network = tp.query(
        'SELECT name, type, packet_length, direction '
        'FROM __intrinsic_android_network_packets '
        'WHERE type = "__intrinsic_android_network_packets"'
    )

    for row in qr_network:
        total_data += int(row.packet_length)

    # Parse temp data
    list_temp = []
    avg_temp = 0
    idx = 0
    qr_temp = tp.query(
        'SELECT name, value '
        'FROM counter_track '
        'JOIN counter ON counter_track.id = counter.track_id '
        'WHERE name = "soc_therm Temperature" '
    )
    for row in qr_temp:
        list_temp.append(row.value)
        avg_temp += int(row.value)
        idx += 1

    if idx > 0:
        avg_temp /= idx

    if len(list_temp) > 1:
        diff_temp = int(list_temp[-1]) - int(list_temp[0])
    else:
        diff_temp = 0

    return rails_data, cpu_little_freq, cpu_medium_freq, cpu_big_freq, gpu0_freq, gpu1_freq, gpu_mem_avg, battery_discharge_total, battery_avg_discharge_rate, total_data, avg_temp, diff_temp, battery_percent #Rounded for comprehension

"""
result_to_csv : Write the data to a CSV file

Parameters:
    data : List of data to write to the CSV file
"""
def result_to_csv(data, output):
    open(output, 'w').close()
    first_line = (
                                'Trace;'
                             'L21S_VDD2L_MEM_ENERGY_AVG_UWS;'
                             'UFS(Disk)_ENERGY_AVG_UWS;'
                             'S12S_VDD_AUR_ENERGY_AVG_UWS;'
                             'Camera_ENERGY_AVG_UWS;'
                             'GPU3D_ENERGY_AVG_UWS;'
                             'Sensor_ENERGY_AVG_UWS;'
                             'Memory_ENERGY_AVG_UWS;'
                             'Memory_ENERGY_AVG_UWS;'
                             'Display_ENERGY_AVG_UWS;'
                             'GPS_ENERGY_AVG_UWS;'
                             'GPU_ENERGY_AVG_UWS;'
                             'WLANBT_ENERGY_AVG_UWS;'
                             'L22M_DISP_ENERGY_AVG_UWS;'
                             'S6M_LLDO1_ENERGY_AVG_UWS;'
                             'S8M_LLDO2_ENERGY_AVG_UWS;'
                             'S9M_VDD_CPUCL0_M_ENERGY_AVG_UWS;'
                             'CPU_BIG_ENERGY_AVG_UWS;'
                             'CPU_LITTLE_ENERGY_AVG_UWS;'
                             'CPU_MID_ENERGY_AVG_UWS;'
                             'INFRASTRUCTURE_ENERGY_AVG_UWS;'
                             'CELLULAR_ENERGY_AVG_UWS;'
                             'CELLULAR_ENERGY_AVG_UWS;'
                             'INFRASTRUCTURE_ENERGY_AVG_UWS;'
                             'TPU_ENERGY_AVG_UWS;'
                             'CPU_LITTLE_FREQ_KHz;'
                             'CPU_MID_FREQ_KHz;'
                             'CPU_BIG_FREQ_KHz;'
                             'GPU0_FREQ;'
                             'GPU_1FREQ;'
                             'GPU_MEM_AVG;'
                             'BATTERY_DISCHARGE_TOTAL_UA;'
                             'BATTERY_DISCHARGE_RATE_UAS;'
                             'TOTAL_DATA_WIFI_BYTES;'
                             'AVG_SOC_TEMP;'
                             'DIFF_SOC_TEMP;'
                             'BATTERY__PERCENT;'
                            'L21S_VDD2L_MEM_ENERGY_UW;'
                            'UFS(Disk)_ENERGY_UW;'
                            'S12S_VDD_AUR_ENERGY_UW;'
                            'Camera_ENERGY_UW;'
                            'GPU3D_ENERGY_UW;'
                            'Sensor_ENERGY_UW;'
                            'Memory_ENERGY_UW;'
                            'Memory_ENERGY_UW;'
                            'Display_ENERGY_UW;'
                            'GPS_ENERGY_UW;'
                            'GPU_ENERGY_UW;'
                            'WLANBT_ENERGY_UW;'
                            'L22M_DISP_ENERGY_UW;'
                            'S6M_LLDO1_ENERGY_UW;'
                            'S8M_LLDO2_ENERGY_UW;'
                            'S9M_VDD_CPUCL0_M_ENERGY_UW;'
                            'CPU_BIG_ENERGY_UW;'
                            'CPU_LITTLE_ENERGY_UW;'
                            'CPU_MID_ENERGY_UW;'
                            'INFRASTRUCTURE_ENERGY_UW;'
                            'CELLULAR_ENERGY_UW;'
                            'CELLULAR_ENERGY_UW;'
                            'INFRASTRUCTURE_ENERGY_UW;'
                            'TPU_ENERGY_UW;'
                             ).split(';')
    first_line.pop()
    with open(output, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(first_line)
        for elt in data:
            spamwriter.writerow(elt)

"""
process_result : Prepare the result to be written to the CSV file

Parameters:
    trace_name : Name of the trace
    data : List of data to write to the CSV file
    power_rails_slice : Slice of the power rails to consider given in the main   
    
Returns:
    line_elements : List of elements to write to the CSV file
"""
def process_result(trace_name, data, power_rails_slice):
    line_elements = [trace_name[:-15]]
    pattern = r"^\[(\d+):(\d+)\]$"
    match = re.match(pattern, power_rails_slice)
    x, y = map(int, match.groups())

    for elt in data[0]:
        num = 0
        den = 0
        size = len(elt["delta_list"])
        start = int((x / 100) * size)
        end = int((y / 100) * size)
        for i in range(start, end-1):
            num += int(elt["delta_list"][i]) * int(elt["delta_time"][i])
            den += int(elt["delta_time"][i])
        if not den == 0:
            line_elements.append(str(num/den/elt["battery_poll_s"]))
        else :
            line_elements.append("0")

    for elt in data[1:]:
        line_elements.append(str(elt))

    for elt in data[0]:
        line_elements.append(str(elt["total_energy"]))

    return line_elements

"""
slice_validation : Validate the slice format

Parameters:
    value : Slice to validate
    
Returns:
    res : True if the slice is valid, False otherwise
"""
def slice_validation(value):
    res = False
    pattern = r"^\[(\d+):(\d+)\]$" # [0:100]
    match = re.match(pattern, value)
    if match:
        res =  True
    x, y = map(int, match.groups())
    if not (0 <= x < 100 and 0 < y <= 100 and x < y):
            res =  False
            print('slice format : [x:y] with 0 < x < 100 and 0 < y < 100 and x < y')
    return res

"""
process_file : Process a trace file

Parameters:
    filename_and_slice : Tuple containing the filename and the power rail slice
    
Returns:
    process_result : Result of the process or None if an error occurred
"""


def process_file(filename_and_slice):
    filename, power_rails_slice, trace_name = filename_and_slice
    try:
        print(f"Processing {trace_name}")
        data = parse_file(filename)
        return process_result(trace_name, data, power_rails_slice)
    except Exception as e:
        print(f"Error processing {trace_name}: {e}")
        return None


def main():
    slice_powerails = "[0:100]"
    dir_input = "./in"
    output = "./out/out.csv"

    parser = argparse.ArgumentParser(description="Optionnal parameter.")

    parser.add_argument("--slice", type=str, help="Slice of the power rails to consider", required=False)
    parser.add_argument("--input", type=str, help="Trace directory", required=False)
    parser.add_argument("--out", type=str, help="CSV for the result", required=False)

    args = parser.parse_args()

    if args.slice:
        slice_powerails = args.slice
    if args.input:
        dir_input = args.input
    if args.out:
        output = args.out

    if not slice_validation(slice_powerails):
        print("Usage: python3 main.py --slice (format : [0:100])")
        sys.exit(0)

    filenames = load_input_filename(dir_input)

    if len(filenames[0]) == 0:
        print("No trace file found")
        sys.exit(0)

    with Pool() as pool:
        results = pool.map(process_file, [(filenames[0][i], slice_powerails, filenames[1][i]) for i in range(len(filenames[0]))])

    valid_results = [result for result in results if result is not None]

    if valid_results:
        result_to_csv(valid_results, output)
    else:
        print("No valid results to write.")

if __name__ == '__main__':
    main()

