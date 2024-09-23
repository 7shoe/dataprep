import psutil
import GPUtil
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def log_resource_usage(kill_minutes, log_path):
    log_data = []

    start_time = time.time()
    end_time = start_time + kill_minutes * 60.0

    while time.time() < end_time:
        # Capture the current timestamp
        current_time = round(time.time()) # datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # CPU and Memory Usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        # GPU Usage
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_name = gpu.name
            gpu_load = gpu.load * 100
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_memory_util = gpu.memoryUtil * 100
            gpu_temp = gpu.temperature

            # Log the data as a dictionary
            log_data.append({
                'time': current_time,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'gpu_name': gpu_name,
                'gpu_load': gpu_load,
                'gpu_memory_used': gpu_memory_used,
                'gpu_memory_total': gpu_memory_total,
                'gpu_memory_util': gpu_memory_util,
                'gpu_temp': gpu_temp
            })
        
        time.sleep(5)

    # Convert log data to DataFrame
    df_log = pd.DataFrame(log_data)

    # Save DataFrame as CSV
    p = Path('/home/siebenschuh/Projects/dataprep/code/performance/single_GPU/logs')
    log_file_path = p / log_path
    df_log.to_csv(log_file_path, index=False)
    print(f"Log saved to {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Track and log resource usage.")
    parser.add_argument('--kill_minutes', type=float, required=True, help="Number of minutes after which the program writes its information into a log file and then is killed.")
    parser.add_argument('--log_file', type=str, required=True, help="The file name of the log file.")

    args = parser.parse_args()

    log_resource_usage(args.kill_minutes, args.log_file)

if __name__ == "__main__":
    main()

