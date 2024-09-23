from pathlib import Path
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import sqlite3
import numpy as np


def load_cpu_frame(parser='marker', p_root=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/performance_tracker/AdaParsePerformanceAnalysis')):
    table_name = 'resource'
    
    # load
    f_cpu = f'{parser}_perf_log.db'

    # check
    p_root = Path(p_root)
    assert p_root.is_dir(), "`p_root` invalid. Not dir path"
    assert (p_root / f_cpu).is_file(), f"`{p_root / f_cpu}` does not exist. Cannot find the file."

    # load resource table
    conn = sqlite3.connect(p_root / f_cpu)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    # pre-processing
    # convert
    columns_to_convert = [
        'psutil_process_time_user', 
        'psutil_process_time_system', 
        'resource_monitoring_interval', 
        'psutil_cpu_num'
    ]
    for column in columns_to_convert:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # - handle any NaN values (e.g., drop rows with NaNs in these columns)
    df.dropna(subset=columns_to_convert, inplace=True)
    
    # calculate CPU utilization (normalized by number of logical CPUs)
    df['cpu_utilization'] = (((df['psutil_process_time_user'] + df['psutil_process_time_system']))) / df['psutil_cpu_num']
    
    # memory utilization is already given as a percentage
    df['memory_utilization'] = df['psutil_process_memory_percent'] # that's iit
    
    # convert time series (time after experiment begin [min])
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip())
    df['timestamp'] = df['timestamp'].astype(int) // 10**9
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / 60.0
    
    # Replace inf values
    df.replace({'cpu_utilization': {np.inf: np.nan, -np.inf: np.nan}, 
                'memory_utilization': {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
    
    # Drop rows where 'cpu_utilization' or 'memory_utilization' is NaN
    df.dropna(subset=['cpu_utilization', 'memory_utilization'], inplace=True)
    
    return df

def load_gpu_frame(parser='marker', p_root=Path('/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/performance_tracker/AdaParsePerformanceAnalysis')):
    """
    Load Pandas DataFrame with GPU-related data
    """
    
    # load
    f_gpu = f'{parser}_gpu_logs.csv'

    # check
    p_root = Path(p_root)
    assert p_root.is_dir(), "`p_root` invalid. Not dir path"
    assert (p_root / f_gpu).is_file(), "`p_root` exists but did not find  file {}"
    
    df = pd.read_csv(p_root / f_gpu)
    # del whitespaces
    df.columns = df.columns.str.strip()
    
    # subset
    cols = ['timestamp', 'utilization.gpu [%]', 'utilization.memory [%]']
    df = df[cols]
    
    # filter & convert to numeric
    df = df[df['utilization.gpu [%]'].str.endswith('%') & df['utilization.memory [%]'].str.endswith('%')]
    # - convert to numeric by removing '%'
    df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.rstrip('%').astype(float)
    df['utilization.memory [%]'] = df['utilization.memory [%]'].str.rstrip('%').astype(float)
    
    # filter header rows in df
    df = df[~df['timestamp'].str.contains('timestamp', case=False, na=False)]
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip()).astype(int) // 10**9
    # convert to minutes
    df['timestamp'] = (df['timestamp'] - min(df['timestamp'])) / 60.

    return df 

def plot_gpu_timeseries(df, parser:str, window_size:int=100):
    """
    Plot timeseries
    """
    # Apply a rolling mean to smooth the data
    df['gpu_smooth'] = df['utilization.gpu [%]'].rolling(window=window_size).mean()
    df['memory_smooth'] = df['utilization.memory [%]'].rolling(window=window_size).mean()
    
    # Plotting the smoothed time series
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['gpu_smooth'], label='GPU Utilization [%]', color='blue')
    plt.plot(df['timestamp'], df['memory_smooth'], label='Memory Utilization [%]', color='red')
    plt.xlabel('Runtime [min]', fontsize=14)
    plt.ylabel('Utilization [%]', fontsize=14)
    plt.suptitle('GPU/Memory utilization over time', fontsize=15, y=0.97)
    plt.title('`nvidia-smi`-based, $n=200$ dataset', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    pass

def plot_cpu_timeseries(df, parser:str, window_size:int=2):
    """
    Plot timeseries
    """
    # Apply a rolling mean to smooth the data
    df['cpu_smooth'] = df['cpu_utilization'].rolling(window=window_size).mean()
    df['memory_smooth'] = df['memory_utilization'].rolling(window=window_size).mean()
    
    # Plotting the smoothed time series
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['cpu_smooth'], label='CPU Utilization [%]', color='blue')
    plt.plot(df['timestamp'], df['memory_smooth'], label='Memory Utilization [%]', color='red')
    plt.xlabel('Runtime [min]', fontsize=14)
    plt.ylabel('Utilization [%]', fontsize=14)
    plt.suptitle('CPU/Memory utilization over time', fontsize=15, y=0.97)
    plt.title('`nvidia-smi`-based, $n=200$ dataset', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    pass