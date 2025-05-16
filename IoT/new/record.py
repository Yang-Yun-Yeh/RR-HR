from serial import Serial
import csv
from datetime import datetime
from godirect import gdx
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate IMU dataset')
    
    # Path
    parser.add_argument('-f', '--folder', type=str, default='csv/date', help='data directory')
    parser.add_argument('-p', '--person', type=str, default='hamham', help='person name')
    parser.add_argument("-a", "--activity", type=str, default='test')

    # Record
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--still', type=int, default=300)
    
    # Visualize
    parser.add_argument("--vs", action='store_true')
    
    
    args = parser.parse_args()
    
    return args

def align_delay(data, delay=10, fs=10):
    gt_shift = data["Force"][int(delay * fs):].to_list()
    data.loc[:len(gt_shift)-1, "Force"] = gt_shift
    data = data[:len(gt_shift)-1]

    return data

def draw_imu_curve(imu_data, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w'], overlap=False, show_gt=False):
    # titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']
    titles = []
    for i in range(len(cols)):
        titles.append(f'${cols[i]}$')

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'yellow']

    legend_handle_ls = []
    row_num = len(sensor_names)
    if not overlap:
        if not show_gt:
            row_num = row_num
        else:
            row_num += 1
    else:
        if not show_gt:
            row_num += 1
        else:
            row_num += 2
    
    row_num += 1
    # fig, axes = plt.subplots(row_num, len(titles), figsize=(20,10), constrained_layout=True)
    fig = plt.figure(figsize=(10, 1.5 * row_num), layout="constrained") # (15, 3 * row_num)
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles)):
            # print(f'{i}, {key}, {cols[i]}')
            # axes[k][i%4].plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            # axes[k][i%4].set_title(titles[i])

            ax = fig.add_subplot(spec[k, i%4])
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])

        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    if overlap:
        for i in range(len(titles)):
            ax = fig.add_subplot(spec[2, i%4])
            for k, key in enumerate(sensor_names):
                ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])

    if show_gt:
        if not overlap:
            ax = fig.add_subplot(spec[2, :])
        else:
            ax = fig.add_subplot(spec[3, :])
        ax.plot(time_ls, imu_data["Force"], color=colors[2])
        ax.set_title('Respiration GT Force')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))        

    # ax = fig.add_subplot(spec[4, :])
    # ax.plot(time_ls, imu_data["imu2_q_y"], color=colors[1])
    # ax.set_title('imu2_q_y')
    
    fig.legend(handles=legend_handle_ls, loc="upper right")
    
    # plt.savefig('output/figures/a.png')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    
    client = Serial("/dev/ttyUSB0", 921600) # imu1, imu2
    gdx = gdx.gdx() # rr gt

    gdx.open_usb()
    gdx.select_sensors([1, 2])  # Force, RR
    gdx.start(period=50) # 50

    print("Warming up!")
    time.sleep(5)
    print(f"Lift off...")

    start_time = time.time()
    duration = 625 # 315
    now = datetime.now().strftime("%m%d_%H%M")
    start_pt = args.start # 100
    still_pt = args.still # 300

    folder = args.folder #"csv/5_11"
    person = args.person
    activity = args.activity #"test"
    
    file_path = f"{folder}/{person}/{activity}_{now}.csv"
    
    if not os.path.exists(f"{folder}/{person}"):
        os.makedirs(f"{folder}/{person}")
        
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, "a") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp",
                    "imu1_q_x", "imu1_q_y", "imu1_q_z", "imu1_q_w",
                    "imu2_q_x", "imu2_q_y", "imu2_q_z", "imu2_q_w",
                    "Force", "RR"])

        try:
            count = 0 # wait for quaternions to stabilize
            while time.time() - start_time < duration:
                # print(time.time() - start_time)
                raw = client.readline()
                resp = gdx.read()
                data = raw.decode('utf-8').strip()

                if data.startswith("$"):
                    item = data[1:].strip().replace("#", ",")
                    results = [float(x) for x in item.split(',')]
                    results = results + resp
                    # if count >= 300 and results:
                    if count >= start_pt and results: # start record: 5 sec
                        if count >= start_pt + still_pt: # start_pt + 300
                            print(time.time() - start_time)
                        t = datetime.now().isoformat()
                        row = [t] + results
                        w.writerow(row)
                        f.flush()
                    count += 1
        finally:
            client.close()
            gdx.stop()
            gdx.close()
            
            if args.vs:
                # time.sleep(1)
                fs = 10
                start_pt, end_pt = 0, -1
                still_pt = 300 # 500

                data = pd.read_csv(file_path)
                data.columns = [
                    "Timestamp",
                    "imu1_q_x",
                    "imu1_q_y",
                    "imu1_q_z",
                    "imu1_q_w",
                    "imu2_q_x",
                    "imu2_q_y",
                    "imu2_q_z",
                    "imu2_q_w",
                    "Force",
                    "RR",
                ]
                sensor_names=['imu1','imu2']
                cols = ['q_x', 'q_y', 'q_z', 'q_w']
                data = data.iloc[start_pt:end_pt]

                data = align_delay(data, delay=10)
                data["Timestamp"] = pd.to_datetime(data["Timestamp"])
                data = data.set_index("Timestamp")

                draw_imu_curve(data, overlap=False, show_gt=True)

    print("Exiting...")
