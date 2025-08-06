import os
import argparse
from datetime import datetime
import pandas as pd
from utils.signal_process import *

def parse_args():
    parser = argparse.ArgumentParser(description="compute dataset time")
    parser.add_argument("--ckpt_dir", type=str, default="models", help="path to save checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="data", help="path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="17P")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    dir = str(f"./{args.dataset_dir}/{args.dataset_name}")
    print(f"target_folder: {dir}")

    t_total = 0
    for person in os.listdir(dir):
        person_name = os.fsdecode(person)

        # iterate all files
        dir_p = os.path.join(dir, person_name)
        for file in os.listdir(dir_p):
            filename = os.fsdecode(file)
            action_name = filename.split("_")[0]

            if filename.endswith(".csv"): 
                # print(os.path.join(dir_p, filename))
                # load data
                data = pd.read_csv(os.path.join(dir_p, filename))
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


                # align delay
                data = align_delay(data, delay=10) # sp.align_delay(data, delay=10)
                data["Timestamp"] = pd.to_datetime(data["Timestamp"])

                N = len(data) - 1
                delta_t = (data["Timestamp"][N] - data["Timestamp"][0]).total_seconds()
                t_total += delta_t

    print(f'all: {t_total:.2f}(s), {t_total/60:.2f}(min)\n')


    # dataset_types = ['train', 'test']
    # for tp in dataset_types:
    #     folder = str(f"./{args.dataset_dir}/{args.dataset_name}")
    #     target_folder = os.path.join(folder, tp)
    #     print(f"target_folder: {target_folder}")

    #     t_total = 0
    #     # iterate all files
    #     for file in os.listdir(target_folder):
    #         filename = os.fsdecode(file)
    #         if filename.endswith(".csv"): 
    #             # print(os.path.join(target_folder, filename))
    #             action_name = filename.split("_")[0]
                
    #             # load data
    #             data = pd.read_csv(os.path.join(target_folder, filename))
    #             data.columns = [
    #                 "Timestamp",
    #                 "imu1_q_x",
    #                 "imu1_q_y",
    #                 "imu1_q_z",
    #                 "imu1_q_w",
    #                 "imu2_q_x",
    #                 "imu2_q_y",
    #                 "imu2_q_z",
    #                 "imu2_q_w",
    #                 "Force",
    #                 "RR",
    #             ]


    #             # align delay
    #             data = align_delay(data, delay=10) # sp.align_delay(data, delay=10)
    #             data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    #             N = len(data) - 1
    #             delta_t = (data["Timestamp"][N] - data["Timestamp"][0]).total_seconds()
    #             t_total += delta_t

    #     print(f'{tp}: {t_total:.2f}(s), {t_total/60:.2f}(min)\n')