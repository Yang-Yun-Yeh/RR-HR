import numpy as np
import pandas as pd

from glob import glob
from utils.visualize import *

if __name__ == '__main__':
    file_path = "./data/2_20/stand2walk_3.csv" #./data/1_19/stand_1.csv, ./data/2_20/stand2walk_3.csv

    fs = 10
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
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.set_index("Timestamp")

    start_pt, end_pt = 50, -50
    draw_imu_curve(data.iloc[start_pt:end_pt], overlap=True, show_gt=True)