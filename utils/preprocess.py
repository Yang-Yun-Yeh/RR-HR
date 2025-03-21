import os
import pandas as pd
import numpy as np

try:
     from . import signal_process as sp
except:
     import signal_process as sp

def prepare_data(dir, fs=10, start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.1):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    q_col_ls = []
    for imu in sensor_names:
        for col in cols:
            q_col_ls.append(imu + "_" + col)

    spectrograms, gts = [], []
    # iterate all files
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            print(os.path.join(dir, filename))
            
            # load data
            data = pd.read_csv(os.path.join(dir, filename))
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

            data = data.iloc[start_pt:end_pt]

            # align delay
            data = sp.align_delay(data, delay=10)
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
            data = data.set_index("Timestamp")

            # align IMU
            q_corr = sp.Q_RANSAC(data[0:still_pt], pool=1.0, d=0.05)
            target, skew = 'imu1', 'imu2'
            Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
            Q_aligned = sp.align_quaternion(q_corr, Q_skew) # (sample num, 4)
            data_aligned = data.copy()

            for i, col in enumerate(cols):
                data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

            # specify data range
            data_sml = data_aligned # data used in sml
            data_sml = data_sml[still_pt+after_still_pt:]
            cols = ['q_x', 'q_y', 'q_z', 'q_w'] # for quaternion

            data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=0.66)

            # Q: (sample_num, channel_num)
            Q = data_sml[q_col_ls].values
            segmented_spectrograms, segmented_gt = sp.segment_data(Q, data_sml["Force"])
            # print(f'sepctrograms:{segmented_spectrograms.shape}')
            # print(f'gt:{segmented_gt.shape}')
            spectrograms.append(segmented_spectrograms)
            gts.append(segmented_gt)
    
    spectrograms = np.concatenate(spectrograms, axis=0)
    gts = np.concatenate(gts, axis=0)

    print('----------------------------')
    print(f'sepctrograms:{spectrograms.shape}')
    print(f'gt:{gts.shape}')

    return spectrograms, gts

def prepare_file(file, fs=10, start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.1):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    q_col_ls = []
    for imu in sensor_names:
        for col in cols:
            q_col_ls.append(imu + "_" + col)

    spectrograms, gts, times = [], [], []

    # load data
    data = pd.read_csv(file)
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

    data = data.iloc[start_pt:end_pt]

    data = sp.align_delay(data, delay=10)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.set_index("Timestamp")

    # align IMU
    q_corr = sp.Q_RANSAC(data[0:still_pt], pool=1.0, d=0.05)
    target, skew = 'imu1', 'imu2'
    Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
    Q_aligned = sp.align_quaternion(q_corr, Q_skew) # (sample num, 4)
    data_aligned = data.copy()

    for i, col in enumerate(cols):
        data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

    # specify data range
    data_sml = data_aligned # data used in sml
    data_sml = data_sml[still_pt+after_still_pt:]
    cols = ['q_x', 'q_y', 'q_z', 'q_w'] # for quaternion

    data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=0.66)

    # Q: (sample_num, channel_num)
    Q = data_sml[q_col_ls].values
    segmented_spectrograms, segmented_gt, times = sp.segment_data(Q, data_sml["Force"], return_t=True)
    # print(f'sepctrograms:{segmented_spectrograms.shape}')
    # print(f'gt:{segmented_gt.shape}')
    spectrograms.append(segmented_spectrograms)
    gts.append(segmented_gt)
    
    spectrograms = np.concatenate(spectrograms, axis=0)
    gts = np.concatenate(gts, axis=0)

    print('----------------------------')
    print(f'sepctrograms:{spectrograms.shape}')
    print(f'gt:{gts.shape}')
    print(f'times:{times.shape}')

    return spectrograms, gts, times