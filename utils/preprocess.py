import os
import pandas as pd
import numpy as np

try:
     from . import signal_process as sp
except:
     import signal_process as sp

def prepare_data(dir, fs=10, start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    omega_axes = ['omega_u', 'omega_v', 'omega_w']
    ang_speed_cols = ['omega']

    q_col_ls = []
    omega_col_ls = []
    ang_speed_col_ls = []

    for imu in sensor_names:
        for col in cols:
            q_col_ls.append(imu + "_" + col)
        for col in omega_axes:
            omega_col_ls.append(imu + "_" + col)
        for col in ang_speed_cols:
            ang_speed_col_ls.append(imu + "_" + col)

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
            q_corr = sp.Q_RANSAC(data[0:still_pt], pool=pool, d=d)
            target, skew = 'imu1', 'imu2'
            Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
            Q_aligned = sp.align_quaternion(q_corr, Q_skew) # (sample num, 4)
            data_aligned = data.copy()

            for i, col in enumerate(cols):
                data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

            # specify data range
            data_sml = data_aligned.copy() # data used in sml
            data_sml = data_sml[still_pt+after_still_pt:]

            # data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=1) # cutoff=0.66

            # calculate omega features
            for imu in sensor_names:
                q = data_sml[[imu + "_" + "q_x", imu + "_" + "q_y", imu + "_" + "q_z", imu + "_" + "q_w"]].values # (sample num, 4)
                omega = sp.q_to_omega(q)
                ang_speed = sp.omega_to_AngSpeed(omega)
                for i, omega_axis in enumerate(omega_axes):
                    data_sml.loc[:, imu + "_" + omega_axis] = omega[:, i]
                
                data_sml.loc[:, imu + "_" + ang_speed_cols[0]] = ang_speed

            # Q: (sample_num, channel_num)
            Q = data_sml[q_col_ls + omega_col_ls + ang_speed_col_ls].values
            segmented_spectrograms, segmented_gt = sp.segment_data(Q, data_sml["Force"], window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1)

            # min-Max normalization
            segmented_spectrograms, segmented_gt, times = sp.segment_data(Q, data_sml["Force"], return_t=True, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1)
            if segmented_spectrograms.shape[1] == 8:
                segmented_spectrograms = sp.normalize_spectrogram(segmented_spectrograms)
            elif segmented_spectrograms.shape[1] == 16:
                segmented_spectrograms[:, :8] = sp.normalize_spectrogram(segmented_spectrograms[:, :8])
                segmented_spectrograms[:, 8:14] = sp.normalize_spectrogram(segmented_spectrograms[:, 8:14])
                segmented_spectrograms[:, 14:] = sp.normalize_spectrogram(segmented_spectrograms[:, 14:])
                
            # print(f'sepctrograms:{segmented_spectrograms.shape}')
            # print(f'gt:{segmented_gt.shape}')
            # exit()
            spectrograms.append(segmented_spectrograms)
            gts.append(segmented_gt)
    
    spectrograms = np.concatenate(spectrograms, axis=0)
    gts = np.concatenate(gts, axis=0)

    print('----------------------------')
    print(f'sepctrograms:{spectrograms.shape}')
    print(f'gt:{gts.shape}')

    return spectrograms, gts

def prepare_file(file, fs=10, start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    omega_axes = ['omega_u', 'omega_v', 'omega_w']
    ang_speed_cols = ['omega']

    q_col_ls = []
    omega_col_ls = []
    ang_speed_col_ls = []

    for imu in sensor_names:
        for col in cols:
            q_col_ls.append(imu + "_" + col)
        for col in omega_axes:
            omega_col_ls.append(imu + "_" + col)
        for col in ang_speed_cols:
            ang_speed_col_ls.append(imu + "_" + col)

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
    q_corr = sp.Q_RANSAC(data[0:still_pt], pool=pool, d=d)
    target, skew = 'imu1', 'imu2'
    Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
    Q_aligned = sp.align_quaternion(q_corr, Q_skew) # (sample num, 4)
    data_aligned = data.copy()

    for i, col in enumerate(cols):
        data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

    # specify data range
    data_sml = data_aligned.copy() # data used in sml
    data_sml = data_sml[still_pt+after_still_pt:]

    # data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=1) # cutoff=0.66
    # data_sml.interpolate()

    # calculate omega features
    for imu in sensor_names:
        q = data_sml[[imu + "_" + "q_x", imu + "_" + "q_y", imu + "_" + "q_z", imu + "_" + "q_w"]].values # (sample num, 4)
        omega = sp.q_to_omega(q)
        ang_speed = sp.omega_to_AngSpeed(omega)
        for i, omega_axis in enumerate(omega_axes):
            data_sml.loc[:, imu + "_" + omega_axis] = omega[:, i]
        
        data_sml.loc[:, imu + "_" + ang_speed_cols[0]] = ang_speed

    # data_sml.interpolate()

    # nan_indices = data_sml.isna()
    # print(f"data_sml nan_indices:{nan_indices}")

    # Q: (sample_num, channel_num)
    # print(q_col_ls + omega_col_ls + ang_speed_col_ls)
    Q = data_sml[q_col_ls + omega_col_ls + ang_speed_col_ls].values
    # Q = np.concatenate((q_normalize, omega_normalize, ang_speed_normalize), axis=1)

    # min-Max normalization
    segmented_spectrograms, segmented_gt, times = sp.segment_data(Q, data_sml["Force"], return_t=True, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1)
    if segmented_spectrograms.shape[1] == 8:
        segmented_spectrograms = sp.normalize_spectrogram(segmented_spectrograms)
    elif segmented_spectrograms.shape[1] == 16:
        segmented_spectrograms[:, :8] = sp.normalize_spectrogram(segmented_spectrograms[:, :8])
        segmented_spectrograms[:, 8:14] = sp.normalize_spectrogram(segmented_spectrograms[:, 8:14])
        segmented_spectrograms[:, 14:] = sp.normalize_spectrogram(segmented_spectrograms[:, 14:])

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