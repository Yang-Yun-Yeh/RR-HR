import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


try:
     from . import signal_process as sp
     from . import visualize as vs
except:
     import signal_process as sp
     import visualize as vs

def compute_file(data, features=['Q', 'omega', 'omega_l2', 'ANC'], start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False, byCol=False):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    omega_axes = ['omega_u', 'omega_v', 'omega_w']
    ang_speed_cols = ['omega']
    anc_methods = ['LMS', 'LMS+LS', 'RLS', 'LRLS']

    q_col_ls = []
    omega_col_ls = []
    ang_speed_col_ls = []
    anc_col_ls = []

    for imu in sensor_names:
        for col in cols:
            q_col_ls.append(imu + "_" + col)
        for col in omega_axes:
            omega_col_ls.append(imu + "_" + col)
        for col in ang_speed_cols:
            ang_speed_col_ls.append(imu + "_" + col)

    for anc_method in anc_methods:
        for col in cols:
            anc_col_ls.append(anc_method + "_" + col)
    
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
    data_anc = data_aligned.copy() # data used in anc
    data_sml = data_sml[still_pt+after_still_pt:]

    # data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=1) # cutoff=0.66

    # calculate omega features
    if 'omega' in features:
        for imu in sensor_names:
            q = data_sml[[imu + "_" + "q_x", imu + "_" + "q_y", imu + "_" + "q_z", imu + "_" + "q_w"]].values # (sample num, 4)
            omega = sp.q_to_omega(q)
            ang_speed = sp.omega_to_AngSpeed(omega)
            for i, omega_axis in enumerate(omega_axes):
                data_sml.loc[:, imu + "_" + omega_axis] = omega[:, i]
            
            data_sml.loc[:, imu + "_" + ang_speed_cols[0]] = ang_speed

    # calculate ANC features
    if 'ANC' in features:
        anc_outputs = sp.anc_process(data_anc, NTAPS=3, LEARNING_RATE=0.001, delta=1, lam_rls=0.9995, epsilon=1e-6, lam_lrls=0.9995)
        for anc_output in anc_outputs:
            method = anc_output['method']
            for col in cols:
                data_sml.loc[:, method + "_" + col] = anc_output[col][still_pt+after_still_pt:]

        
    # print(f'anc_outputs:{anc_outputs}')            

    # Q: (sample_num, channel_num)
    features_cols = []
    if 'Q' in features:
        features_cols.extend(q_col_ls)
    if 'omega' in features:
        features_cols.extend(omega_col_ls)
    if 'omega_l2' in features:
        features_cols.extend(ang_speed_col_ls)
    if 'ANC' in features:
        features_cols.extend(anc_col_ls)
    Q = data_sml[features_cols].values
    # print(f"features_cols:{features_cols}")
    # print(f'Q.shape:{Q.shape}')
    segmented_spectrograms, segmented_gt, times = sp.segment_data(Q, data_sml["Force"], return_t=True, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1)

    # segmented_spectrograms: (num_windows, num_spectrograms, freq_bins, time_steps)
    # min-Max normalization
    index_sp = 0
    feature_index = {}
    if 'Q' in features:
        feature_index['Q'] = (index_sp, index_sp+8)
        segmented_spectrograms[:, index_sp:index_sp+8] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+8], byCol=byCol)
        index_sp += 8
    if 'omega' in features:
        feature_index['omega'] = (index_sp, index_sp+6)
        segmented_spectrograms[:, index_sp:index_sp+6] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+6], byCol=byCol)
        index_sp += 6
    if 'omega_l2' in features:
        feature_index['omega_l2'] = (index_sp, index_sp+2)
        segmented_spectrograms[:, index_sp:index_sp+2] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+2], byCol=byCol)
        index_sp += 2
    if 'ANC' in features:
        feature_index['ANC'] = (index_sp, index_sp+16)
        segmented_spectrograms[:, index_sp:index_sp+4] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+4], byCol=byCol)
        segmented_spectrograms[:, index_sp+4:index_sp+8] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+4:index_sp+8], byCol=byCol)
        segmented_spectrograms[:, index_sp+8:index_sp+12] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+8:index_sp+12], byCol=byCol)
        segmented_spectrograms[:, index_sp+12:index_sp+16] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+12:index_sp+16], byCol=byCol)
        index_sp += 16   

    return segmented_spectrograms, segmented_gt, times, feature_index

def prepare_file(file, fs=10, features=['Q', 'omega', 'omega_l2', 'ANC'], start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False, byCol=False):
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

    segmented_spectrograms, segmented_gt, times, feature_index = compute_file(data, features=features, start_pt=start_pt, end_pt=end_pt, still_pt=still_pt, after_still_pt=after_still_pt, pool=pool, d=d, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1, byCol=byCol)

    spectrograms.append(segmented_spectrograms)
    gts.append(segmented_gt)
    
    spectrograms = np.concatenate(spectrograms, axis=0)
    gts = np.concatenate(gts, axis=0)

    print('----------------------------')
    print(f'sepctrograms:{spectrograms.shape}')
    print(f'gt:{gts.shape}')
    print(f'times:{times.shape}')

    return spectrograms, gts, times

def prepare_data(dir, fs=10, features=['Q', 'omega', 'omega_l2', 'ANC'], start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False, byCol=False):
    dataset = {}
    feature_index = None

    # iterate all people
    for person in os.listdir(dir):
        person_name = os.fsdecode(person)
        dataset[person_name] = []

        # iterate all files
        dir_p = os.path.join(dir, person_name)
        for file in os.listdir(dir_p):
            filename = os.fsdecode(file)
            action_name = filename.split("_")[0]

            if filename.endswith(".csv"): 
                print(os.path.join(dir_p, filename))
                
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

                segmented_spectrograms, segmented_gt, times, feature_index = compute_file(data, features=features, start_pt=start_pt, end_pt=end_pt, still_pt=still_pt, after_still_pt=after_still_pt, pool=pool, d=d, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1, byCol=byCol)
                feature_index = feature_index
                
                dataset[person_name].append({'action': action_name,
                                             'file_name': filename,
                                             'input': segmented_spectrograms,
                                             'gt': segmented_gt,
                                             'time': times,})
                
    dataset['feature_index'] = feature_index

    print('--------------------------------------')
    people = [key for key in dataset if key != 'feature_index']
    for i, person in enumerate(people):
        data = dataset[person]
        print(f'No.{i+1}, {person}, number of files: {len(data)}')

    # print(f'feature_index:{feature_index}')
    # exit()

    return dataset

def split_dataset(dataset, test_set=['m2', 'm5', 'm7', 'w1', 'w4'], features=['Q', 'omega', 'omega_l2', 'ANC'], ByAction=False, exclude={'w5': ['walk', 'run']}):
    if not ByAction:
        spectrograms_train, gts_train = [], []
        spectrograms_test, gts_test = [], []
    else:
        spectrograms_train, gts_train = {}, {}
        spectrograms_test, gts_test = {}, {}

    people = [key for key in dataset if key != 'feature_index']
    for i, person in enumerate(people):
        data = dataset[person]
        
        for j, file in enumerate(data):
            action = file['action']
            # print(f'person:{person}, action:{action}')
            if not (person in list(exclude.keys()) and action in exclude[person]):
                segmented_spectrograms_all, segmented_gt = file['input'], file['gt']

                # print(f'segmented_spectrograms_all:{segmented_spectrograms_all.shape}')
                segmented_spectrograms = []
                for i, feature in enumerate(features):
                    start, end = dataset['feature_index'][feature][0], dataset['feature_index'][feature][1]
                    segmented_spectrograms.append(segmented_spectrograms_all[:, start:end])
                segmented_spectrograms = np.concatenate(segmented_spectrograms, axis=1)
                # print(f'segmented_spectrograms:{segmented_spectrograms.shape}')
                # exit()

                if not ByAction:
                    if person not in test_set:
                        spectrograms_train.append(segmented_spectrograms)
                        gts_train.append(segmented_gt)
                    else:
                        spectrograms_test.append(segmented_spectrograms)
                        gts_test.append(segmented_gt)
                else: # action-wise
                    if person not in test_set:
                        if action not in spectrograms_train:
                            spectrograms_train[action] = []
                            gts_train[action] = []
                        spectrograms_train[action].append(segmented_spectrograms)
                        gts_train[action].append(segmented_gt)
                    else:
                        if action not in spectrograms_test:
                            spectrograms_test[action] = []
                            gts_test[action] = []
                        spectrograms_test[action].append(segmented_spectrograms)
                        gts_test[action].append(segmented_gt)

    if not ByAction:
        input_train, gt_train = np.concatenate(spectrograms_train, axis=0), np.concatenate(gts_train, axis=0)
        input_test, gt_test = np.concatenate(spectrograms_test, axis=0), np.concatenate(gts_test, axis=0)
    else: # action-wise
        input_train, gt_train = {}, {}
        input_test, gt_test = {}, {}

        # print(f'Training set.............')
        for k, v in spectrograms_train.items():
            input_train[k] = np.concatenate(spectrograms_train[k], axis=0)
            gt_train[k] = np.concatenate(gts_train[k], axis=0)
        
            # print(f'action: {k}')
            # print(f'spectrograms_train[{k}]:{input_train[k].shape}')
            # print(f'gt_train[{k}]:{gt_train[k].shape}')

        # print(f'Test set.............')
        for k, v in spectrograms_test.items():
            input_test[k] = np.concatenate(spectrograms_test[k], axis=0)
            gt_test[k] = np.concatenate(gts_test[k], axis=0)
        
            # print(f'action: {k}')
            # print(f'spectrograms_test[{k}]:{input_test[k].shape}')
            # print(f'gt_test[{k}]:{gt_test[k].shape}')

    return input_train, gt_train, input_test, gt_test

def prepare_data_anc(dir, fs=10, start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, visualize=True, test_set=['m2', 'm5', 'm7', 'w1', 'w4'], exclude={'w5': ['walk', 'run']}):
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    block_files = ['run_0514_1048.csv']

    method_ls = ['LMS', 'LMS+LS', 'RLS', 'LRLS']
    pred_test = {key:{} for key in method_ls}
    gt_test = {}
    mae_test = {key:{} for key in method_ls} # overall mae in each [method][action]
    relative_mae = {key:{} for key in method_ls} # each sample point relative mae in [method][action]

    # all sample points
    pred_ls = []
    gt_ls = []

    # iterate all people
    for person in os.listdir(dir):
        person_name = os.fsdecode(person)

        # iterate all files in test set
        if person_name in test_set:
            dir_p = os.path.join(dir, person_name)
            for file in os.listdir(dir_p):
                filename = os.fsdecode(file)
                action_name = filename.split("_")[0]
                
                if filename.endswith(".csv") and not (person_name in list(exclude.keys()) and action_name in exclude[person]):
                    print(os.path.join(dir_p, filename))

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
                    data_anc = data_aligned.copy() # data used in sml
                    data_anc = data_anc[still_pt+after_still_pt:]

                    outputs = sp.anc_process(data_anc, NTAPS=3, LEARNING_RATE=0.001, delta=1, lam_rls=0.9995, epsilon=1e-6, lam_lrls=0.9995)
                    # print(f"outputs:{outputs}")
                    pred, gt, mae = sp.auto_correlation(
                        data_anc,
                        outputs=outputs,
                        window=window_size/fs,
                        overlap=(window_size-stride)/fs,
                        visualize=False,
                        return_pgm=True
                    )

                    # calculate error
                    for i, method in enumerate(method_ls):
                        if action_name in pred_test[method]:
                            pred_test[method][action_name].extend(list(pred[method]))
                            if i == 0:
                                gt_test[action_name].extend(list(gt))
                            relative_mae[method][action_name].extend(list(abs(pred[method] - gt) / gt * 100))
                        else:
                            pred_test[method][action_name] = list(pred[method])
                            if i == 0:
                                gt_test[action_name] = list(gt)
                            relative_mae[method][action_name] = list(abs(pred[method] - gt) / gt * 100)

                    pred_ls.extend(list(pred))
                    gt_ls.extend(list(gt))
                    # break

    # calculate overall error
    action_ls = pred_test[next(iter(pred_test))].keys()
    if len(action_ls) == 4:
        action_ls = ['sit', 'stand', 'walk', 'run']

    for method in method_ls:
        pred_test[method] = {k : pred_test[method][k] for k in action_ls}
        gt_test = {k : gt_test[k] for k in action_ls}
        relative_mae[method] = {k : relative_mae[method][k] for k in action_ls}

    for i, method in enumerate(method_ls):
        print(f"{method}:")
        for action_name in action_ls:
            mae_test[method][action_name] = mean_absolute_error(gt_test[action_name], pred_test[method][action_name])
            avg_l1_loss = mae_test[method][action_name]
            avg_relative_mae = np.mean(relative_mae[method][action_name])
            print(f"{action_name} - L1 Loss: {avg_l1_loss:.4f} 1/min, E%: {avg_relative_mae:.4f}%")
        print()
    
    if visualize:
        vs.draw_learning_results_action(pred_test, gt_test, mae_test, models_name=method_ls, paper=True)
        vs.draw_learning_results_action_bar(mae_test, models_name=method_ls)
        vs.draw_learning_results_action_relative(relative_mae, sigma_num=1, models_name=method_ls)

# Legacy
# def prepare_action_data(dir, fs=10, features=['Q', 'omega', 'omega_l2', 'ANC'], start_pt=0, end_pt=-1, still_pt=300, after_still_pt=0, pool=1.0, d=0.05, window_size=128, stride=64, nperseg=128, noverlap=64, out_1=False, byCol=False):
#     sensor_names=['imu1','imu2']
#     cols = ['q_x', 'q_y', 'q_z', 'q_w']
#     omega_axes = ['omega_u', 'omega_v', 'omega_w']
#     ang_speed_cols = ['omega']
#     anc_methods = ['LMS', 'LMS+LS', 'RLS', 'LRLS']

#     q_col_ls = []
#     omega_col_ls = []
#     ang_speed_col_ls = []
#     anc_col_ls = []

#     for imu in sensor_names:
#         for col in cols:
#             q_col_ls.append(imu + "_" + col)
#         for col in omega_axes:
#             omega_col_ls.append(imu + "_" + col)
#         for col in ang_speed_cols:
#             ang_speed_col_ls.append(imu + "_" + col)

#     for anc_method in anc_methods:
#         for col in cols:
#             anc_col_ls.append(anc_method + "_" + col)

#     spectrograms, gts = {}, {}
#     # iterate all files
#     for file in os.listdir(dir):
#         filename = os.fsdecode(file)
#         if filename.endswith(".csv"): 
#             print(os.path.join(dir, filename))
#             action_name = filename.split("_")[0]

#             # print(filename)
#             # print(filename.split("_"))
#             # print(action_name)
#             # exit()
            
#             # load data
#             data = pd.read_csv(os.path.join(dir, filename))
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

#             data = data.iloc[start_pt:end_pt]

#             # align delay
#             data = sp.align_delay(data, delay=10)
#             data["Timestamp"] = pd.to_datetime(data["Timestamp"])
#             data = data.set_index("Timestamp")

#             # align IMU
#             q_corr = sp.Q_RANSAC(data[0:still_pt], pool=pool, d=d)
#             target, skew = 'imu1', 'imu2'
#             Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
#             Q_aligned = sp.align_quaternion(q_corr, Q_skew) # (sample num, 4)
#             data_aligned = data.copy()

#             for i, col in enumerate(cols):
#                 data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

#             # specify data range
#             data_sml = data_aligned.copy() # data used in sml
#             data_anc = data_aligned.copy() # data used in anc
#             data_sml = data_sml[still_pt+after_still_pt:]

#             # data_sml.loc[:, "Force"] = sp.butter_filter(data_sml["Force"], cutoff=1) # cutoff=0.66

#             # calculate omega features
#             if 'omega' in features:
#                 for imu in sensor_names:
#                     q = data_sml[[imu + "_" + "q_x", imu + "_" + "q_y", imu + "_" + "q_z", imu + "_" + "q_w"]].values # (sample num, 4)
#                     omega = sp.q_to_omega(q)
#                     ang_speed = sp.omega_to_AngSpeed(omega)
#                     for i, omega_axis in enumerate(omega_axes):
#                         data_sml.loc[:, imu + "_" + omega_axis] = omega[:, i]
                    
#                     data_sml.loc[:, imu + "_" + ang_speed_cols[0]] = ang_speed

#             # calculate ANC features
#             if 'ANC' in features:
#                 anc_outputs = sp.anc_process(data_anc, NTAPS=3, LEARNING_RATE=0.001, delta=1, lam_rls=0.9995, epsilon=1e-6, lam_lrls=0.9995)
#                 for anc_output in anc_outputs:
#                     method = anc_output['method']
#                     for col in cols:
#                         data_sml.loc[:, method + "_" + col] = anc_output[col][still_pt+after_still_pt:]

#             # Q: (sample_num, channel_num)
#             features_cols = []
#             if 'Q' in features:
#                 features_cols.extend(q_col_ls)
#             if 'omega' in features:
#                 features_cols.extend(omega_col_ls)
#             if 'omega_l2' in features:
#                 features_cols.extend(ang_speed_col_ls)
#             if 'ANC' in features:
#                 features_cols.extend(anc_col_ls)
#             Q = data_sml[features_cols].values

#             segmented_spectrograms, segmented_gt = sp.segment_data(Q, data_sml["Force"], window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out_1)

#             # min-Max normalization
#             index_sp = 0
#             if 'Q' in features:
#                 segmented_spectrograms[:, index_sp:index_sp+8] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+8], byCol=byCol)
#                 index_sp += 8
#             if 'omega' in features:
#                 segmented_spectrograms[:, index_sp:index_sp+6] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+6], byCol=byCol)
#                 index_sp += 6
#             if 'omega_l2' in features:
#                 segmented_spectrograms[:, index_sp:index_sp+2] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+2], byCol=byCol)
#                 index_sp += 2
#             if 'ANC' in features:
#                 segmented_spectrograms[:, index_sp:index_sp+4] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp:index_sp+4], byCol=byCol)
#                 segmented_spectrograms[:, index_sp+4:index_sp+8] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+4:index_sp+8], byCol=byCol)
#                 segmented_spectrograms[:, index_sp+8:index_sp+12] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+8:index_sp+12], byCol=byCol)
#                 segmented_spectrograms[:, index_sp+12:index_sp+16] = sp.normalize_spectrogram(segmented_spectrograms[:, index_sp+12:index_sp+16], byCol=byCol)
#                 index_sp += 16
                
#             # print(f'sepctrograms:{segmented_spectrograms.shape}')
#             # print(f'gt:{segmented_gt.shape}')
#             # exit()
#             if action_name in spectrograms:
#                 spectrograms[action_name].append(segmented_spectrograms)
#                 gts[action_name].append(segmented_gt)
#             else:
#                 spectrograms[action_name] = [segmented_spectrograms]
#                 gts[action_name] = [segmented_gt]
    
#     for k, v in spectrograms.items():
#         spectrograms[k] = np.concatenate(spectrograms[k], axis=0)
#         gts[k] = np.concatenate(gts[k], axis=0)

#         print(f'action: {k}')
#         print(f'sepctrograms[{k}]:{spectrograms[k].shape}')
#         print(f'gt[{k}]:{gts[k].shape}')

#     return spectrograms, gts