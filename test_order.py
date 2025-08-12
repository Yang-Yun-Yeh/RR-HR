import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.signal_process import *
from utils.visualize import *

def parse_args():
    parser = argparse.ArgumentParser(description="compute dataset time")
    parser.add_argument("--ckpt_dir", type=str, default="models", help="path to save checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="data", help="path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="all")

    # Order
    parser.add_argument("-m", "--max_order", type=int, default=50)
    parser.add_argument("--visualize", action='store_true')

    # Dataset
    parser.add_argument('--test', nargs='+', default=['m2', 'm5', 'm7', 'w1', 'w4'])

    args = parser.parse_args()
    return args

# Least Square
def ls(x, d, m, compute_h=True):
    n = len(x)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if i - j >= 0:
                A[i][j] = x[i-j]
    
    if compute_h:
        # h_hat = np.linalg.inv(A.T @ A) @ A.T @ d
        # print(f'h_hat:{h_hat}')

        # LR with sklearn
        reg = LinearRegression(fit_intercept=False).fit(A, d)
        # print(f'coef:{reg.coef_}')

    if compute_h:
        return A, reg.coef_
    else:
        return A

def RMSE(error):
    return np.sqrt(np.sum(error**2) / len(error))

if __name__ == '__main__':
    args = parse_args()
    sensor_names=['imu1','imu2']
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    iter = 0

    # Dataset
    test_set=args.test
    exclude={'w5': ['walk', 'run']}

    # preprocess
    start_pt=0
    end_pt=-1
    still_pt=300
    pool=1.0
    d=0.05

    # find order
    M = args.max_order # 50
    p1 = 500 # 500
    p2 = 800 # 700
    p3 = 1100 # 900
    action_order = ['sit', 'stand', 'walk', 'run']
    m_best_action = {k: [] for k in action_order} #[action]


    # folder
    target_folder = str(f"./{args.dataset_dir}/{args.dataset_name}")
    print(f"target_folder: {target_folder}")

    # rmse_order_train = {m:[] for m in range(1, M+1)} # [order]
    # rmse_order_val = {m:[] for m in range(1, M+1)} # [order]

    # iterate all people
    for person in os.listdir(target_folder):
        person_name = os.fsdecode(person)

        # iterate all files in training set
        if person_name not in test_set:
            dir_p = os.path.join(target_folder, person_name)
            for file in os.listdir(dir_p):
                filename = os.fsdecode(file)
                action_name = filename.split("_")[0]
                
                if filename.endswith(".csv") and not (person_name in list(exclude.keys()) and action_name in exclude[person]):
                # if filename.endswith(".csv") and filename == 'run_stand_1.csv' and not (person_name in list(exclude.keys()) and action_name in exclude[person]):
                # sit_1.csv, sit_0407_0751.csv, walk_stand_4.csv, run_stand_1.csv
                    # print(os.path.join(dir_p, filename))

                    # if action_name == 'run':
                    if action_name in action_order:
                        print(os.path.join(dir_p, filename))
                        pass
                    else:
                        continue
                    
                    rmse_order_train = {m:[] for m in range(1, M+1)} # [order]
                    rmse_order_val = {m:[] for m in range(1, M+1)} # [order]
                    
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
                    data = align_delay(data, delay=10)
                    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

                    # align IMU
                    q_corr = Q_RANSAC(data[0:still_pt], pool=pool, d=d)
                    target, skew = 'imu1', 'imu2'
                    Q_skew = data[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()
                    Q_aligned = align_quaternion(q_corr, Q_skew) # (sample num, 4)
                    data_aligned = data.copy()

                    for i, col in enumerate(cols):
                        data_aligned[[skew + '_' + col]] = Q_aligned[:, i].reshape(-1, 1)

                    # specify data range
                    data_anc = data_aligned.copy() # data used in anc

                    N_file = len(data) - 1
                    if N_file >= p3:
                        for m in range(1, M+1):
                            rmse_train = []
                            rmse_val = []
                            for col in cols:
                                a_train = data_anc.loc[p1:p2, skew + '_' + col].values
                                d_train = data_anc.loc[p1:p2, target + '_' + col].values
                                a_val = data_anc.loc[p2:p3, skew + '_' + col].values
                                d_val = data_anc.loc[p2:p3, target + '_' + col].values

                                A_train, h_hat = ls(x=a_train, d=d_train, m=m)
                                A_val = ls(x=a_val, d=d_val, m=m, compute_h=False)
                                e_train = d_train - A_train @ h_hat
                                e_val = d_val - A_val @ h_hat

                                rmse_train.append(RMSE(e_train))
                                rmse_val.append(RMSE(e_val))

                                # print(f'm={m}, e_train={RMSE(e_train)}, e_val={RMSE(e_val)}')

                            rmse_order_train[m].append(np.mean(rmse_train))
                            rmse_order_val[m].append(np.mean(rmse_val))

                        # show order result
                        rmse_val_min = float('inf')
                        m_best = -1
                        rmse_train_ls, rmse_val_ls = [], []
                        for m in range(1, M+1):
                            rmse_train = np.mean(rmse_order_train[m])
                            rmse_val = np.mean(rmse_order_val[m])
                            rmse_train_ls.append(rmse_train)
                            rmse_val_ls.append(rmse_val)

                            if rmse_val < rmse_val_min:
                                rmse_val_min = rmse_val
                                m_best = m

                            # print(f'm={m}, rmse_train={rmse_train}, rmse_val={rmse_val}')

                        m_best_action[action_name].append(m_best)
                        print(f'm_best = {m_best}, rmse_val_min={rmse_val_min}\n')        

                        # if args.visualize:
                        #     draw_order_RMSE(rmse_train_ls, rmse_val_ls)
                                
                    # exit()
                    # if iter > 5:
                    #     break
                    # iter += 1
    
    # show order result
    # print()
    # rmse_val_min = float('inf')
    # m_best = -1
    # rmse_train_ls, rmse_val_ls = [], []
    # for m in range(1, M+1):
    #     rmse_train = np.mean(rmse_order_train[m])
    #     rmse_val = np.mean(rmse_order_val[m])
    #     rmse_train_ls.append(rmse_train)
    #     rmse_val_ls.append(rmse_val)

    #     if rmse_val < rmse_val_min:
    #         rmse_val_min = rmse_val
    #         m_best = m

    #     print(f'm={m}, rmse_train={rmse_train}, rmse_val={rmse_val}')

    # print(f'\n m_best = {m_best}, rmse_val_min={rmse_val_min}')        

    # if args.visualize:
    #     draw_order_RMSE(rmse_train_ls, rmse_val_ls)

    if args.visualize:
        draw_order_action(m_best_action, M)