import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils.visualize import *
from utils.signal_process import *
from utils.model import *

import torch
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="training for predict RR task")
    parser.add_argument("--ckpt_dir", type=str, default="models", help="path to save checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="dataset/", help="path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="imu_dataset")
    
    
    # Model
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--model_type", type=str, default="MLP")
    
    # Training
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-n", "--num_epoch", type=int, default=100)
    parser.add_argument("--visualize", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Load dataset
    pkl_train = pickle.load(open(os.path.join(args.dataset_dir, f'{args.dataset_name}_train.pkl'), 'rb'))
    pkl_test = pickle.load(open(os.path.join(args.dataset_dir, f'{args.dataset_name}_test.pkl'), 'rb'))
    input_train, gt_train = pkl_train['input'], pkl_train['gt']
    input_test, gt_test = pkl_test['input'], pkl_test['gt']

    print('Preparing data......')
    print(f'Training for {len(input_train)} windows, shape: {input_train.shape}')
    print(f'Testing for {len(input_test)} windows, shape: {input_test.shape}')

    num_channels = input_train.shape[1]
    num_freq_bins = input_train.shape[2]
    num_time_steps = input_train.shape[3]

    dataset_train = IMUSpectrogramDataset(input_train, gt_train)
    dataset_test = IMUSpectrogramDataset(input_test, gt_test)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    if args.model_type == "MLP":
        # model = MLP(num_freq_bins, num_time_steps, num_channels=num_channels)
        model = MLP_out1(num_freq_bins, num_time_steps, num_channels=num_channels)
    elif args.model_type == "CNN":
        # model = CNN_1D(num_freq_bins, num_time_steps, num_channels=num_channels)
        # model = CNN_out1(num_channels=num_channels)
        # model = CNN_1D_2(num_channels=num_channels)
        model = CNN_out1_2(num_channels=num_channels)
    elif args.model_type == "BiLSTM":
        model = BiLSTM(num_freq_bins, num_time_steps, num_channels=num_channels)
    elif args.model_type == "GRU":
        model = GRU(num_freq_bins, num_time_steps, num_channels=num_channels)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now()
    if args.model_name == "model":
        model_name = now.strftime("%m%d_%H.%M.%S") # MM_DD_HH:mm:ss
    train_model(model, train_loader, test_loader, num_epochs=args.num_epoch, name=args.model_name, ckpt_dir=args.ckpt_dir, device=device,visualize=args.visualize)