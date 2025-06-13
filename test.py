import os
import argparse
import pickle
import numpy as np

from utils.preprocess import *
from utils.visualize import *
from utils.signal_process import *
from utils.model import *
import utils.vision_transformer as VT

import torch
from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='evalute IMU dataset (testing)')
    
    # Path
    parser.add_argument("--dataset_dir", type=str, default="dataset/", help="path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default="imu_dataset")

    # Model
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--model_type", type=str, default="MLP")
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load dataset
    pkl_test = pickle.load(open(os.path.join(args.dataset_dir, f'{args.dataset_name}_test.pkl'), 'rb'))
    input_test, gt_test = pkl_test['input'], pkl_test['gt']

    print('Testing data......')
    print(f'Testing shape: {input_test.shape}')

    num_channels = input_test.shape[1]
    num_freq_bins = input_test.shape[2]
    num_time_steps = input_test.shape[3]

    dataset_test = IMUSpectrogramDataset(input_test, gt_test)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    elif args.model_type == "VT":
        # model = VT.ViTRegression(in_channels=num_channels, patch_size=(3, 3), emb_dim=256, mlp_dim=512, device=device) # patch_size=(h, w)=(3, 3)
        model = VT.ViTRegression(in_channels=num_channels, patch_size=(3, 3), emb_dim=256, mlp_dim=512, num_heads=8, device=device) # patch_size=(h, w)=(3, 3)

    model.load_state_dict(torch.load(f'./models/{args.model_name}.pt'))

    # Evaluate model in whole testing set
    mse, mae = evaluate_model(model, test_loader, device=device)