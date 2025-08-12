import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils.visualize import *
from utils.FIR_filter import *
from utils.signal_process import *
from utils.preprocess import *
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate IMU dataset')
    
    # Path
    parser.add_argument('-f', '--dataset_folder', required=True, type=str, help='dataset directory')

    # Preprocess
    parser.add_argument("--fs", type=int, default=10)
    parser.add_argument("--start_pt", type=int, default=0)
    parser.add_argument("--end_pt", type=int, default=-1)
    parser.add_argument("--still_pt", type=int, default=300)
    parser.add_argument("--after_still_pt", type=int, default=0)
    parser.add_argument("--pool", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)

    # Dataset
    parser.add_argument('--test', nargs='+', default=['m2', 'm5', 'm7', 'w1', 'w4'])
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    target_folder = args.dataset_folder
    print(f"target_folder: {target_folder}")

    prepare_data_anc(target_folder,
                fs=args.fs,
                start_pt=args.start_pt,
                end_pt=args.end_pt,
                still_pt=args.still_pt,
                after_still_pt=args.after_still_pt,
                pool=args.pool,
                d=args.d,
                window_size=args.window_size,
                stride=args.stride,
                test_set=args.test,)