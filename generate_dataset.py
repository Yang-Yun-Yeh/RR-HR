import os
import argparse
import pickle
import numpy as np
from utils.preprocess import *

def parse_args():
    parser = argparse.ArgumentParser(description='Generate IMU dataset')
    
    # Path
    parser.add_argument('-f', '--dataset_folder', required=True, type=str, help='dataset directory')
    parser.add_argument("-n", "--dataset_name", type=str, default='dataset/imu_dataset')

    # Preprocess
    parser.add_argument("--fs", type=int, default=10)
    parser.add_argument("--start_pt", type=int, default=0)
    parser.add_argument("--end_pt", type=int, default=-1)
    parser.add_argument("--still_pt", type=int, default=300)
    parser.add_argument("--after_still_pt", type=int, default=0)
    parser.add_argument("--pool", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=64)
    parser.add_argument("--out_1", action='store_true')

    # Type
    parser.add_argument("--action", action='store_true')
    
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    
    dataset_types = ['train', 'test']
    for tp in dataset_types:
        target_folder = os.path.join(args.dataset_folder, tp)
        print(f"target_folder: {target_folder}")
        # files = os.listdir(target_folder)
        
        if not args.action:
            spectrograms, gts = prepare_data(target_folder,
                                            fs=args.fs,
                                            start_pt=args.start_pt,
                                            end_pt=args.end_pt,
                                            still_pt=args.still_pt,
                                            after_still_pt=args.after_still_pt,
                                            pool=args.pool,
                                            d=args.d,
                                            window_size=args.window_size,
                                            stride=args.stride,
                                            nperseg=args.nperseg,
                                            noverlap=args.noverlap,
                                            out_1=args.out_1)
        else:
            spectrograms, gts = prepare_action_data(target_folder,
                                            fs=args.fs,
                                            start_pt=args.start_pt,
                                            end_pt=args.end_pt,
                                            still_pt=args.still_pt,
                                            after_still_pt=args.after_still_pt,
                                            pool=args.pool,
                                            d=args.d,
                                            window_size=args.window_size,
                                            stride=args.stride,
                                            nperseg=args.nperseg,
                                            noverlap=args.noverlap,
                                            out_1=args.out_1)

        pickle.dump(
            {'input': spectrograms, 'gt': gts},
            open(f'{args.dataset_name}_{tp}.pkl', 'wb')
        )