import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils.visualize import *
from utils.signal_process import *
from utils.preprocess import *
from utils.model import *

from torch.utils.data import Dataset, DataLoader

file_path = "./data/hamham/train/walk_stand_2.csv"

# 1-D spectrogram
window_size=128
stride=64
nperseg=128
noverlap=64
out1=True

# 2-D spectrogram
# window_size=256
# stride=64
# nperseg=128
# noverlap=64
# out1=True

spectrograms_file, gts_file, times_file = prepare_file(file_path, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out1)
dataset_file= IMUSpectrogramDataset(spectrograms_file, gts_file)
file_loader = DataLoader(dataset_file, batch_size=1, shuffle=False)

num_channels = spectrograms_file.shape[1]
num_freq_bins = spectrograms_file.shape[2]  # From computed spectrogram
num_time_steps = spectrograms_file.shape[3]  # From computed spectrogram

print(spectrograms_file[0].max())
print(spectrograms_file[0].min())