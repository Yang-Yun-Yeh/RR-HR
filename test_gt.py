from utils.visualize import *
from utils.FIR_filter import *
from utils.signal_process import *
from utils.preprocess import *
from utils.model import *

from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    # Test one file
    action_name = "walk_stand_0407_0858" # walk_stand_0407_0852, walk_stand_0407_0858, walk_stand_0407_0903
    # sit,  stand, walk_stand, run_stand, mix
    # path_file = f'./data/hamham/test/{action_name}.csv'
    # path_file = f'./data/3_28/{action_name}.csv'
    path_file = f'./data/2P/test/{action_name}.csv'

    # 1-D spectrogram
    # window_size=128
    # stride=64
    # nperseg=128
    # noverlap=64
    # out1=True

    # 2-D spectrogram
    window_size=256
    stride=64
    nperseg=128
    noverlap=64
    out1=True

    spectrograms_file, gts_file, times_file = prepare_file(path_file, window_size=window_size, stride=stride, nperseg=nperseg, noverlap=noverlap, out_1=out1)
    dataset_file= IMUSpectrogramDataset(spectrograms_file, gts_file)
    file_loader = DataLoader(dataset_file, batch_size=1, shuffle=False)

    num_channels = spectrograms_file.shape[1]
    num_freq_bins = spectrograms_file.shape[2]  # From computed spectrogram
    num_time_steps = spectrograms_file.shape[3]  # From computed spectrogram

    # Load model
    # model = MLP(65, 1, num_channels=16)
    model = MLP_out1(num_freq_bins, num_time_steps, num_channels=num_channels)
    # model = CNN_out1_2(num_channels=num_channels)
    model_name = "MLP_2P_2D" # MLP_2P_2D, CNN_2P_2D_2
    model.load_state_dict(torch.load(f'./models/{str(model_name)}.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    avg_mse_loss, avg_l1_loss, preds = evaluate_model_file(model, file_loader, model_name=model_name, gt=gts_file, times=times_file, action_name=action_name)