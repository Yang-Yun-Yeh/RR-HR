from utils.visualize import *
from utils.FIR_filter import *
from utils.signal_process import *
from utils.preprocess import *
from utils.model import *
import utils.vision_transformer as VT

from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    # Test one file
    # action_name = "run_stand_2"
    # path_file = f'./data/8P/train/{action_name}.csv'

    # action_name = "walk_stand_0407_0858" # walk_stand_0407_0852, walk_stand_0407_0858, walk_stand_0407_0903
    # path_file = f'./data/2P/test/{action_name}.csv'

    # sit_0516_0501, sit_0516_0509
    # stand_0516_0542, stand_0516_0548
    # walk_0516_0644, *walk_0516_0650, *walk_0516_0701, walk_0516_0707, walk_0516_0713
    # **run_0516_0735, *run_0516_0741, run_0516_0752, *run_0516_0807, run_0516_0812
    action_name = "run_0516_0741"
    path_file = f'./data/5_16/kiri/{action_name}.csv'

    # action_name = "walk_0514_1043" # sit_0514_1030, walk_0514_1043, walk_0520_0730, *walk_0520_0901, walk_0520_1031
    # path_file = f'./data/8P/test/{action_name}.csv'

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MLP(65, 1, num_channels=16)
    # model = MLP_out1(num_freq_bins, num_time_steps, num_channels=num_channels)
    # model = CNN_out1_2(num_channels=num_channels)
    model = VT.ViTRegression(in_channels=num_channels, patch_size=(3, 3), emb_dim=256, mlp_dim=512, num_heads=8, device=device)
    model_name = "VT_7.75P_BC"
    model.load_state_dict(torch.load(f'./models/{str(model_name)}.pt'))
    
    avg_mse_loss, avg_l1_loss, preds = evaluate_model_file(model, file_loader, model_name=model_name, gt=gts_file, times=times_file, action_name=action_name)