from utils.visualize import *
from utils.FIR_filter import *
from utils.signal_process import *
from utils.preprocess import *
from utils.model import *

from torch.utils.data import Dataset, DataLoader
import pickle

if __name__ == '__main__':
    # Load dataset
    dataset_dir = "dataset/action"
    dataset_name = "2P_16_2D" # 2P_16_1D, 2P_16_2D
    pkl_test = pickle.load(open(os.path.join(dataset_dir, f'{dataset_name}_test.pkl'), 'rb'))
    input_test, gt_test = pkl_test['input'], pkl_test['gt']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_key = next(iter(input_test))
    num_channels = input_test[first_key].shape[1]
    num_freq_bins = input_test[first_key].shape[2]
    num_time_steps = input_test[first_key].shape[3]

    print('Testing data......')
    for k, v in input_test.items():
        print(f'action: {k}')

    print()
    # 2-D spectrogram
    models_name = ['MLP_2P_2D', 'CNN_2P_2D_2', '0417_BiLSTM_2P_2D_lr_0.001', '0417_GRU_2P_2D_lr_0.001']
    models = [MLP_out1(num_freq_bins, num_time_steps, num_channels=num_channels),
            CNN_out1_2(num_channels=num_channels),
            BiLSTM(num_freq_bins, num_time_steps, num_channels=num_channels),
            GRU(num_freq_bins, num_time_steps, num_channels=num_channels)]

    for i in range(len(models_name)):
        models[i].load_state_dict(torch.load(f'./models/{str(models_name[i])}.pt'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_models_action_relative(models, input_test, gt_test, models_name=["MLP_2D", "CNN_2D", "BiLSTM_2D", "GRU_2D"], device=device, visualize=True)