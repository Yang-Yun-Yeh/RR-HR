from utils.visualize import *
from utils.FIR_filter import *
from utils.signal_process import *
from utils.preprocess import *
from utils.model import *

from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    # Load model
    model = MLP(65, 1, num_channels=8)
    model_name = "0327_MLP"
    model.load_state_dict(torch.load(f'./models/{str(model_name)}.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test one file
    action_name = "run_stand_3"
    # sit,  stand, walk_stand, run_stand, mix
    path_file = f'./data/hamham/test/{action_name}.csv'
    # action_name = "walk_stand_0328_0236"
    # path_file = f'./data/3_28/{action_name}.csv'

    spectrograms_file, gts_file, times_file = prepare_file(path_file)
    # print(times_file)
    dataset_file= IMUSpectrogramDataset(spectrograms_file, gts_file)
    file_loader = DataLoader(dataset_file, batch_size=1, shuffle=False)

    avg_mse_loss, avg_l1_loss, preds = evaluate_model_file(model, file_loader, gt=np.reshape(gts_file, (-1)), times=times_file, action_name=action_name)