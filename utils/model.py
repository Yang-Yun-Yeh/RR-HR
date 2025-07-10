import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import copy
import pickle

try:
     from . import visualize as vs
except:
     import visualize as vs

class MLP(nn.Module):
    def __init__(self, num_freq_bins, num_time_steps, num_channels=8, hidden_dim=512): # hidden_dim=128
        super(MLP, self).__init__()

        input_dim = num_channels * num_freq_bins  # Flattened input per time step

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)  # Output one value per time step
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape

        # Flatten channel & frequency bins at each time step
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)  # (batch, time_steps, features)

        # Pass through NN
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # (batch, time_steps, 1)

        # print(f'x.shape: {x.shape}')
        # print(f'x.squeeze(-1): {x.squeeze(-1).shape}')
        # exit()

        return x.squeeze(-1)  # (batch, time_steps)

# conv1_out=32, hidden_dim=64
class CNN_1D(nn.Module):
    def __init__(self, num_freq_bins, num_time_steps=1, num_channels=8, conv1_out=32, hidden_dim=64):
        super(CNN_1D, self).__init__()

        # 1D Convolution over frequency bins (treating time_steps as a single channel)
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=conv1_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global Pooling over frequency bins
        self.fc = nn.Linear(hidden_dim, 1)  # Fully connected layer for output

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape

        x = x.squeeze(-1)  # Remove time_steps dimension -> (batch, channels, freq_bins)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.global_pool(x).squeeze(-1)  # Reduce frequency bins -> (batch, hidden_dim)
        x = self.fc(x).squeeze(-1)  # Fully connected output -> (batch,)

        return x.unsqueeze(-1)  # Final shape: (batch, time_steps)
    
class CNN_1D_2(nn.Module):
    def __init__(self, num_channels=16, dropout_rate=0.3):
        super(CNN_1D_2, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to reduce freq_bins → 1
        )

        self.fc = nn.Linear(256, 1)  # Final output

    def forward(self, x):
        # x shape: (batch_size, 16, freq_bins, 1)
        x = x.squeeze(-1)  # → (batch_size, 16, freq_bins)

        x = self.network(x)  # → (batch_size, 256, 1)
        x = x.squeeze(-1)    # → (batch_size, 256)
        x = self.fc(x)       # → (batch_size, 1)

        return x  # (batch_size, 1)

class MLP_out1(nn.Module):
    def __init__(self, num_freq_bins, num_time_steps, num_channels=8, hidden_dim=512): # hidden_dim=128
        super(MLP_out1, self).__init__()

        input_dim = num_channels * num_freq_bins * num_time_steps  # Flattened input

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output one global value

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten all but batch dimension
        x = x.view(batch_size, -1)  # (batch_size, features)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # (batch_size, 1)

        # print(f'x.shape: {x.shape}')
        # exit()

        return x  # (batch_size, 1)
     
class CNN_out1(nn.Module):
    def __init__(self, num_channels=8, hidden_dim=64):
        super(CNN_out1, self).__init__()

        # 2D CNN feature extractor (operates on (freq_bins, time_steps))
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling to (1,1) per channel
        self.fc = nn.Linear(64, 1)  # Fully connected layer for output

    def forward(self, x):
        batch_size = x.shape[0]  # (batch_size, 8, freq_bins, time_steps)
        # print(f'x.shape: {x.shape}')
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # Reduce to (batch_size, 64)
        # x = self.fc(x).squeeze(-1)  # Fully connected output -> (batch_size, 1)
        x = self.fc(x)
        # print(f'x.shape: {x.shape}')
        # exit()

        return x
    
class CNN_out1_2(nn.Module):
    def __init__(self, num_channels=16, dropout_rate=0.3):
        super(CNN_out1_2, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, 256, 1, 1)
        )

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # Input shape: (batch_size, num_channels, freq_bins, time_steps)
        x = self.features(x)               # → (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)          # → (batch_size, 256)
        x = self.fc(x)                     # → (batch_size, 1)
        return x

class BiLSTM(nn.Module):
    def __init__(self, num_freq_bins, num_time_steps, num_channels=8, hidden_dim=512):
        super(BiLSTM, self).__init__()

        input_dim = num_channels * num_freq_bins  # Flattened input per time step

        # BiLSTM layer
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Adjusted for BiLSTM output
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape

        # Flatten channel & frequency bins at each time step
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)  # (batch_size, time_steps, features)

        # Pass through BiLSTM
        x, _ = self.bilstm(x)  # (batch_size, time_steps, hidden_dim * 2)
        x = self.relu(x)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (batch_size, time_steps, 1)
        x = x[:, -1, :]

        return x  # (batch_size, 1)

class GRU(nn.Module):
    def __init__(self, num_freq_bins, num_time_steps, num_channels=8, hidden_dim=512):
        super(GRU, self).__init__()

        input_dim = num_channels * num_freq_bins  # Flattened input per time step

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.shape

        # Flatten channel & frequency bins at each time step
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, -1)  # (batch_size, time_steps, features)

        # Pass through GRU
        x, _ = self.gru(x)  # (batch_size, time_steps, hidden_dim)
        x = self.relu(x)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (batch_size, time_steps, 1)
        x = x[:, -1, :]

        return x  # (batch_size, 1)

# class CNN_GRU(nn.Module):
#     def __init__(self, num_freq_bins, num_time_steps=1, num_channels=8, hidden_dim=64, output_dim=1):
#         super(CNN_GRU, self).__init__()

#         # 1D Convolution over frequency bins (treating time_steps as a single channel)
#         self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global Pooling over frequency bins

#         # GRU expects input as (batch, seq_len, input_dim)
#         self.gru = nn.GRU(64, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
#         self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer

#     def forward(self, x):
#         batch_size, channels, freq_bins, time_steps = x.shape  # (batch, channels, freq_bins, time_steps)

#         x = x.squeeze(-1)  # Remove time_steps -> (batch, channels, freq_bins)

#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)

#         x = self.global_pool(x).squeeze(-1)  # Reduce frequency bins -> (batch, 64)

#         x = x.unsqueeze(1)  # Add seq_len=1 -> (batch, 1, 64)

#         x, _ = self.gru(x)  # (batch, 1, hidden_dim)
#         x = self.relu(x)

#         x = self.fc(x[:, -1, :])  # Take last GRU output -> (batch, output_dim)

#         return x  # Final shape: (batch, output_dim)
    
# class CNN_BiLSTM(nn.Module):
#     def __init__(self, num_freq_bins, num_time_steps, num_channels=8, hidden_dim=64):
#         super(CNN_BiLSTM, self).__init__()

#         # CNN feature extractor (per time step)
#         self.cnn = nn.Sequential(
#             nn.Conv1d(num_channels, 32, kernel_size=3, padding=1),  # Conv on freq axis
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         # Compute CNN output shape dynamically
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, num_channels, num_freq_bins)
#             cnn_out_shape = self.cnn(dummy_input).shape
#             self.cnn_feature_dim = cnn_out_shape[1]  # Extracted feature dimension

#         # BiLSTM for sequence modeling (maintains time_steps)
#         self.lstm = nn.LSTM(input_size=self.cnn_feature_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

#         # Fully connected layer for regression
#         self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

#     def forward(self, x):
#         batch_size, channels, freq_bins, time_steps = x.shape
#         # print(f'1: (batch_size, channels, freq_bins, time_steps) {x.shape}')

#         # Reshape to apply CNN independently per time step
#         x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, freq_bins)
#         x = x.reshape(batch_size * time_steps, channels, freq_bins)  # Flatten batch & time for CNN

#         # Apply CNN (per time step)
#         x = self.cnn(x)  # (batch*time_steps, cnn_feature_dim, freq_bins)
#         x = torch.mean(x, dim=2)  # Global average pooling over freq_bins (batch*time_steps, cnn_feature_dim)

#         # Reshape back to (batch, time_steps, features)
#         x = x.view(batch_size, time_steps, -1)

#         # BiLSTM for sequence modeling
#         print(f'lstm input: {x.shape}')
#         x, _ = self.lstm(x)  # (batch, time_steps, hidden_dim*2)

#         # Fully connected output per time step
#         x = self.fc(x).squeeze(-1)  # (batch, time_steps)
#         # print(f'6: (batch, time_steps), {x.shape}')

#         return x
    
class IMUSpectrogramDataset(Dataset):
    def __init__(self, spectrograms, respiration_rates):
        """
        Args:
            spectrograms: (num_windows, 8, freq_bins, time_steps)
            respiration_rates: (num_windows, time_steps)
        """
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
        self.respiration_rates = torch.tensor(respiration_rates, dtype=torch.float32)
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.respiration_rates[idx]

def train_model(model, train_loader, test_loader, name=None, ckpt_dir='models', num_epochs=20, device="cuda", visualize=False):
    model.to(device)

    # Best loss
    # l1_test_best = float('inf')
    l1_train_best = float('inf')
    model_best = None
    
    # Loss functions
    criterion_mse = nn.MSELoss()  # Mean Squared Error (Regression Loss)
    criterion_l1 = nn.L1Loss()  # L1 Loss (Mean Absolute Error)
    mse_train_ls, l1_train_ls, mse_test_ls, l1_test_ls = [], [], [], []
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5) # lr decay

    # model.train()
    for epoch in range(num_epochs):
        # train
        model.train()
        total_mse_loss_train = 0
        total_l1_loss_train = 0
        for spectrograms, respiration_rates in train_loader:
            spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

            # print(f'respiration_rates: {respiration_rates}')

            optimizer.zero_grad()
            outputs = model(spectrograms)  # Shape: (batch, time_steps)
            # print(f'outputs: {outputs}')
            
            mse_loss = criterion_mse(outputs, respiration_rates)
            l1_loss = criterion_l1(outputs, respiration_rates)

            # eta = 5
            # l1_loss = torch.clamp(l1_loss,max = eta)
           
            if not l1_loss.isnan(): # no nan values 
                 # mse_loss.backward()
                l1_loss.backward()
                optimizer.step()
            
                total_mse_loss_train += mse_loss.item()
                total_l1_loss_train += l1_loss.item()

        avg_mse_loss_train = (60**2) * total_mse_loss_train / len(train_loader)
        avg_l1_loss_train = 60 * total_l1_loss_train / len(train_loader) # 1/min
        mse_train_ls.append(avg_mse_loss_train)
        l1_train_ls.append(avg_l1_loss_train)

        # scheduler.step(l1_loss)

        # eval
        model.eval()
        total_mse_loss_test = 0
        total_l1_loss_test = 0

        with torch.no_grad():  # No gradient computation during evaluation
            for spectrograms, respiration_rates in test_loader:
                spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

                # Forward pass
                outputs = model(spectrograms)  # Shape: (batch, time_steps)

                # Compute losses
                mse_loss = criterion_mse(outputs, respiration_rates)
                l1_loss = criterion_l1(outputs, respiration_rates)

                if not l1_loss.isnan(): # no nan values 
                    total_mse_loss_test += mse_loss.item()
                    total_l1_loss_test += l1_loss.item()

        # Compute average loss over all batches
        avg_mse_loss_test = (60**2) * total_mse_loss_test / len(test_loader)
        avg_l1_loss_test = 60 * total_l1_loss_test / len(test_loader)
        mse_test_ls.append(avg_mse_loss_test)
        l1_test_ls.append(avg_l1_loss_test)

        # Save model
        # if avg_l1_loss_test < l1_test_best:
            # l1_test_best = avg_l1_loss_test
        if avg_l1_loss_train < l1_train_best:
            l1_train_best = avg_l1_loss_train
            # torch.save(model.state_dict(), f'./{ckpt_dir}/{str(name)}.pt')
            model_best = copy.deepcopy(model.state_dict())
            # print("Best model saved!")
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {avg_mse_loss_train:.4f}, L1: {avg_l1_loss_train:.4f}, Test MSE: {avg_mse_loss_test:.4f}, L1: {avg_l1_loss_test:.4f}, lr:{scheduler.get_last_lr()}")
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {avg_mse_loss_train:.4f}, L1: {avg_l1_loss_train:.4f}, Test MSE: {avg_mse_loss_test:.4f}, L1: {avg_l1_loss_test:.4f}")

    # Save model
    torch.save(model_best, f'./{ckpt_dir}/{str(name)}.pt')

    # draw results
    if visualize:
        # pass
        vs.draw_loss_epoch(mse_train_ls, l1_train_ls, mse_test_ls, l1_test_ls, name=name)

def evaluate_model(model, test_loader, model_name='model', device="cuda"):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    total_mse_loss = 0
    total_l1_loss = 0
    num_batches = 0

    with torch.no_grad():  # No gradient computation during evaluation
        for spectrograms, respiration_rates in test_loader:
            spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

            # Forward pass
            outputs = model(spectrograms)  # Shape: (batch, time_steps)

            # Compute losses
            mse_loss = criterion_mse(outputs, respiration_rates)
            l1_loss = criterion_l1(outputs, respiration_rates)

            total_mse_loss += mse_loss.item()
            total_l1_loss += l1_loss.item()
            num_batches += 1

    # Compute average loss over all batches
    avg_mse_loss = (60**2) * total_mse_loss / num_batches
    avg_l1_loss = 60 * total_l1_loss / num_batches

    print(f"{model_name} Evaluation Results - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min")
    return avg_mse_loss, avg_l1_loss

def evaluate_model_file(model, file_loader, model_name, device="cuda", gt=None, times=None, visualize=True, action_name=None):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    total_mse_loss = 0
    total_l1_loss = 0
    num_batches = 0

    pred = []

    with torch.no_grad():  # No gradient computation during evaluation
        for spectrograms, respiration_rates in file_loader:
            spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

            # Forward pass
            outputs = model(spectrograms)  # Shape: (batch, time_steps)
            pred.append(60 * outputs.cpu().numpy()[0][0])

            # Compute losses
            mse_loss = criterion_mse(outputs, respiration_rates)
            l1_loss = criterion_l1(outputs, respiration_rates)

            total_mse_loss += mse_loss.item()
            total_l1_loss += l1_loss.item()
            num_batches += 1

    # Compute average loss over all batches
    avg_mse_loss = total_mse_loss / num_batches
    avg_l1_loss = 60 * total_l1_loss / num_batches

    pred =  np.array(pred)
    preds = {model_name: pred}
    r2 = r2_score(60 * gt, pred) # R-squared

    if visualize:
        vs.draw_learning_results(preds, 60 * gt, times, action_name)

    print(f"Evaluation Results - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min, R\u00b2:{r2:.4f}")
    return avg_mse_loss, avg_l1_loss, preds

def evaluate_models_file(models, file_loader, models_name, device="cuda", gt=None, times=None, visualize=True, action_name=None):
    avg_mse_loss_ls = []
    avg_l1_loss_ls = []
    preds = {}

    for i in range(len(models)):
        model = models[i]
        model_name = models_name[i]

        model.to(device)
        model.eval()  # Set model to evaluation mode

        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

        total_mse_loss = 0
        total_l1_loss = 0
        num_batches = 0

        pred = []

        with torch.no_grad():  # No gradient computation during evaluation
            for spectrograms, respiration_rates in file_loader:
                spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

                # Forward pass
                outputs = model(spectrograms)  # Shape: (batch, time_steps)
                pred.append(60 * outputs.cpu().numpy()[0][0])

                # Compute losses
                mse_loss = criterion_mse(outputs, respiration_rates)
                l1_loss = criterion_l1(outputs, respiration_rates)

                total_mse_loss += mse_loss.item()
                total_l1_loss += l1_loss.item()
                num_batches += 1

        # Compute average loss over all batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_l1_loss = 60 * total_l1_loss / num_batches
        
        avg_mse_loss_ls.append(avg_mse_loss)
        avg_l1_loss_ls.append(avg_l1_loss)

        pred =  np.array(pred)
        preds[model_name] = pred
        r2 = r2_score(60 * gt, pred) # R-squared

        print(f"{model_name} Evaluation Results - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min, R\u00b2:{r2:.4f}")

    if visualize:
        vs.draw_learning_results(preds, 60 * gt, times, action_name)

    return avg_mse_loss, avg_l1_loss, preds

def evaluate_models_action(models, input_test, gt_test, models_name, device="cuda", visualize=False):
    pred_test = {key:{} for key in models_name}
    mae_test = {key:{} for key in models_name}
    
    for i in range(len(models)):
        model = models[i]
        model_name = models_name[i]

        model.to(device)
        model.eval()  # Set model to evaluation mode

        print(models_name[i])
        for k, v in input_test.items():
            # print(f'action: {k}')
            dataset_test = IMUSpectrogramDataset(input_test[k], gt_test[k])
            test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

            criterion_mse = nn.MSELoss()
            criterion_l1 = nn.L1Loss()

            total_mse_loss = 0
            total_l1_loss = 0
            num_batches = 0

            with torch.no_grad():  # No gradient computation during evaluation
                for spectrograms, respiration_rates in test_loader:
                    spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

                    # Forward pass
                    outputs = model(spectrograms)  # Shape: (batch, time_steps)
                    if k in pred_test[model_name]:
                        pred_test[model_name][k].append(outputs.cpu().numpy()[0][0])
                    else:
                        pred_test[model_name][k] = [outputs.cpu().numpy()[0][0]]
                
                    # Compute losses
                    mse_loss = criterion_mse(outputs, respiration_rates)
                    l1_loss = criterion_l1(outputs, respiration_rates)

                    total_mse_loss += mse_loss.item()
                    total_l1_loss += l1_loss.item()
                    num_batches += 1

            # Compute average loss over all batches
            avg_mse_loss = total_mse_loss / num_batches
            avg_l1_loss = 60 * total_l1_loss / num_batches
            mae_test[model_name][k] = avg_l1_loss

            print(f"{k} - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min")
        print()
        
    if visualize:
        vs.draw_learning_results_action(pred_test, gt_test, mae_test, models_name=models_name)
        vs.draw_learning_results_action_bar(mae_test, models_name=models_name)

def evaluate_models_action_relative(models, input_test, gt_test, models_name, device="cuda", visualize=False):
    pred_test = {key:{} for key in models_name}
    relative_mae = {key:{} for key in models_name} # each sample point relative mae in [method][action]
    mae_test = {key:{} for key in models_name} # overall mae in each [method][action]
    overall_relative_mae = {key:{} for key in models_name} # overall relative mae in [method][action]
    
    for i in range(len(models)):
        model = models[i]
        model_name = models_name[i]

        model.to(device)
        model.eval()  # Set model to evaluation mode

        print(models_name[i])
        for k, v in input_test.items():
            # print(f'action: {k}')
            dataset_test = IMUSpectrogramDataset(input_test[k], gt_test[k])
            test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

            criterion_mse = nn.MSELoss()
            criterion_l1 = nn.L1Loss()

            total_mse_loss = 0
            total_l1_loss = 0
            num_batches = 0

            with torch.no_grad():  # No gradient computation during evaluation
                for spectrograms, respiration_rates in test_loader:
                    spectrograms, respiration_rates = spectrograms.to(device), respiration_rates.to(device)

                    # Forward pass
                    outputs = model(spectrograms)  # Shape: (batch, time_steps)
                    outputs_np = outputs.cpu().numpy()[0][0]
                    gts_np = respiration_rates.cpu().numpy()[0][0]
                    
                    if k in pred_test[model_name]:
                        pred_test[model_name][k].append(outputs_np)
                        relative_mae[model_name][k].append(abs(outputs_np - gts_np) / gts_np * 100)
                    else:
                        pred_test[model_name][k] = [outputs_np]
                        relative_mae[model_name][k] = [abs(outputs_np - gts_np) / gts_np * 100]
                
                    # Compute losses
                    mse_loss = criterion_mse(outputs, respiration_rates)
                    l1_loss = criterion_l1(outputs, respiration_rates)

                    total_mse_loss += mse_loss.item()
                    total_l1_loss += l1_loss.item()
                    num_batches += 1

            # Compute average loss over all batches
            avg_mse_loss = total_mse_loss / num_batches
            avg_l1_loss = 60 * total_l1_loss / num_batches
            avg_relative_mae = np.mean(relative_mae[model_name][k])
            mae_test[model_name][k] = avg_l1_loss
            overall_relative_mae[model_name][k] = avg_relative_mae

            print(f"{k} - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min E%: {avg_relative_mae:.4f}%")
        print()
        
    if visualize:
        vs.draw_learning_results_action_relative(relative_mae, sigma_num=1, models_name=models_name)