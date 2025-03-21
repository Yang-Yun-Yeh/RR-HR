import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

        return x.squeeze(-1)  # (batch, time_steps)
    
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

def train_model(model, train_loader, test_loader, name=None, num_epochs=20, device="cuda", visualize=False):
    model.to(device)

    # Best loss
    l1_test_best = float('inf')
    
    # Loss functions
    criterion_mse = nn.MSELoss()  # Mean Squared Error (Regression Loss)
    criterion_l1 = nn.L1Loss()  # L1 Loss (Mean Absolute Error)
    mse_train_ls, l1_train_ls, mse_test_ls, l1_test_ls = [], [], [], []
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        avg_mse_loss_train = total_mse_loss_train / len(train_loader)
        avg_l1_loss_train = 60 * total_l1_loss_train / len(train_loader) # 1/min
        mse_train_ls.append(avg_mse_loss_train)
        l1_train_ls.append(avg_l1_loss_train)

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
        avg_mse_loss_test = total_mse_loss_test / len(test_loader)
        avg_l1_loss_test = 60 * total_l1_loss_test / len(test_loader)
        mse_test_ls.append(avg_mse_loss_test)
        l1_test_ls.append(avg_l1_loss_test)

        # Save model
        if avg_l1_loss_test < l1_test_best:
            l1_test_best = avg_l1_loss_test
            torch.save(model.state_dict(), f'./models/{str(name)}.pt')
            # print("Best model saved!")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {avg_mse_loss_train:.4f}, L1: {avg_l1_loss_train:.4f}, Test MSE: {avg_mse_loss_test:.4f}, L1: {avg_l1_loss_test:.4f}")

    # draw results
    if visualize:
        vs.draw_loss_epoch(mse_train_ls, l1_train_ls, mse_test_ls, l1_test_ls)

def evaluate_model(model, test_loader, device="cuda"):
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
    avg_mse_loss = total_mse_loss / num_batches
    avg_l1_loss = 60 * total_l1_loss / num_batches

    print(f"Evaluation Results - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min")
    return avg_mse_loss, avg_l1_loss

def evaluate_model_file(model, file_loader, device="cuda", gt=None, times=None, visualize=True, action_name=None):
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

    preds = {"Spectrogram + MLP": np.array(pred)}

    if visualize:
        vs.draw_learning_results(preds, 60 * gt, times, action_name)

    print(f"Evaluation Results - MSE Loss: {avg_mse_loss:.4f}, L1 Loss: {avg_l1_loss:.4f} 1/min")
    return avg_mse_loss, avg_l1_loss, preds