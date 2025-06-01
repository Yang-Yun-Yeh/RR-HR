# Respiration Rate (RR)

## Environment
- Requirements in `cmd/rr.txt`

## Data collecting
- Use raspberry pi, IMUs, and RR band.
- Run `IoT/daq.py` file to start record data (`code/daq.py` in raspberry pi directory).
- Output file will store in `code/csv/.` in raspberry pi directory.

## Visualization
- Use visualize_data.ipynb or run
```
python visualize_data.py
```

## ANC (FIR filter)
- Use whole_process.ipynb

## Spectrogram + Learning base
- Use learning_base_16.ipynb or run below (all arguments can be modified):

### Step.1 Generate training, test set:
```
# 1-D Spectrogram (dataset:2P)
python generate_dataset.py -f data/2P -n dataset/2P_16_1D
```
```
# 2-D Spectrogram (dataset:2P)
python generate_dataset.py -f data/2P -n dataset/2P_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1
```
```
# 2-D Spectrogram (dataset:8P)
python generate_dataset.py -f data/8P -n dataset/8P_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1
```
```
# 2-D Spectrogram-action (dataset:8P)
python generate_dataset.py -f data/8P -n dataset/action/8P_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --action
```


### Step.2 Train model:
```
# 2-D Spectrogram (dataset:8P)
# MLP
python train.py --dataset_name 8P_16_2D --model_name MLP_8P_2D --model_type MLP -b 8 --visualize
# CNN
python train.py --dataset_name 8P_16_2D --model_name CNN_8P_2D --model_type CNN -b 8 --visualize
# BiLSTM
python train.py --dataset_name 8P_16_2D --model_name BiLSTM_8P_2D --model_type BiLSTM -b 8 --visualize
# GRU
python train.py --dataset_name 8P_16_2D --model_name GRU_8P_2D --model_type GRU -b 8 --visualize
# ViT
python train.py --dataset_name 7.75P_BC --model_name VT_7.75P_BC_emht2 --model_type VT -b 8 --visualize
```

### Step.3 Test model:
```
# 2-D Spectrogram (dataset:2P)
python test.py --dataset_name 2P_16_2D --model_name MLP_2P_2D --model_type MLP
```

### Optional: Test model by actions
```
# Generate dataset with action name info.
python generate_dataset.py -f data/8P -n dataset/action/8P_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --action
```
- Use learning_base_16_8P.ipynb Testing Actions part