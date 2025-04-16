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
# 1-D Spectrogram (dataset:hamham)
python generate_dataset.py -f data/hamham -n dataset/hamham_16_1D
```
```
# 2-D Spectrogram (dataset:hamham)
python generate_dataset.py -f data/hamham -n dataset/hamham_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1
```
```
# 1-D Spectrogram (dataset:2P)
python generate_dataset.py -f data/2P -n dataset/2P_16_1D
```
```
# 2-D Spectrogram (dataset:2P)
python generate_dataset.py -f data/2P -n dataset/2P_16_2D --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1
```

### Step.2 Train model:
```
# 1-D Spectrogram (dataset:hamham)
python train.py --dataset_name hamham_16_1D --model_name MLP_16_1D --visualize
```
```
# 2-D Spectrogram (dataset:hamham)
python train.py --dataset_name hamham_16_2D --model_name CNN_16_2D_2 --model_type CNN --visualize
```
```
# 2-D Spectrogram (dataset:2P)
python train.py --dataset_name 2P_16_2D --model_name MLP_2P_2D --visualize
```

### Step.3 Test model:
```
# 1-D Spectrogram (dataset:hamham)
python test.py --dataset_name hamham --model_name MLP_1D --model_type MLP
```
```
# 2-D Spectrogram (dataset:hamham)
python test.py --dataset_name hamham_16_2D --model_name MLP_16_2D --model_type MLP
```
```
# 2-D Spectrogram (dataset:2P)
python test.py --dataset_name 2P_16_2D --model_name MLP_2P_2D --model_type MLP
```