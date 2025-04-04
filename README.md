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
- Use learning_base.ipynb or run below (all arguments can be modified):

### Step.1 Generate training, test set:
```
# 1-D Spectrogram
python generate_dataset.py -f data/hamham -n dataset/hamham
```
```
# 2-D Spectrogram
python generate_dataset.py -f data/hamham -n dataset/hamham_out1 --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1
```

### Step.2 Train model:
```
# 1-D Spectrogram
python train.py --dataset_name hamham --model_name MLP_1D --visualize
```
```
# 2-D Spectrogram
python train.py --dataset_name hamham_out1 --model_name CNN_out1 --model_type CNN --visualize
```

### Step.3 Test model:
```
# 1-D Spectrogram
python test.py --dataset_name hamham --model_name MLP_1D --model_type MLP
```
```
# 2-D Spectrogram
python test.py --dataset_name hamham_out1 --model_name CNN_out1 --model_type CNN
```