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
# 2-D Spectrogram (dataset:10P_16)
python generate_dataset.py -f data/10P -n dataset/10P_16 --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --features Q omega omega_l2
python generate_dataset.py -f data/10P -n dataset/10P_16 --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --action --features Q omega omega_l2
```
```
# 2-D Spectrogram-action (dataset:10P_32)
python generate_dataset.py -f data/10P -n dataset/10P_32 --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --features Q omega omega_l2 ANC
python generate_dataset.py -f data/10P -n dataset/10P_32 --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --action --features Q omega omega_l2 ANC
```
```
# 2-D Spectrogram-action (dataset:10P_ANC)
python generate_dataset.py -f data/10P -n dataset/10P_ANC --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --features ANC
python generate_dataset.py -f data/10P -n dataset/action/10P_ANC --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --action --features ANC
```
```
# 2-D Spectrogram-action (dataset:10P_Q)
python generate_dataset.py -f data/10P -n dataset/10P_Q --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --features Q
python generate_dataset.py -f data/10P -n dataset/action/10P_Q --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --action --features Q
```

### Step.2 Train model:
```
# 2-D Spectrogram (dataset:10P_16)
python train.py --dataset_name 10P_16 --model_name VT_10P_16_emt2 --model_type VT -b 8 --visualize
```
```
# 2-D Spectrogram (dataset:10P_32)
# ViT
python train.py --dataset_name 10P_32 --model_name VT_10P_32_emt2 --model_type VT -b 8 --visualize
```
```
# 2-D Spectrogram (dataset:10P_ANC)
# ViT
python train.py --dataset_name 10P_ANC --model_name VT_10P_ANC_emt2 --model_type VT -b 8 --visualize
```
```
# 2-D Spectrogram (dataset:10P_Q)
# ViT
python train.py --dataset_name 10P_Q --model_name VT_10P_Q_emt2 --model_type VT -b 8 --visualize
```

### Step.3 Test model:
```
# 2-D Spectrogram (dataset:10P_32)
python test.py --dataset_name 10P_32 --model_name VT_10P_32_emt2 --model_type VT
```

### Optional: Test model by actions
- Use learning_base_10P_32.ipynb Testing Actions part