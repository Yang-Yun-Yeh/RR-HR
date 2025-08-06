# Respiration Rate (RR)

## Environment
- Requirements in `cmd/rr.txt`

## Data collecting
- Use raspberry pi, IMUs, and RR band.
- Run `IoT/record.py` file to start record data (`code/record.py` in raspberry pi directory).
- Output file will store in `code/csv/.` in raspberry pi directory.

## ANC (FIR filter)
- Use ANC_action.ipynb # previous: whole_process.ipynb
- Find order:
```
 python test_order.py -m 20 --visualize --dataset_name 17P
 ```

## Spectrogram + Learning base
- Use learning_base_17P_32.ipynb or run below (all arguments can be modified):

### Step.1 Generate dataset:
```
# 2-D Spectrogram (dataset:17P)
python generate_dataset.py -f data/all -n dataset/17P --window_size 256 --stride 64 --nperseg 128 --noverlap 64 --out_1 --byCol --features Q omega omega_l2 ANC
```

### Step.2 Train model:
```
# 2-D Spectrogram (dataset:17P)
# MLP
python train.py --dataset_name 17P --model_name MLP_test --model_type MLP -b 8 --visualize --features Q ANC --test m2 m5 m7 w1 w4
```

### Step.3 Test model:
```
# 2-D Spectrogram (dataset:17P)
python test.py --dataset_name 17P --model_name MLP_test --model_type MLP --features Q ANC --test m2 m5 m7 w1 w4
```