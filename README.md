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
python generate_dataset.py -f data/hamham -n dataset/hamham
```

### Step.2 Train model:
```
python train.py --dataset_name hamham --visualize
```

### Step.3 Test model:
```
python test.py --dataset_name hamham --model_name 0327_MLP --model_type MLP
```