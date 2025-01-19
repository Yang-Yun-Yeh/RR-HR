# Respiration Rate (RR) & Heart Rate (HR)

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
