import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import FIR_filter
import visualize

fs = 10

NTAPS = 10 # 30
LEARNING_RATE = 0.001 # 0.05
start_pt, end_pt = 50, -50

file_path = "./data_test/walk_stand_5.csv"
data = pd.read_csv(file_path)
data.columns = [
    "Timestamp",
    "imu1_q_x",
    "imu1_q_y",
    "imu1_q_z",
    "imu1_q_w",
    "imu2_q_x",
    "imu2_q_y",
    "imu2_q_z",
    "imu2_q_w",
    "Force",
    "RR",
]
sensor_names=['imu1','imu2']
cols = ['q_x', 'q_y', 'q_z', 'q_w']
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data = data.set_index("Timestamp")
data = data.iloc[start_pt:end_pt]

# d = data["imu1_q_x"].to_numpy()
# x = data["imu2_q_x"].to_numpy()
# plt.figure(1)
# plt.plot(d)

outputs_dict = {}
for col in cols:
    d = data[sensor_names[0] + '_' + col].to_numpy()
    x = data[sensor_names[1] + '_' + col].to_numpy() 
    f = FIR_filter.FIR_filter(np.zeros(NTAPS))
    y = np.empty(len(d))
    coefficients_history = np.zeros((len(d), NTAPS))

    for i in range((len(d))):
        ref_noise = x[i]
        canceller = f.filter(ref_noise)
        output_signal = d[i] - canceller
        f.lms(output_signal, LEARNING_RATE)
        coefficients_history[i, :] = f.coefficients
        y[i] = output_signal
    
    outputs_dict[col] = y

visualize.draw_anc_curve(data, outputs=outputs_dict)
visualize.draw_fir_coefficients_curve(coefficients_history, fs)

# plt.figure(2)
# plt.plot(y)
# plt.show()