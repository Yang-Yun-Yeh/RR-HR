import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

def draw_imu_curve(imu_data, sensor_names=['imu1','imu2'], overlap=False, show_gt=False):
    cols = ['q_x', 'q_y', 'q_z', 'q_w']
    titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'yellow']

    legend_handle_ls = []
    row_num = len(sensor_names)
    if not overlap:
        if not show_gt:
            row_num = row_num
        else:
            row_num += 1
    else:
        if not show_gt:
            row_num += 1
        else:
            row_num += 2
    
    # fig, axes = plt.subplots(row_num, len(titles), figsize=(20,10), constrained_layout=True)
    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles)):
            # print(f'{i}, {key}, {cols[i]}')
            # axes[k][i%4].plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            # axes[k][i%4].set_title(titles[i])

            ax = fig.add_subplot(spec[k, i%4])
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])

        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    if overlap:
        for i in range(len(titles)):
            ax = fig.add_subplot(spec[2, i%4])
            for k, key in enumerate(sensor_names):
                ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
                ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])

    if show_gt:
        if not overlap:
            ax = fig.add_subplot(spec[2, :])
        else:
            ax = fig.add_subplot(spec[3, :])
        ax.plot(time_ls, imu_data["Force"], color=colors[2])
        ax.set_title('Respiration GT Force')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))
            

        

    fig.legend(handles=legend_handle_ls, loc="upper right")
    
    # plt.savefig('output/figures/a.png')
    plt.show()

def draw_anc_curve(imu_data, outputs, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w']):
    titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'purple']

    legend_handle_ls = []
    
    row_num = 3
    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    # show overlap
    for i in range(len(titles)):
        ax = fig.add_subplot(spec[0, i%4])
        for k, key in enumerate(sensor_names):
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
        ax.set_title(titles[i])
    for k, key in enumerate(sensor_names):
        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    # show anc results
    for i in range(len(cols)):
        ax = fig.add_subplot(spec[1, i%4])
        ax.plot(time_ls, outputs[cols[i]], color=colors[3])
        ax.set_title(titles[i])
    legend_handle_ls.append(Line2D([0], [0], label='ANC output', color=colors[3]))

    # show gt
    ax = fig.add_subplot(spec[2, :])
    ax.plot(time_ls, imu_data["Force"], color=colors[2])
    ax.set_title('Respiration GT Force')
    legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))
            

    fig.legend(handles=legend_handle_ls, loc="upper right")
    
    plt.show()

def draw_anc_curve_multi(imu_data, outputs, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w']):
    titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']
    # 1:'lightsteelblue', 2:'palegreen', 3:'yellow', 4:'plum'
    legend_handle_ls = []
    
    row_num = 3
    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    # show overlap
    for i in range(len(titles)):
        ax = fig.add_subplot(spec[0, i%4])
        for k, key in enumerate(sensor_names):
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
        ax.set_title(titles[i])
    for k, key in enumerate(sensor_names):
        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    # show anc results
    for i in range(len(cols)):
        ax = fig.add_subplot(spec[1, i%4])
        for k in range(len(outputs)):
            ax.plot(time_ls, outputs[k][cols[i]], color=colors[3+k])
        ax.set_title(titles[i])
    for k in range(len(outputs)):
        legend_handle_ls.append(Line2D([0], [0], label=outputs[k]['method'], color=colors[3+k]))

    # show gt
    ax = fig.add_subplot(spec[2, :])
    ax.plot(time_ls, imu_data["Force"], color=colors[2])
    ax.set_title('Respiration GT Force')
    legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))
            

    fig.legend(handles=legend_handle_ls, loc="upper right")

    plt.show()

def draw_fir_coefficients_curve(imu_data, coefficients, cols=['q_x', 'q_y', 'q_z', 'q_w']):
    titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']
    
    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, col in enumerate(cols):
        coeffs = coefficients[col]
        NTAPS = coeffs.shape[1]
        colors = plt.cm.get_cmap('tab10', NTAPS)
        
        ax = axs[i]
        
        legend_handle_ls = []
        
        for j in range(NTAPS):
            line, = ax.plot(time_ls, coeffs[:, j], color=colors(j), label=f'Tap {j+1}')
            legend_handle_ls.append(line)
        
        ax.set_title('FIR Coefficients Over Time - ' + titles[i])
        ax.legend(loc="upper right")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = "./data_test/sit_4.csv"
    fs = 10
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
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.set_index("Timestamp")
    draw_imu_curve(data, overlap=True, show_gt=True)