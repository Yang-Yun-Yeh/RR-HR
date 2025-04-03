import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

def draw_imu_curve(imu_data, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w'], overlap=False, show_gt=False):
    # titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']
    titles = []
    for i in range(len(cols)):
        titles.append(f'${cols[i]}$')

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
    
    row_num += 1
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
            ax.set_title(titles[i])

    if show_gt:
        if not overlap:
            ax = fig.add_subplot(spec[2, :])
        else:
            ax = fig.add_subplot(spec[3, :])
        ax.plot(time_ls, imu_data["Force"], color=colors[2])
        ax.set_title('Respiration GT Force')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))        

    ax = fig.add_subplot(spec[4, :])
    ax.plot(time_ls, imu_data["imu2_q_y"], color=colors[1])
    ax.set_title('imu2_q_y')
    
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

def draw_anc_curve_multi(imu_data, outputs, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w'], overlap=True, show_gt=True):
    # titles = ['$q_x$', '$q_y$', '$q_z$', '$q_w$']
    titles = []
    for i in range(len(cols)):
        titles.append(f'${cols[i]}$')

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']
    # 1:'lightsteelblue', 2:'palegreen', 3:'yellow', 4:'plum'
    legend_handle_ls = []

    row_num = 3
    row = 0
    if overlap:
        if not show_gt:
            row_num -= 1
    else: # not overlap
        if not show_gt:
            row_num += 2
        else:
            row_num += 3
    
    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    # show overlap (IMU)
    for i in range(len(titles)):
        ax = fig.add_subplot(spec[0, i%4])
        for k, key in enumerate(sensor_names):
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
        ax.set_title(titles[i])
    for k, key in enumerate(sensor_names):
        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))
    row += 1

    # show anc results
    if overlap:
        for i in range(len(cols)):
            ax = fig.add_subplot(spec[row, i%4])
            for k in range(len(outputs)):
                ax.plot(time_ls, outputs[k][cols[i]], color=colors[3+k])
            ax.set_title(titles[i])
        for k in range(len(outputs)):
            legend_handle_ls.append(Line2D([0], [0], label=outputs[k]['method'], color=colors[3+k]))
        row += 1
    else:
        for k in range(len(outputs)):
            for i in range(len(cols)):
                ax = fig.add_subplot(spec[row, i%4])
                ax.plot(time_ls, outputs[k][cols[i]], color=colors[3+k])
                ax.set_title(titles[i])
            row += 1
            legend_handle_ls.append(Line2D([0], [0], label=outputs[k]['method'], color=colors[3+k])) 

    # show gt
    if show_gt:
        ax = fig.add_subplot(spec[row, :])
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

def draw_imu_curve_euler(imu_data, euler_angles, sensor_names=['imu1','imu2'], overlap=False, show_gt=False):
    cols = ['u', 'v', 'w']
    titles = ['$u$', '$v$', '$w$']

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
            ax = fig.add_subplot(spec[k, i%3])
            ax.plot(time_ls, euler_angles[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])

        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    if overlap:
        for i in range(len(titles)):
            ax = fig.add_subplot(spec[2, i%3])
            for k, key in enumerate(sensor_names):
                ax.plot(time_ls, euler_angles[key + "_" + cols[i]], color=colors[k])
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
    
    plt.show()

def draw_anc_curve_multi_euler(imu_data, euler_angles, outputs, sensor_names=['imu1','imu2'], overlap=True, show_gt=True):
    cols = ['u', 'v', 'w']
    titles = ['$u$', '$v$', '$w$']

    time_ls = imu_data.index - imu_data.index[0]
    time_ls = time_ls.total_seconds()
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']
    legend_handle_ls = []

    row_num = 3
    row = 0
    if overlap:
        if not show_gt:
            row_num -= 1
    else: # not overlap
        if not show_gt:
            row_num += 2
        else:
            row_num += 3
    
    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    # show overlap (IMU)
    for i in range(len(titles)):
        ax = fig.add_subplot(spec[0, i%4])
        for k, key in enumerate(sensor_names):
            ax.plot(time_ls, euler_angles[key + "_" + cols[i]], color=colors[k])
            ax.plot(time_ls, euler_angles[key + "_" + cols[i]], color=colors[k])
        ax.set_title(titles[i])
    for k, key in enumerate(sensor_names):
        legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))
    row += 1

    # show anc results
    if overlap:
        for i in range(len(cols)):
            ax = fig.add_subplot(spec[row, i%4])
            for k in range(len(outputs)):
                ax.plot(time_ls, outputs[k][cols[i]], color=colors[3+k])
            ax.set_title(titles[i])
        for k in range(len(outputs)):
            legend_handle_ls.append(Line2D([0], [0], label=outputs[k]['method'], color=colors[3+k]))
        row += 1
    else:
        for k in range(len(outputs)):
            for i in range(len(cols)):
                ax = fig.add_subplot(spec[row, i%4])
                ax.plot(time_ls, outputs[k][cols[i]], color=colors[3+k])
                ax.set_title(titles[i])
            row += 1
            legend_handle_ls.append(Line2D([0], [0], label=outputs[k]['method'], color=colors[3+k])) 

    # show gt
    if show_gt:
        ax = fig.add_subplot(spec[row, :])
        ax.plot(time_ls, imu_data["Force"], color=colors[2])
        ax.set_title('Respiration GT Force')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))
                

    fig.legend(handles=legend_handle_ls, loc="upper right")

    plt.show()

def draw_autocorrelation_results(preds, gt, times, cols=['q_x', 'q_y', 'q_z', 'q_w']):
    markers = ["^", "s", "P", "*"]

    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gt_ok_idx = np.where(gt['freq'] > 0)[0]

    plt.plot(times[gt_ok_idx],gt['freq'][gt_ok_idx],marker="o",label='gt',)

    for i, method in enumerate(preds):
        pred_ok_idx = np.where(preds[method] > 0)[0]
        all_ok_idx = list(set(gt_ok_idx) & set(pred_ok_idx))
        
        plt.plot(times[all_ok_idx], preds[method][all_ok_idx], marker=markers[i], label=method)
    
    plt.title("Prediction Results")
    plt.xlabel("Time (s)")
    plt.ylabel("RR (1/min)")
    plt.gca().set_ylim(bottom=0)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# spectrograms: (channel_nums, freq_bins, time_steps)
def plot_spectrogram(spectrograms, sensor_names=['imu1','imu2'], cols=['q_x', 'q_y', 'q_z', 'q_w']):
    """
    Visualizes a spectrogram.
    
    Args:
        spectrogram (numpy.ndarray or torch.Tensor): 2D array of shape (freq_bins, time_steps).
        title (str): Title of the plot.
        cmap (str): Colormap for visualization.
    """
    # if isinstance(spectrogram, torch.Tensor):
    #     spectrogram = spectrogram.cpu().numpy()  # Convert tensor to numpy

    titles = []
    for i in range(len(cols)):
        titles.append(f'${cols[i]}$')

    title = "Spectrogram"
    cmap = "viridis"

    row_num = len(sensor_names)
    fig = plt.figure(figsize=(15, (spectrograms.shape[0] // 2) * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles)):
            ax = fig.add_subplot(spec[k, i % len(cols)])
            a = ax.imshow(spectrograms[k * (spectrograms.shape[0] // 2) + i % len(cols)], aspect="auto", origin="lower", cmap=cmap)
            # ax.set_xlabel("Time Steps")
            # ax.set_ylabel("Frequency Bins")
            ax.set_title(key + " " + titles[i])

            # show color bar
            plt.colorbar(a, ax=ax)
    
    fig.supxlabel('Time Steps')
    fig.supylabel('Frequency Bins')
    plt.show()

def draw_loss_epoch(mse_train_ls, l1_train_ls, mse_test_ls, l1_test_ls, name="model"):
    colors = ['blue', 'orange']
    titles = ['MSE loss', 'L1 loss (1/min)']
    labels = ['train', 'test']
    loss_ls = [[mse_train_ls, mse_test_ls], [l1_train_ls, l1_test_ls]]

    fig = plt.figure(figsize=(10, 5), layout="constrained")
    spec = fig.add_gridspec(1, len(loss_ls))
    epoch = np.arange(1, len(mse_train_ls) + 1)

    for i in range(len(loss_ls)):
        ax = fig.add_subplot(spec[0, i])
        for j in range(len(loss_ls[i])):
            ax.plot(epoch, loss_ls[i][j], color=colors[j], label=labels[j])
        ax.set_title(titles[i])
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(loc="upper right")
        ax.grid(True)
    
    plt.savefig(f'models/fig/{name}.png')
    plt.show()

def draw_learning_results(preds, gt, times, action_name, cols=['q_x', 'q_y', 'q_z', 'q_w']):
    markers = ["^", "s", "P", "*"]

    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gt_ok_idx = np.where(gt > 0)[0]

    plt.plot(times[gt_ok_idx], gt[gt_ok_idx], marker="o", label='gt',)

    for i, method in enumerate(preds):
        pred_ok_idx = np.where(preds[method] > 0)[0]
        all_ok_idx = list(set(gt_ok_idx) & set(pred_ok_idx))
        
        plt.plot(times[all_ok_idx], preds[method][all_ok_idx], marker=markers[i], label=method)
    
    plt.title(f"Prediction Results: {action_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("RR (1/min)")
    plt.gca().set_ylim(bottom=0)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def draw_acf(acf, lag, frame_segment, fs=10, acf_filtered=None):
    # plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
 
    time = np.arange(len(acf))
    # plt.plot(time, acf, marker='o', linestyle='-', color=colors[0], label="acf")
    # plt.plot(time[lag], acf[lag], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")

    # plt.xlabel("index")
    # plt.ylabel("acf")
    # plt.title("ACF")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Create a figure with 2 rows and 1 column of subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Plot on the first subplot
    ax[0].plot(time / fs, frame_segment, label='force', color=colors[1])
    ax[0].set_title('RR Force')
    ax[0].set_ylabel('rr force')
    ax[0].legend()
    ax[0].grid(True)

    # Plot on the second subplot
    ax[1].plot(time, acf, marker='o', linestyle='-', color=colors[0], label="acf")
    if acf_filtered is not None:
        ax[1].plot(time, acf_filtered, marker='o', linestyle='-', color=colors[1], label="acf_filtered")
        ax[1].plot(time[lag], acf_filtered[lag], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    else:
        ax[1].plot(time[lag], acf[lag], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    ax[1].set_title('ACF')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('acf')
    ax[1].legend()
    ax[1].grid(True)

    # Adjust layout
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