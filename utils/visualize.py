import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

def draw_imu_curve(imu_data, sensor_names=['imu1','imu2'], cols = ['q_x', 'q_y', 'q_z', 'q_w'], overlap=False, overlap_only=False, show_gt=False):
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
    
    # row_num += 1
    # fig, axes = plt.subplots(row_num, len(titles), figsize=(20,10), constrained_layout=True)
    if overlap_only:
        row_num = 1

    fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles))
    ax_ls = []

    if not overlap_only:
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
            if overlap_only:
                ax = fig.add_subplot(spec[0, i%4])
            else:
                ax = fig.add_subplot(spec[2, i%4])
            for k, key in enumerate(sensor_names):
                ax.plot(time_ls, imu_data[key + "_" + cols[i]], color=colors[k])
            ax.set_title(titles[i])
    if overlap_only:
        for k, key in enumerate(sensor_names):
            legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[k]))

    if show_gt and not overlap_only:
        if not overlap:
            ax = fig.add_subplot(spec[2, :])
        else:
            ax = fig.add_subplot(spec[3, :])
        ax.plot(time_ls, imu_data["Force"], color=colors[2])
        ax.set_title('Respiration GT Force')
        ax.set_ylabel('N')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))        

    # ax = fig.add_subplot(spec[4, :])
    # ax.plot(time_ls, imu_data["imu2_q_y"], color=colors[1])
    # ax.set_title('imu2_q_y')
    
    fig.supxlabel('seconds')
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
        ax.set_ylabel('N')
        legend_handle_ls.append(Line2D([0], [0], label='Force', color=colors[2]))
    
    fig.supxlabel('seconds')
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

def draw_autocorrelation_results(preds, gt, times, cols=['q_x', 'q_y', 'q_z', 'q_w'], action_name=None):
    markers = ["^", "s", "P", "*"]

    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gt_ok_idx = np.where(gt['freq'] > 0)[0]

    plt.plot(times[gt_ok_idx],gt['freq'][gt_ok_idx],marker="o",label='gt',)
    # plot mean of gt
    plt.plot(times[gt_ok_idx], np.zeros_like(gt['freq'][gt_ok_idx]) + np.mean(gt['freq'][gt_ok_idx]), label='gt_mean')

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

# spectrograms: (channel_nums, freq_bins, time_steps)
def plot_spectrogram_16(spectrograms, sensor_names=['imu1','imu2'], q_cols=['q_x', 'q_y', 'q_z', 'q_w'], omega_cols=['omega_u', 'omega_v', 'omega_w'], as_cols=['omega']):
    """
    Visualizes a spectrogram.
    
    Args:
        spectrogram (numpy.ndarray or torch.Tensor): 2D array of shape (freq_bins, time_steps).
        title (str): Title of the plot.
        cmap (str): Colormap for visualization.
    """
    # if isinstance(spectrogram, torch.Tensor):
    #     spectrogram = spectrogram.cpu().numpy()  # Convert tensor to numpy

    titles_q = []
    for i in range(len(q_cols)):
        titles_q.append(f'${q_cols[i]}$')

    titles_omega = []
    for i in range(len(omega_cols)):
        titles_omega.append(f'$\{omega_cols[i]}$')

    titles_as = []
    for i in range(len(as_cols)):
        titles_as.append(f'$\{as_cols[i]}$')

    title = "Spectrogram"
    cmap = "viridis"

    row_num = 4
    fig = plt.figure(figsize=(15, 4 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles_q))
    ax_ls = []

    row = 0
    count = 0
    # quaternion
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_q)):
            ax = fig.add_subplot(spec[row, i % len(titles_q)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            # ax.set_xlabel("Time Steps")
            # ax.set_ylabel("Frequency Bins")
            ax.set_title(key + " " + titles_q[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1

    # angular velocity
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_omega)):
            ax = fig.add_subplot(spec[row, i % len(titles_omega)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(key + " " + titles_omega[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1

    row -= 2
    # angular speed
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_as)):
            ax = fig.add_subplot(spec[row, 3 + i % len(titles_omega)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(key + " " + titles_as[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1
    
    fig.supxlabel('Time Steps')
    fig.supylabel('Frequency Bins')
    plt.show()

# spectrograms: (channel_nums, freq_bins, time_steps)
def plot_spectrogram_32(spectrograms, sensor_names=['imu1','imu2'], q_cols=['q_x', 'q_y', 'q_z', 'q_w'], omega_cols=['omega_u', 'omega_v', 'omega_w'], as_cols=['omega'], anc_methods = ['LMS', 'LMS+LS', 'RLS', 'LRLS']):
    titles_q = []
    for i in range(len(q_cols)):
        titles_q.append(f'${q_cols[i]}$')

    titles_omega = []
    for i in range(len(omega_cols)):
        titles_omega.append(f'$\{omega_cols[i]}$')

    titles_as = []
    for i in range(len(as_cols)):
        titles_as.append(f'$\{as_cols[i]}$')

    titles_anc = []
    for anc_method in anc_methods:
        titles_anc.append(f'${anc_method}$')


    title = "Spectrogram"
    cmap = "viridis"

    row_num = 8
    fig = plt.figure(figsize=(15, 4 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles_q))
    ax_ls = []

    row = 0
    count = 0
    # quaternion
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_q)):
            ax = fig.add_subplot(spec[row, i % len(titles_q)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            # ax.set_xlabel("Time Steps")
            # ax.set_ylabel("Frequency Bins")
            ax.set_title(key + " " + titles_q[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1

    # angular velocity
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_omega)):
            ax = fig.add_subplot(spec[row, i % len(titles_omega)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(key + " " + titles_omega[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1

    row -= 2
    # angular speed
    for k, key in enumerate(sensor_names):
        if key not in sensor_names:
            continue
        for i in range(len(titles_as)):
            ax = fig.add_subplot(spec[row, 3 + i % len(titles_omega)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(key + " " + titles_as[i])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1

    # ANC
    for i in range(len(titles_anc)):
        for j in range(len(titles_q)):
            ax = fig.add_subplot(spec[row, j % len(titles_anc)])
            a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(titles_anc[i] + " " + titles_q[j])

            # show color bar
            plt.colorbar(a, ax=ax)
            count += 1
        row += 1
    
    fig.supxlabel('Time Steps')
    fig.supylabel('Frequency Bins')
    plt.show()

# spectrograms: (channel_nums, freq_bins, time_steps)
def plot_spectrogram(spectrograms, features = ['Q', 'omega', 'omega_l2', 'ANC'], sensor_names=['imu1','imu2'], q_cols=['q_x', 'q_y', 'q_z', 'q_w'], omega_cols=['omega_u', 'omega_v', 'omega_w'], as_cols=['omega'], anc_methods = ['LMS', 'LMS+LS', 'RLS', 'LRLS']):
    titles_q = []
    for i in range(len(q_cols)):
        titles_q.append(f'${q_cols[i]}$')

    titles_omega = []
    for i in range(len(omega_cols)):
        titles_omega.append(f'$\{omega_cols[i]}$')

    titles_as = []
    for i in range(len(as_cols)):
        titles_as.append(f'$\{as_cols[i]}$')

    titles_anc = []
    for anc_method in anc_methods:
        titles_anc.append(f'${anc_method}$')


    title = "Spectrogram"
    cmap = "viridis"

    row_num = 0
    if 'Q' in features:
        row_num += 2
    if 'omega' in features:
        row_num += 2
    if 'omega_l2' in features and 'omega' not in features:
        row_num += 2
    if 'ANC' in features:
        row_num += 4

    fig = plt.figure(figsize=(15, 4 * row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, len(titles_q))

    row = 0
    count = 0

    # quaternion
    if 'Q' in features:
        for k, key in enumerate(sensor_names):
            if key not in sensor_names:
                continue
            for i in range(len(titles_q)):
                ax = fig.add_subplot(spec[row, i % len(titles_q)])
                a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
                # ax.set_xlabel("Time Steps")
                # ax.set_ylabel("Frequency Bins")
                ax.set_title(key + " " + titles_q[i])

                # show color bar
                plt.colorbar(a, ax=ax)
                count += 1
            row += 1

    # angular velocity
    if 'omega' in features:
        for k, key in enumerate(sensor_names):
            if key not in sensor_names:
                continue
            for i in range(len(titles_omega)):
                ax = fig.add_subplot(spec[row, i % len(titles_omega)])
                a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
                ax.set_title(key + " " + titles_omega[i])

                # show color bar
                plt.colorbar(a, ax=ax)
                count += 1
            row += 1

    
    # angular speed
    if 'omega_l2' in features:
        row -= 2
        for k, key in enumerate(sensor_names):
            if key not in sensor_names:
                continue
            for i in range(len(titles_as)):
                ax = fig.add_subplot(spec[row, 3])
                a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
                ax.set_title(key + " " + titles_as[i])

                # show color bar
                plt.colorbar(a, ax=ax)
                count += 1
            row += 1

    # ANC
    if 'ANC' in features:
        for i in range(len(titles_anc)):
            for j in range(len(titles_q)):
                ax = fig.add_subplot(spec[row, j % len(titles_anc)])
                a = ax.imshow(spectrograms[count], aspect="auto", origin="lower", cmap=cmap)
                ax.set_title(titles_anc[i] + " " + titles_q[j])

                # show color bar
                plt.colorbar(a, ax=ax)
                count += 1
            row += 1
    
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
    markers = ["^", "s", "P", "*", "o", "D", "v", "X", "h", "+", "x", "|", "_"]

    fig = plt.figure(figsize=(15, 5), layout="constrained")
    gt_ok_idx = np.where(gt > 0)[0]

    plt.plot(times[gt_ok_idx], gt[gt_ok_idx], marker="o", label='gt',)
    # plot mean of gt
    plt.plot(times[gt_ok_idx], np.zeros_like(gt[gt_ok_idx]) + np.mean(gt[gt_ok_idx]), label='gt_mean')

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

    # Create a figure with 2 rows and 1 column of subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Plot on the first subplot
    ax[0].plot(time / fs, frame_segment, label='force', color=colors[1])
    ax[0].set_title('RR force in running activity')
    ax[0].set_xlabel('second')
    ax[0].set_ylabel('RR force')
    ax[0].legend()
    ax[0].grid(True)

    # Plot on the second subplot
    ax[1].plot(time, acf, marker='o', linestyle='-', color=colors[0], label="acf")
    if acf_filtered is not None:
        ax[1].plot(time, acf_filtered, marker='o', linestyle='-', color=colors[1], label="acf_filtered")
        ax[1].plot(time[lag], acf_filtered[lag], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    else:
        ax[1].plot(time[lag], acf[lag], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
    ax[1].set_title('ACF of RR force')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('ACF')
    ax[1].legend()
    ax[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def draw_learning_results_action(preds, gt, mae_test, models_name=None, paper=False):
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']
    
    col_num = len(preds[next(iter(preds))]) # action num
    row_num = len(preds) # method num
    fig = plt.figure(figsize=(3*col_num, 3*row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, col_num)
    legend_handle_ls = []
    axix_lim = 40

    for i, method in enumerate(preds):
        for j, action in enumerate(preds[method]):
            ax = fig.add_subplot(spec[i, j % col_num])
            preds[method][action] = np.array(preds[method][action])
            ax.set_title(f'{action} (MAE={mae_test[method][action]:.4f})')
            if not paper:
                ax.scatter(60 * gt[action], 60 * preds[method][action], color=colors[i], s=5)
            else:
                ax.scatter(gt[action], preds[method][action], color=colors[i], s=5)
            ax.plot(np.linspace(0, axix_lim, num=1000), np.linspace(0, axix_lim, num=1000), color='black')
            ax.grid(True)
            # ax.set_ylabel('gt (1/min)')
            # ax.set_xlabel('pred (1/min)')
            ax.set_ylim(0, axix_lim)
            ax.set_xlim(0, axix_lim)

        legend_handle_ls.append(Line2D([], [], label=models_name[i], color="white", marker='o', markerfacecolor=colors[i]))
    
    
    fig.supxlabel('gt (1/min)')
    fig.supylabel('pred (1/min)')
    fig.legend(handles=legend_handle_ls, loc="upper right")
    
    plt.show()

def draw_learning_results_action_bar(mae_test, models_name=None, paper=False):
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']

    col_num = 2
    row_num = 2
    fig = plt.figure(figsize=(4*col_num, 3*row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, col_num)
    y_axis_upper = 15
    delta_y = 2

    action_ls = mae_test[next(iter(mae_test))].keys()
    method_ls = mae_test.keys()
    x = np.arange(len(method_ls))
    
    # Draw bar plot
    for i, action in enumerate(action_ls):
        ax = fig.add_subplot(spec[i // col_num , i % col_num])

        for j, method in enumerate(method_ls):
            plt.bar(x[j], mae_test[method][action], color=colors[j], label=method)
        

        ax.set_title(f'{action}')
        ax.legend(loc="upper right")
        plt.xticks(x, method_ls)
        ax.yaxis.grid(True)
        ax.set_ylabel('MAE (1/min)')
        ax.set_ylim(0, y_axis_upper)
        ax.set_yticks(np.arange(0, y_axis_upper + delta_y, delta_y))
 
    plt.show()



def draw_learning_results_action_relative(relative_mae, sigma_num=1, models_name=None):
    colors = ['blue', 'orange', 'green', 'purple', 'cyan', 'deeppink', 'olivedrab']

    col_num = 2 # len(relative_mae[next(iter(relative_mae))]) # action num
    row_num = 2 # len(relative_mae) # method num
    fig = plt.figure(figsize=(4*col_num, 3*row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, col_num)
    legend_handle_ls = []
    y_axis_upper = 80

    action_ls = relative_mae[next(iter(relative_mae))].keys()
    method_ls = relative_mae.keys()
    x = np.arange(len(method_ls))
    relative_mae_means = {key:{} for key in action_ls} # [action][method]
    relative_mae_std = {key:{} for key in action_ls} # [action][method]
    
    # Calculate relative mae for errorbar plot
    for i, method in enumerate(relative_mae):
        for j, action in enumerate(relative_mae[method]):
            relative_mae_means[action][method] = np.mean(relative_mae[method][action])
            relative_mae_std[action][method] = np.std(relative_mae[method][action])

    # Draw errorbar plot
    for i, action in enumerate(relative_mae_means):
        ax = fig.add_subplot(spec[i // col_num , i % col_num])

        mean_ls = []
        error_ls = []
        for j, method in enumerate(relative_mae_means[action]):
            mean_ls.append(relative_mae_means[action][method])
            error_ls.append(sigma_num * np.array(relative_mae_std[action][method]))
            plt.errorbar(x[j], mean_ls[j], yerr=error_ls[j], fmt='o', capsize=5, color=colors[j], label=method)
        

        ax.set_title(f'{action}')
        ax.legend(loc="upper right")
        # plt.errorbar(x, mean_ls, yerr=error_ls, fmt='o', capsize=5, color=colors[i])
        # plt.xticks(x, method_ls, rotation=45)
        plt.xticks(x, method_ls)
        if len(x) == 2:
            ax.set_xlim(x[0] - 1, x[-1] + 1)
        ax.yaxis.grid(True)
        ax.set_ylabel('MAPE (%)')
        ax.set_ylim(0, y_axis_upper)
        ax.set_yticks(np.arange(0, y_axis_upper + 10, 10))
        # ax.set_xlim(0, len(x))
 
    plt.show()

def plot_mae_comparison(mae_ml, mae_paper):
    model_names = mae_ml['model_name']
    features = [key for key in mae_ml if key != 'model_name']
    
    num_models = len(model_names)
    num_features = len(features)
    
    x_labels = features + list(mae_paper.keys())
    x = np.arange(len(x_labels))
    total_bar_width = 0.8
    bar_width = total_bar_width / num_models

    fig, ax = plt.subplots(figsize=(16, 6))

    # Color
    colors = ["#418aff", "#ff6060", "#ffcd76", "#44ff70"]  # MLP, CNN, ViT_emt2, ViT_emht2
    paper_colors = ["#a386ff", "#ff8fff"]      # paper_4, paper_5

    # Model result
    for i, model in enumerate(model_names):
        values = [mae_ml[feat][i] for feat in features]
        bar_positions = x[:num_features] + i * bar_width - total_bar_width / 2 + bar_width / 2
        bars = ax.bar(bar_positions, values, width=bar_width, color=colors[i], label=model)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Paper result
    for i, (paper, value) in enumerate(mae_paper.items()):
        pos = x[num_features + i]
        bar = ax.bar(pos, value, width=bar_width * 1.5, color=paper_colors[i], label=paper)
        ax.text(pos, value + 0.05, f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Features / Papers")
    ax.set_ylabel("MAE (1/min)")
    ax.set_title("L1 Loss (MAE) Comparison")
    ax.legend(title="Models / Papers")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def draw_order_RMSE(train_rmse_ls, val_rmse_ls):
    # X-axis values (Orders)
    M = len(train_rmse_ls)
    orders = np.arange(1, M+1)

    # Coordinates for the highlighted circle
    highlight_order = np.argmin(val_rmse_ls) + 1
    highlight_rmse = val_rmse_ls[np.where(orders == highlight_order)[0][0]]

    # # --- Plotting ---
    # # Create the figure and axes
    # plt.figure(figsize=(12, 4))
    # ax = plt.gca()

    # # Plot Training and Validation RMSE
    # ax.plot(orders, train_rmse_ls, marker='.', linestyle='-', label='Training RMSE')
    # ax.plot(orders, val_rmse_ls, marker='.', linestyle='-', label='Validation RMSE')

    # # Add the red circle highlight
    # ax.plot(highlight_order, highlight_rmse, marker='o', markersize=12,
    #         fillstyle='none', markeredgecolor='red', markeredgewidth=1.0)

    # # Set titles and labels
    # ax.set_title('RMSE vs. orders for system ID')
    # ax.set_ylabel('RMSE')
    # # The x-axis label is not in the original image, but it is good practice to include it.
    # ax.set_xlabel('Orders')


    # # Set axis limits and ticks
    # ax.set_xlim(0, M+1)
    # ax.set_xticks(np.arange(0, M+1, 1))

    # # Add legend and grid
    # ax.legend(loc="upper right")
    # ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')

    # # Display the plot
    # plt.show()


    # Create a figure with 2 rows and 1 column of subplots
    colors = ["#418aff", "#ff6060", "#ffcd76", "#44ff70"]
    row_num = 2
    fig, ax = plt.subplots(row_num, 1, figsize=(12, 6))

    for i in range(row_num):
        if i == 0:
            ax[i].plot(orders, train_rmse_ls, marker='.', linestyle='-', label='Training RMSE', color=colors[i])
        elif i == 1:
            ax[i].plot(orders, val_rmse_ls, marker='.', linestyle='-', label='Validation RMSE', color=colors[i])
            ax[i].plot(highlight_order, highlight_rmse, marker='o', markersize=12,
            fillstyle='none', markeredgecolor='red', markeredgewidth=1.0)
        
        ax[i].set_title('RMSE vs. orders of FIR filter')
        ax[i].set_ylabel('RMSE')
        ax[i].set_xlabel('Orders')

        ax[i].set_xlim(0, M+1)
        ax[i].set_xticks(np.arange(0, M+1, 2))
        ax[i].legend(loc="upper right")
        ax[i].grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')

    plt.tight_layout()
    plt.show()

def draw_order_action(m_best_action, M):
    c = "#418aff"

    col_num = 2
    row_num = 2
    fig = plt.figure(figsize=(5*col_num, 3*row_num), layout="constrained")
    spec = fig.add_gridspec(row_num, col_num)
    y_axis_upper = 15
    delta_y = 2

    action_ls = m_best_action.keys()
    x = np.arange(0.5, M+1+0.5, 1)
    # print(x)
    
    # Draw bar plot
    for i, action in enumerate(action_ls):
        ax = fig.add_subplot(spec[i // col_num , i % col_num])

        counts, bin_edges, _ = ax.hist(m_best_action[action], bins=x, color=c, edgecolor='black')
        # print(f'{action}: {counts}')
        # max_frequency = np.max(counts)
        # max_freq_index = np.argmax(counts)
        # x_value_at_max_freq = bin_edges[max_freq_index]

        y_axis_upper = np.max(counts)    

        ax.set_title(f'{action}, median of order: {np.median(m_best_action[action])}')
        ax.set_xlabel('order of impulse response')
        ax.set_ylabel('number')

        ax.yaxis.grid(True)
        ax.set_xlim(0, M+1)
        ax.set_xticks(np.arange(0, M+1, 2))
        
        ax.set_ylim(0, y_axis_upper)
        ax.set_yticks(np.arange(0, y_axis_upper+1, 2))

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