import numpy as np
import matplotlib.pyplot as plt
import FIR_filter
from matplotlib.lines import Line2D

fs = 1000

NTAPS = 100
LEARNING_RATE = 0.001
fnoise = 50

# RLS
delta = 1000 # 1
lam = 0.998 # 0.9995

ecg = np.loadtxt("data_test/ecg50hz.dat")
# plt.figure(1)
# plt.plot(ecg)

f_lms = FIR_filter.FIR_filter(np.zeros(NTAPS))
f_lmsls = FIR_filter.FIR_filter(np.zeros(NTAPS))
f_rls = FIR_filter.FIR_filter(np.zeros(NTAPS))

# LS
x = np.arange(len(ecg))
x = np.sin(2.0 * np.pi * fnoise/fs * x)
d = ecg
f_lmsls.ls(x, d)
# LEARNING_RATE = np.max(f.coefficients) / 1e6 # / 100
# print(f'LEARNING_RATE:{LEARNING_RATE}')

y_lms = np.empty(len(ecg))
y_lmsls = np.empty(len(ecg))
y_rls = np.empty(len(ecg))

for i in range((len(ecg))):
    ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i)

    canceller_lms = f_lms.filter(ref_noise)
    canceller_lmsls = f_lmsls.filter(ref_noise)
    canceller_rls = f_rls.filter(ref_noise)

    output_signal_lms = ecg[i] - canceller_lms
    output_signal_lmsls = ecg[i] - canceller_lmsls
    output_signal_rls = ecg[i] - canceller_rls

    f_lms.lms(output_signal_lms, LEARNING_RATE)
    f_lmsls.lms(output_signal_lmsls, LEARNING_RATE)
    f_rls.rls(output_signal_rls, delta=delta, lam=lam)

    y_lms[i] = output_signal_lms
    y_lmsls[i] = output_signal_lmsls
    y_rls[i] = output_signal_rls

output = {'LMS':y_lms, 'LMS+LS':y_lmsls, 'RLS':y_rls}

# plt.figure(2)
# plt.plot(y)
# plt.show()

# Draw results
colors = ['blue', 'red', 'orange', 'green', 'purple', 'cyan']
row_num = 5 # 5
row = 0
fig = plt.figure(figsize=(15, 3 * row_num), layout="constrained")
spec = fig.add_gridspec(row_num, 1)
legend_handle_ls = []

# d(n): ecg
ax = fig.add_subplot(spec[row, :])
ax.plot(np.arange(len(ecg)), d, color=colors[row])
ax.set_title('ECG signal')
ax.grid(True, linestyle='--', linewidth=0.5)
legend_handle_ls.append(Line2D([0], [0], label='d(n)', color=colors[row]))
row += 1

# x(n): sine
end = 1000 # 1000, -1
ax = fig.add_subplot(spec[row, :])
# ax.plot(np.arange(len(ecg))[:end], x[:end], color=colors[row])
ax.plot(np.arange(len(ecg)), x, color=colors[row])
ax.set_title('Reference noise: sine wave')
ax.grid(True, linestyle='--', linewidth=0.5)
legend_handle_ls.append(Line2D([0], [0], label='x(n)', color=colors[row]))
row += 1

# ANC results
for key, value in output.items():
    ax = fig.add_subplot(spec[row, :])
    ax.plot(np.arange(len(ecg)), value, color=colors[row])
    ax.set_title(key)
    # ax.set_ylim(1500, 3300)
    ax.grid(True, linestyle='--', linewidth=0.5)
    legend_handle_ls.append(Line2D([0], [0], label=key, color=colors[row]))
    row += 1

ax.set_xlabel("sample points")
fig.legend(handles=legend_handle_ls, loc="upper right")

plt.show()