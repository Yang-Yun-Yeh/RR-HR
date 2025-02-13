import numpy as np
import matplotlib.pyplot as plt
import FIR_filter

fs = 1000

NTAPS = 100
LEARNING_RATE = 0.001
fnoise = 50

# RLS
delta = 1000 # 1
lam = 0.998 # 0.9995

ecg = np.loadtxt("data_test/ecg50hz.dat")
plt.figure(1)
plt.plot(ecg)

f = FIR_filter.FIR_filter(np.zeros(NTAPS))

# LS
# x = np.arange(len(ecg))
# x = np.sin(2.0 * np.pi * fnoise/fs * x)
# d = ecg
# f.ls(x, d)
# LEARNING_RATE = np.max(f.coefficients) / 1e6 # / 100
# print(f'LEARNING_RATE:{LEARNING_RATE}')

y = np.empty(len(ecg))
for i in range((len(ecg))):
    ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i)
    canceller = f.filter(ref_noise)
    output_signal = ecg[i] - canceller
    # LMS
    # f.lms(output_signal, LEARNING_RATE)
    # RLS
    f.rls(output_signal, delta=delta, lam=lam)
    y[i] = output_signal

plt.figure(2)
plt.plot(y)
plt.show()