import numpy as np
import matplotlib.pyplot as plt
import FIR_filter

fs = 1000

NTAPS = 100
LEARNING_RATE = 0.001
fnoise = 50

ecg = np.loadtxt("data_test/ecg50hz.dat")
plt.figure(1)
plt.plot(ecg)

f = FIR_filter.FIR_filter(np.zeros(NTAPS))

y = np.empty(len(ecg))
for i in range((len(ecg))):
    ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i)
    canceller = f.filter(ref_noise)
    output_signal = ecg[i] - canceller
    f.lms(output_signal, LEARNING_RATE)
    y[i] = output_signal

plt.figure(2)
plt.plot(y)
plt.show()