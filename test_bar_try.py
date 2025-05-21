import matplotlib.pyplot as plt
import numpy as np

# Bone categories
bones = ['pelvis', 'thigh', 'calf', 'spine 1', 'spine 2', 'neck', 'head', 'shoulder', 'upper arm', 'forearm']
x = np.arange(len(bones))

# Synthetic (smpl) data
synthetic_means = [0.15, 0.44, 0.43, 0.25, 0.25, 0.12, 0.11, 0.15, 0.27, 0.24]
synthetic_errors = [0.01, 0.02, 0.03, 0.01, 0.01, 0.005, 0.005, 0.01, 0.015, 0.015]

# Human3.6M data
human_means = [0.13, 0.46, 0.45, 0.24, 0.25, 0.12, 0.11, 0.16, 0.26, 0.24]
human_errors = [0.008, 0.01, 0.01, 0.008, 0.008, 0.004, 0.004, 0.009, 0.012, 0.01]

# Plot
plt.figure(figsize=(10, 4))
plt.errorbar(x, synthetic_means, yerr=synthetic_errors, fmt='o', capsize=5, label='Synthetic (smpl)')
plt.errorbar(x, human_means, yerr=human_errors, fmt='o', capsize=5, label='Human3.6M', color='orange')

# Axis settings
plt.xticks(x, bones, rotation=45)
plt.ylabel('Bone length (mm)')
plt.legend()
plt.tight_layout()
plt.show()