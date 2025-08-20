import numpy as np
from scipy import signal as sg
from scipy.spatial.transform import Rotation as R
import statsmodels.api as sm
from sklearn.metrics import r2_score
import os
import pickle

try:
     from . import visualize as vs
     from . import FIR_filter as FIRF
except:
     import visualize as vs
     import FIR_filter as FIRF

def min_max_normalize(X):
    return (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))

# S:(window_num, channel_num, freq_num, time_num)
def normalize_spectrogram(S, byCol=False):
    S_normalize = np.zeros_like(S)
    # print(f'S_normalize.shape:{S_normalize.shape}')
    if byCol:
        for i in range(S.shape[0]):
            for j in range(S.shape[3]):
                S_normalize[i, :, :, j] = min_max_normalize(S[i, :, :, j])
    else:
        for i in range(S.shape[0]):
            S_normalize[i] = min_max_normalize(S[i])
    
    return S_normalize

def align_delay(data, delay=10, fs=10):
    gt_shift = data["Force"][int(delay * fs):].to_list()
    data.loc[:len(gt_shift)-1, "Force"] = gt_shift
    data = data[:len(gt_shift)-1]

    return data

# Low pass filer
def butter_filter(data, cutoff=0.33, fs=10, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sg.butter(order, normal_cutoff, btype="low", analog=False)
    return sg.filtfilt(b, a, data)

# q: (sample_num, 4)
def quaternion_to_euler(q):
    r = R.from_quat(q)
    rot = r.as_euler('xyz', degrees=True)
    return rot[:, 0], rot[:, 1], rot[:, 2]

def quaternion_correction(q_target, q_skew):
    """
    計算校正四元數 q_corr，使得 q_skew 旋轉後與 q_target 對齊
    返回 q_corr
    """
    q_target = R.from_quat(q_target)  # 轉換為 Rotation 物件
    q_skew = R.from_quat(q_skew)
    
    q_corr = q_target * q_skew.inv()  # 計算校正四元數
    return q_corr.as_quat()  # 轉回四元數格式 (x, y, z, w)

def align_quaternion(q_corr, q_skew):
    """
    使用校正四元數修正 q_skew
    q_corr: 校正四元數 (w, x, y, z)
    q_skew: 需要修正的四元數
    返回對齊後的 q_skew
    """
    q_corr = R.from_quat(q_corr)
    q_skew = R.from_quat(q_skew)

    # print(f'matrix:{q_corr.as_matrix()}')
    
    q_Sb_aligned = q_corr * q_skew  # 應用校正
    return q_Sb_aligned.as_quat()

def quaternion_distance(q1, q2):
    """
    Compute the angular distance (in radians) between two unit quaternions.
    
    Parameters:
    q1, q2 : array-like
        Two quaternions (4D vectors in the form [w, x, y, z]).

    Returns:
    float
        The angular distance in radians.
    """
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # Compute dot product
    dot_product = np.abs(np.dot(q1, q2))  # Absolute to handle double-cover property

    # Clamp the value to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute angle
    theta = 2 * np.arccos(dot_product)
    
    return theta

def getRANSACInlier(Q_target, Q_skew, q_corr, d):
    inlier = 0
    for i in range(len(Q_target)):
        q_aligned = align_quaternion(q_corr, Q_skew[i])
        distance = quaternion_distance(Q_target[i], q_aligned)
        if distance < d:
            inlier += 1
    return inlier

def Q_RANSAC(data_still, pool=0.5, d=0.1): # data_still: No motion duration, d:threshold (distance for inlier)
    N = int(pool * len(data_still)) # number of iterations
    sz = len(data_still) # size
    best_q_corr = 0
    best_score = -1

    target, skew = 'imu1', 'imu2'
    Q_target = data_still[[target + '_q_x', target + '_q_y', target + '_q_z', target + '_q_w']].to_numpy()
    Q_skew = data_still[[skew + '_q_x', skew + '_q_y', skew + '_q_z', skew + '_q_w']].to_numpy()

    idx_ls = np.random.choice(sz, N, replace=False)
    
    for i, idx in enumerate(idx_ls):
        q_corr = quaternion_correction(Q_target[idx], Q_skew[idx])
        score = getRANSACInlier(Q_target, Q_skew, q_corr, d)
        # print(f'score: {score}')
        if score > best_score:
            best_q_corr = q_corr
            best_score = score

    print(f'best_score/total: {best_score}/{sz}')
            
    return best_q_corr

# q1, q2:(x, y, z, w)
def calculate_omega(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1],
        q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
        q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])

# q:(sample num, 4)
def q_to_omega(q, fs=10):
    omega = np.zeros((q.shape[0], 3))
    dt = 1/fs
    for i in range(1, q.shape[0]):
        omega[i] = calculate_omega(q[i], q[i-1], dt)

    omega[0] = omega[1]
    
    return omega

# omega:(sample num, 3)
def omega_to_AngSpeed(omega):
    return np.linalg.norm(omega, axis=1)

def anc_process(data_anc, NTAPS=3, LEARNING_RATE=0.001, delta=1, lam_rls=0.9995, epsilon=1e-6, lam_lrls=0.9995):
    # Init
    outputs_dict_LMS = FIRF.LMS(data_anc, NTAPS, LEARNING_RATE)
    outputs_dict_LMSLS = FIRF.LMSLS(data_anc, NTAPS, LEARNING_RATE)
    outputs_dict_RLS = FIRF.RLS(data_anc, NTAPS, delta, lam_rls)
    outputs_dict_LRLS = FIRF.LRLS(data_anc, NTAPS, epsilon, lam_lrls)

    outputs = [outputs_dict_LMS, outputs_dict_LMSLS, outputs_dict_RLS, outputs_dict_LRLS]

    return outputs

def auto_correlation(data, outputs, cols=['q_x', 'q_y', 'q_z', 'q_w'], fs=10, window=10, overlap=5, visualize=True, action_name=None, return_r2=False, return_pgm=False):
    N = len(data) # Signal length in samples
    T = 1/fs # Sampling period
    n = int(window * fs) # lag (window size)
    t = N / fs # Signal length in seconds
    flag = -10 # No freq. symbol
    # print(f'f_s :{fs}, T:{T}, N:{N}, n:{n}, t:{t}')

    window_size = n
    overlap_size = int(overlap * fs)
    window_num = int((N - n) / (n - overlap_size)) + 1

    # print(f'window_num:{window_num}, overlap_size:{overlap_size}, window_size:{window_size}')

    # Initialize
    freqs, calrities, mae, r2 = {}, {}, {}, {}
    gt, preds, times = {}, {}, []
    for i in range(len(outputs)):
        freqs[outputs[i]['method']] = {key:[] for key in cols}
        calrities[outputs[i]['method']] = {key:[] for key in cols}
        mae[outputs[i]['method']] = None
        preds[outputs[i]['method']] = []

    # Auto-correlation for gt, add times
    gt['freq'], gt['calrity'] = [], []
    for i in range(window_num):
        frame_start = i * (window_size - overlap_size)
        frame_segment = data['Force'][frame_start:frame_start+window_size]
        
        acf = sm.tsa.acf(frame_segment, nlags=n)
        acf_filtered = butter_filter(acf, cutoff=1) # cutoff=0.66
        peaks = sg.find_peaks(acf_filtered)[0]
        # peaks = sg.find_peaks(acf)[0] # Find peaks of the autocorrelation
        # peaks = sg.find_peaks(acf, height=np.median(acf) / 3, distance=fs)[0]
        if peaks.any():
            lag = peaks[0] # Choose the first peak as our pitch component lag
            pitch = fs / lag # Transform lag into frequency
            clarity = acf_filtered[lag] / acf_filtered[0]

            peak_id = 0
            threshold = -0.1
            while (clarity < threshold or pitch*60 > 35) and peak_id < len(peaks):
                lag = peaks[peak_id]
                pitch = fs / lag
                clarity = acf_filtered[lag] / acf_filtered[0]
                peak_id += 1
            
            if clarity < threshold:
                pitch = flag

            gt['freq'].append(pitch * 60)
            gt['calrity'].append(clarity)
        else: # peaks is empty
            gt['freq'].append(flag)
            gt['calrity'].append(flag)
        
        times.append((frame_start + frame_start + window_size) / (2 * fs))

    # Auto-correlation for outputs
    for i, method in enumerate(freqs):
        for j, col in enumerate(cols):
            for k in range(window_num):
                frame_start = k * (window_size - overlap_size)
                frame_segment = outputs[i][col][frame_start:frame_start+window_size]
        
                acf = sm.tsa.acf(frame_segment, nlags=n)
                acf_filtered = butter_filter(acf, cutoff=1) # cutoff=0.66
                peaks = sg.find_peaks(acf_filtered)[0]
                # peaks = sg.find_peaks(acf)[0] # Find peaks of the autocorrelation
                # peaks = sg.find_peaks(acf, height=np.median(acf) / 3, distance=fs)[0]
                if peaks.any():
                    lag = peaks[0] # Choose the first peak as our pitch component lag
                    pitch = fs / lag # Transform lag into frequency
                    clarity = acf_filtered[lag] / acf_filtered[0]

                    peak_id = 0
                    threshold = -0.1
                    while (clarity < threshold or pitch*60 > 40) and peak_id < len(peaks):
                        lag = peaks[peak_id]
                        pitch = fs / lag
                        clarity = acf_filtered[lag] / acf_filtered[0]
                        peak_id += 1
                    
                    if clarity < threshold:
                        pitch = flag
                    freqs[method][col].append(pitch * 60)
                    calrities[method][col].append(clarity)
                else: # peaks is empty
                    freqs[method][col].append(flag)
                    calrities[method][col].append(flag)
    
    # Estimate error (MAE)
    gt_ok_idx = np.where(np.array(gt['freq']) > 0)[0]
    for i, method in enumerate(mae):
        mae_ls = []
        for j in range(len(gt['freq'])):
            if j in gt_ok_idx:
                mae_sample_ls = []
                pred_sample_ls = []
                for k, col in enumerate(cols):
                    if freqs[method][col][j] >= 0:
                        mae_sample = abs(freqs[method][col][j] - gt['freq'][j])
                        mae_sample_ls.append(mae_sample)
                        pred_sample_ls.append(freqs[method][col][j])

                if mae_sample_ls:
                    # min_idx = np.argmin(mae_sample_ls)
                    # mae_ls.append(mae_sample_ls[min_idx])
                    # preds[method].append(pred_sample_ls[min_idx])

                    # mae_ls.append(np.mean(mae_sample_ls))
                    mae_ls.append(abs(np.mean(mae_sample_ls) - np.mean(pred_sample_ls)))
                    preds[method].append(np.mean(pred_sample_ls))
                else:
                    preds[method].append(flag)
        if mae_ls:
            mae[method] = np.mean(mae_ls)
            preds[method] = np.array(preds[method])
            # gt_shape = np.array(gt['freq']).shape
            # pred_shape = np.array(preds[method]).shape
            # print(f'gt: {gt_shape}, pred: {pred_shape}')
            r2[method] = r2_score(np.array(gt['freq']), np.array(preds[method]))

    # Convert to numpy array
    gt['freq'] = np.array(gt['freq'])
    gt['calrity'] = np.array(gt['calrity'])
    times = np.array(times)

    # print(f'preds:{preds}')
    # gt_test = gt['freq']
    # print(f'gt:{gt_test}')

    # Draw results
    if visualize:
        vs.draw_autocorrelation_results(preds, gt, times, cols=cols, action_name=action_name)
    
    if return_r2:
        return mae, r2
    if return_pgm:
        return preds, gt['freq'], mae
    else:
        return mae

# fs=10, nperseg=128, noverlap=64
def compute_spectrogram(imu_data, fs=10, nperseg=128, noverlap=64):
    """
    Convert raw IMU data into spectrograms.
    
    Args:
        imu_data: (num_samples, 8) raw IMU data
        fs: Sampling frequency of IMU data (adjust as needed)
        nperseg: Window size for STFT
        noverlap: Overlapping samples
        
    Returns:
        spectrograms: (num_samples, 8, freq_bins, time_steps)
    """
    num_samples, num_channels = imu_data.shape
    spectrograms = []
    # print(f'nperseg:{nperseg}, noverlap:{noverlap}')

    for i in range(num_channels):  # Loop over 8 IMU channels
        f, t, Sxx = sg.spectrogram(imu_data[:, i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        spectrograms.append(Sxx)  # Shape (freq_bins, time_steps)

    spectrograms = np.stack(spectrograms, axis=0)  # (8, freq_bins, time_steps)
    
    return spectrograms

def compute_gt(force_seg, fs=10, nperseg=128, noverlap=64, start_t=0, return_t=False):
    N = len(force_seg) # Signal length in samples
    T = 1/fs # Sampling period
    n = nperseg # lag
    t = N / fs # Signal length in seconds
    flag = -10 # No freq. symbol
    # print(f'f_s :{fs}, T:{T}, N:{N}, n:{n}, t:{t}')

    window_size = n
    overlap_size = noverlap
    window_num = int((N - n) / (n - overlap_size)) + 1

    # print(f'window_num:{window_num}, overlap_size:{overlap_size}, window_size:{window_size}')

    # Initialize
    freqs, calrities, mae = {}, {}, {}
    gt, times = {}, []

    # Auto-correlation for gt, add times
    gt['freq'], gt['calrity'] = [], []
    # has_draw = [False]
    for i in range(window_num):
        frame_start = i * (window_size - overlap_size)
        frame_segment = force_seg[frame_start:frame_start+window_size]
        
        acf = sm.tsa.acf(frame_segment, nlags=n)
        acf_filtered = butter_filter(acf, cutoff=1) # cutoff=0.66
        peaks = sg.find_peaks(acf_filtered)[0]
        # peaks = sg.find_peaks(acf)[0] # Find peaks of the autocorrelation
        # peaks = sg.find_peaks(acf, height=np.median(acf) / 3, distance=fs)[0]
        # peaks = sg.find_peaks(acf, height=np.median(acf) / 3)[0]
        if peaks.any():
            lag = peaks[0] # Choose the first peak as our pitch component lag
            pitch = fs / lag # Transform lag into frequency
            clarity = acf_filtered[lag] / acf_filtered[0]

            peak_id = 0
            threshold = -0.1
            while (clarity < threshold or pitch*60 > 35) and peak_id < len(peaks): # pitch*60 > 35
                lag = peaks[peak_id]
                pitch = fs / lag
                clarity = acf_filtered[lag] / acf_filtered[0]
                peak_id += 1
            
            if clarity < threshold:
                pitch = flag

            # if pitch * 60 > 40 or pitch * 60 < 20:
            #     print(f"clarity: {clarity}, gt:{pitch * 60}")
            #     vs.draw_acf(acf, lag, frame_segment, acf_filtered=acf_filtered)

            # print(f"clarity: {clarity}, gt:{pitch * 60}")
            # vs.draw_acf(acf, lag, frame_segment, acf_filtered=acf_filtered)

            # print(has_draw[0])
            # print(i)
            # if not has_draw[0]:
            #     has_draw[0] = True
            #     vs.draw_acf(acf, lag, frame_segment, acf_filtered=acf_filtered)
            #     print(clarity)
                
            gt['freq'].append(pitch)
            gt['calrity'].append(clarity)
        else: # peaks is empty
            gt['freq'].append(flag)
            gt['calrity'].append(flag)
        
        times.append((start_t + (frame_start + frame_start + window_size)/2) / fs)

    if not return_t:
        return gt['freq']
    else:
        return gt['freq'], times
    
def gt_labelled_to_rr(gt_file):
    gt_labelled_rr = np.zeros(gt_file['file_len'])
    peaks_t, peaks_i = gt_file['peaks_t'], gt_file['peaks_i']

    # First 2 peaks
    gt_labelled_rr[:peaks_i[1]] = 1 / (peaks_t[1] - peaks_t[0])

    # Other peaks
    for i in range(1, len(peaks_i) - 2):
        gt_labelled_rr[peaks_i[i]:peaks_i[i+1]] = 1 / (peaks_t[i+1] - peaks_t[i])

    # Last 2 peaks
    gt_labelled_rr[peaks_i[-2]:] = 1 / (peaks_t[-1] - peaks_t[-2])

    return gt_labelled_rr
    
def compute_gt_labelled(gt_file_rr, delay=10, still_pt=300, fs=10, window_size=256, start_t=0, return_t=False):
    gt = [np.mean(gt_file_rr[int(start_t + delay*fs + still_pt): int(start_t + delay*fs + still_pt + window_size)])]
    time = [((start_t + window_size) / 2) / fs]

    if not return_t:
        return gt
    else:
        return gt, time

# window_size=1000, stride=500
# window_size=512, stride=256
# window_size=1024, stride=256
def segment_data(imu_data, force, window_size=128, stride=64, nperseg=128, noverlap=64, fs=10, return_t=False, out_1=False, labelled_path='./dataset/gt/all.pkl', file_name=None):
    """
    Create spectrogram windows from IMU data.

    Args:
        imu_data: (num_samples, 8) IMU data
        window_size: Number of IMU samples per spectrogram window
        stride: Step size for moving window
        
    Returns:
        segmented_spectrograms: (num_windows, 8, freq_bins, time_steps)
        segmented_gt: (num_windows, time_steps)
    """
    num_samples, num_channels = imu_data.shape
    # nperseg, noverlap = 128, 64
    windows = []
    gts = []
    t_ls = []

    if os.path.exists(labelled_path) and file_name is not None:
        gt_labelled = pickle.load(open(labelled_path, 'rb'))
        if file_name in gt_labelled:
            gt_file = gt_labelled[file_name]
            gt_file_rr = gt_labelled_to_rr(gt_file)
        else:
            gt_file_rr = None
    else:
        gt_file_rr = None

    for start in range(0, num_samples - window_size, stride):
        window_q = imu_data[start:start + window_size, :]
        window_force = force[start:start + window_size]
        spectrograms = compute_spectrogram(window_q, nperseg=nperseg, noverlap=noverlap)  # (8, freq_bins, time_steps)
        if not out_1:
            gt, t = compute_gt(window_force, nperseg=nperseg, noverlap=noverlap, start_t=start, return_t=True)
        else:
            gt, t = compute_gt(window_force, nperseg=window_size, noverlap=0, start_t=start, return_t=True)

        if gt_file_rr is not None:
            gt, t = compute_gt_labelled(gt_file_rr, window_size=window_size, start_t=start, return_t=True)

        windows.append(spectrograms)
        gts.append(gt)
        t_ls.append(t)
        
    if return_t:
        return np.stack(windows, axis=0), np.stack(gts, axis=0), np.stack(t_ls, axis=0)
    else:
        return np.stack(windows, axis=0), np.stack(gts, axis=0)  # (num_windows, 8, freq_bins, time_steps), (num_windows, time_steps)

# # 測試範例
# q_Sa = [0, 0, 0.707, 0.707]  # Sa 的初始四元數 (假設是 90 度旋轉)
# q_Sb = [0.707, 0, 0, 0.707]  # Sb 的初始四元數 (假設是另一個 90 度旋轉)

# # 計算校正四元數
# q_corr = quaternion_correction(q_Sa, q_Sb)
# print("校正四元數:", q_corr)

# # 校正 Sb 的四元數
# q_Sb_aligned = align_quaternion(q_corr, q_Sb)
# print("校正後的 Sb 四元數:", q_Sb_aligned)