import numpy as np
from scipy import signal as sg
from scipy.spatial.transform import Rotation as R
import statsmodels.api as sm

try:
     from . import visualize as vs
except:
     import visualize as vs

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

def auto_correlation(data, outputs, cols=['q_x', 'q_y', 'q_z', 'q_w'], fs=10, window=10, overlap=5, visualize=True):
    N = len(data) # Signal length in samples
    T = 1/fs # Sampling period
    n = window * fs # lag (window size)
    t = N / fs # Signal length in seconds
    flag = -10 # No freq. symbol
    print(f'f_s :{fs}, T:{T}, N:{N}, n:{n}, t:{t}')

    window_size = n
    overlap_size = overlap * fs
    window_num = int((N - n) / (n - overlap_size)) + 1

    print(f'window_num:{window_num}, overlap_size:{overlap_size}, window_size:{window_size}')

    # Initialize
    freqs, calrities, mae = {}, {}, {}
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
        peaks = sg.find_peaks(acf)[0] # Find peaks of the autocorrelation
        if peaks.any():
            lag = peaks[0] # Choose the first peak as our pitch component lag
            pitch = fs / lag # Transform lag into frequency
            clarity = acf[lag] / acf[0]
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
                peaks = sg.find_peaks(acf)[0] # Find peaks of the autocorrelation
                if peaks.any():
                    lag = peaks[0] # Choose the first peak as our pitch component lag
                    pitch = fs / lag # Transform lag into frequency
                    clarity = acf[lag] / acf[0]
                    freqs[method][col].append(pitch * 60)
                    calrities[method][col].append(clarity)
                else: # peaks is empty
                    freqs[method][col].append(flag)
                    calrities[method][col].append(flag)
    
    # Estimate error(MAE)
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
                        min_idx = np.argmin(mae_sample_ls)
                        mae_ls.append(mae_sample_ls[min_idx])
                        preds[method].append(pred_sample_ls[min_idx])
                    else:
                        preds[method].append(flag)
        if mae_ls:
            mae[method] = np.mean(mae_ls)
            preds[method] = np.array(preds[method])

    # Convert to numpy array
    gt['freq'] = np.array(gt['freq'])
    gt['calrity'] = np.array(gt['calrity'])
    times = np.array(times)

    # Draw results
    if visualize:
        vs.draw_autocorrelation_results(preds, gt, times, cols=cols)
    
    return mae

# # 測試範例
# q_Sa = [0, 0, 0.707, 0.707]  # Sa 的初始四元數 (假設是 90 度旋轉)
# q_Sb = [0.707, 0, 0, 0.707]  # Sb 的初始四元數 (假設是另一個 90 度旋轉)

# # 計算校正四元數
# q_corr = quaternion_correction(q_Sa, q_Sb)
# print("校正四元數:", q_corr)

# # 校正 Sb 的四元數
# q_Sb_aligned = align_quaternion(q_corr, q_Sb)
# print("校正後的 Sb 四元數:", q_Sb_aligned)