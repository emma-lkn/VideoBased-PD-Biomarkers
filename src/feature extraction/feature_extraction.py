import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from math import pi
import pickle
from scipy.stats import f_oneway, shapiro, ttest_ind, mannwhitneyu
from scipy.signal import butter, filtfilt, medfilt
from scipy.signal import welch, correlate
from scipy.stats import entropy
import matplotlib.cm as cm
from scipy.stats import spearmanr
from scipy.stats import zscore

sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)

class feature_ext_analysis:
    def __init__(self, config):
        self.config = config
    
        
    def _extract_features(self, distances, fps):
         feature_names = [
            'avg_amplitude',
            'avg_percycle_max_speed',
            'avg_percycle_avg_speed',
            'avg_cycle_duration',
            'amp_slope',
            'cycle_slope',
            'speed_slope',
            'cov_cycle_duration',
            'cov_amp',
            'cov_percycle_max_speed',
            'cov_percycle_avg_speed',
            'num_interruptions',
         ]
         
         distances_ = np.array(distances, dtype=float) # interpolation later needs float arrays
         
         nan_mask = np.isnan(distances_)
         if nan_mask.all():
             return [np.nan] * len(feature_names), ['ids', 'video_path', 'label'] + feature_names
         if nan_mask.any():
             indices = np.arange(len(distances_))
             distances_[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], distances_[~nan_mask])
 
         distances_ = medfilt(distances_, kernel_size=5) # TRY: changing kernel size

         fs = 60.0                
         order = 3 # CHANGED reduced from 4                
         nyq = 0.5 * fs   

         cutoff_frequencies =  6.0 # changed from 9 to 6; more aggressive to jitter from mediapipe
         normal_cutoff = cutoff_frequencies / nyq  
         b, a = butter(order, normal_cutoff, btype='low', analog=False)
         filtered_signal = filtfilt(b, a, distances_)
         distances_ = filtered_signal
         print(f"After butter: mean={np.mean(distances_):.3f}, std={np.std(distances_):.3f}, nan_count={np.sum(np.isnan(distances_))}")
 
 
         ##################################### TEMPORAL CUTTING 
         # replaced peak/trough detection
         # TRY: changing distance (og=5)
         # mean isnt very robust; so switched to std (better for movement variability/motion magnitude)
         # prominence: better than absolute height since it ignores baseline drift/offsets/slow trends
         signal_std_cut = np.std(distances_)
         peaks_cut, _ = find_peaks(distances_,distance=8,prominence=0.10 * signal_std_cut)
         troughs_cut, _ = find_peaks(-distances_,distance=8,prominence=0.10 * signal_std_cut)
         
         # since some later computations depend on AT LEAST 2 detected peaks we add this safeguard
         if len(peaks_cut) < 2 or len(troughs_cut) < 2:
            return [np.nan] * len(feature_names), ['ids', 'video_path', 'label'] + feature_names   
         
         # TEMPORAL CUTTING: find middle peak, take 300 ((60fps x 10sec)/2) frames in both drections from there
         mid_frame = (peaks_cut[0] + peaks_cut[-1]) // 2
         start = max(0, mid_frame - 250)
         end = min(len(distances_), mid_frame + 250)           
         distances_ = distances_[start:end]
         
         # Re-detection of peaks after cutting
         signal_std = np.std(distances_)
         peaks, _ = find_peaks(distances_, distance=8, prominence=0.10 * signal_std)
         troughs, _ = find_peaks(-distances_, distance=8, prominence=0.10 * signal_std)
         
        
         ############################################################## Compute speed signal
         time_interval = 1 / fps
         speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime
         
         ######### peak detection check AGAIN and NaN safeguard
         while len(peaks) > 0 and len(troughs) > 0 and troughs[0] > peaks[0]:
             peaks = peaks[1:]
        
         if len(peaks) < 2 or len(troughs) < 2:
             return [np.nan] * len(feature_names), ['ids', 'video_path', 'label'] + feature_names
    
         ############################################################## Compute amplitude
         amplitudes = []
         amp_frame_numbers = []

         for i in range(min(len(peaks), len(troughs))):
             peak = peaks[i]
             valid_troughs = troughs[troughs < peak]
             if len(valid_troughs) == 0:
                 continue
             last_trough = valid_troughs[-1]
             amp = abs(distances_[peak] - distances_[last_trough])
             amplitudes.append(amp)
             amp_frame_numbers.append(peak)

         amp_frame_numbers = np.array(amp_frame_numbers).reshape(-1, 1)
         avg_amplitude = np.mean(amplitudes)

         time_points_amp = np.arange(len(amplitudes)).reshape(-1, 1)
         model_amp = LinearRegression()
         model_amp.fit(time_points_amp, amplitudes)
         amp_slope = model_amp.coef_[0]######################################################################################

         ################################################################# Generate per-cycle speed 
         per_cycle_speed_maxima = []
         per_cycle_speed_avg = []
         per_cycle_speed_avg_frame_numbers = []
         for i in range(len(amplitudes) - 1):
             start_idx = peaks[i]  # Start of the window
             end_idx = peaks[i + 1]  # End of the window
             window_speed = speed_signal[start_idx:end_idx]  # Slice the speed signal
             
             if len(window_speed) > 0:  # Ensure the window is not empty
                 per_cycle_speed_maxima.append(np.percentile(np.abs(window_speed), 95))
                 per_cycle_speed_avg.append(np.mean(np.abs(window_speed)))
                 # mid-frame to regress the average values (just for viz/trend)
                 avg_frame = (start_idx + end_idx) // 2
                 per_cycle_speed_avg_frame_numbers.append(avg_frame)
         
         
         # Compute the median and max of per-cycle speed maxima
         mean_percycle_max_speed = np.mean(per_cycle_speed_maxima)
         mean_percycle_avg_speed = np.mean(per_cycle_speed_avg)
         per_cycle_speed_avg_frame_numbers = np.array(per_cycle_speed_avg_frame_numbers).reshape(-1, 1)
         avg_speed = np.mean(np.abs(speed_signal))
         
         time_points_speed = np.arange(len(per_cycle_speed_avg)).reshape(-1, 1)
         model_speed = LinearRegression()
         model_speed.fit(time_points_speed, np.abs(per_cycle_speed_avg))
         speed_slope = model_speed.coef_[0]#####################################################################################
         
         # Compute tapping intervals (time between consecutive maxima)
         tapping_intervals = np.diff(peaks) / fps

         median_tapping_interval = np.median(tapping_intervals)
         mean_tapping_interval = np.mean(tapping_intervals)

         time_points_ti = np.arange(len(tapping_intervals)).reshape(-1, 1)
         model_ti = LinearRegression()
         model_ti.fit(time_points_ti, tapping_intervals)
         ti_slope = model_ti.coef_[0]

         ################################## ratio decrement
         mid = len(amplitudes) // 2
         first_half = amplitudes[:mid]
         second_half = amplitudes[mid:]
         mid = len(np.abs(speed_signal)) // 2
         first_half = np.abs(speed_signal)[:mid]
         second_half = np.abs(speed_signal)[mid:]
         
         ####################################################################   hesitation-halts
         std_tapping_intervals = np.std(tapping_intervals)
         cov_tapping_interval = std_tapping_intervals/mean_tapping_interval

         std_amp = np.std(amplitudes)
         cov_amp = std_amp/avg_amplitude

         std_per_cycle_speed_maxima = np.std(per_cycle_speed_maxima)
         cov_per_cycle_speed_maxima = std_amp/mean_percycle_max_speed

         std_per_cycle_speed_avg = np.std(per_cycle_speed_avg)
         cov_per_cycle_speed_avg = std_per_cycle_speed_avg/mean_percycle_avg_speed
         
         std_speed = np.std(speed_signal)
         cov_speed = std_speed/avg_speed

         # Compute total number of interruptions
         threshold = 2 * median_tapping_interval
         num_interruptions1 = sum(interval > threshold for interval in tapping_intervals)
         
         threshold = 1.5 * median_tapping_interval
         num_interruptions2 = sum(interval > threshold for interval in tapping_intervals)
         
         ################################################################################################
         features = {'avg_amplitude': avg_amplitude,
                     'avg_percycle_max_speed':mean_percycle_max_speed,
                     'avg_percycle_avg_speed':mean_percycle_avg_speed,
                     'avg_cycle_duration':mean_tapping_interval,
                     'amp_slope':amp_slope, 
                     'cycle_slope':ti_slope,
                     'speed_slope':speed_slope,
                     'cov_cycle_duration':cov_tapping_interval, 
                     'cov_amp':cov_amp, 
                     'cov_percycle_max_speed':cov_per_cycle_speed_maxima,
                     'cov_percycle_avg_speed':cov_per_cycle_speed_avg,
                     'num_interruptions':num_interruptions2,
                     }

         feat_name = ['ids', 'video_path', 'label'] + list(features.keys())
         return list(features.values()),  feat_name


    def load_data(self):
            with open(self.config['annotated_pkl_path'], 'rb') as f:
                annotated_data = pickle.load(f)
            
            combined_video_paths = annotated_data['video_path'] 
            combined_distances = annotated_data['distances'] 
            combined_keypoints = annotated_data['keypoints'] 
            combined_ids = annotated_data['id'] 
            combined_labels = annotated_data['label'] 
            combined_fps = annotated_data['fps'] 
            
            all_data = []
            
            items = []
            for i in range(len(combined_video_paths)):
                items.append({
                    'distance': combined_distances[i],
                    'fps': combined_fps[i],
                    'label': combined_labels[i],
                    'id': combined_ids[i],
                    'path': combined_video_paths[i]
                })
            print(f"Total items loaded: {len(items)}")
            
            for item in tqdm(items):
                print(f"Processing: {item['path']}")
                features, feat_name = self._extract_features(item['distance'], item['fps'])
                row = [
                    item['id'],
                    item['path'],
                    item['label']
                ] + features

                all_data.append(row)
                     
            final_column_names = feat_name
            df = pd.DataFrame(all_data, columns=final_column_names)
            
            filename = f"finger_tapping_features.csv" # removed some stuff
            self.csv_save_path = os.path.join(self.config['save_path'], filename)
            df.to_csv(self.csv_save_path, index=False)
     
        
if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))  # repo/src/feature_extraction
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))  # repo root

    CONFIG = {
        'annotated_pkl_path': os.path.join(project_root, "data/raw/video_keypoints.pkl"),
        'save_path': os.path.join(project_root, "data/processed"),
    }

    os.makedirs(CONFIG['save_path'], exist_ok=True)  
   
    feature_ext_ana = feature_ext_analysis(CONFIG)
    feature_ext_ana.load_data()
