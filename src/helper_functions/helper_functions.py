import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_functions import *


CSV_FEATURES = r"PATH_TO_CSV.csv"
CSV_METADATA = r"PATH_TO_METADATA.csv"

MODEL_FUNCS = {"linear": (fit_linear_model, predict_linear_model),
               "exponential": (fit_exponential_model, predict_exponential_model),
               "logarithmic": (fit_logarithmic_model, predict_logarithmic_model),
               "logistic": (fit_logistic_model, predict_logistic_model)}


########################################################################################## load data
def load_speed_amp(csv_features, csv_metadata, participant_id=None, exclude_ids=None):
    """
    Load feature CSV and metadata CSV 
    ---
    participant_id: if given, only returns rows for that participant
    exclude_ids:    list of participant IDs to drop entirely
    """
    
    features = pd.read_csv(csv_features)
    metadata = pd.read_csv(csv_metadata)
    metadata = metadata.iloc[2:].reset_index(drop=True)

    if exclude_ids is not None:
        features = features[~features["ids"].isin(exclude_ids)]

    metadata_lookup = {}
    for _, row in metadata.iterrows():
        try:
            pid = int(row["1"])
        except (ValueError, KeyError):
            continue
        metadata_lookup[pid] = {"handedness": row["4"], "motor_skills": row["7_1"],}

    if participant_id is not None:
        features = features[features["ids"] == participant_id]

    amplitude = []
    speed = []
    trial_names  = []
    trial_numbers = []
    handedness  = []
    motor_skills = []

    for _, row in features.iterrows():
        pid = int(row["ids"])
        video_path = row["video_path"]

        filename = video_path.split("\\")[-1]
        filename = filename.replace("_frontal.mp4", "").replace("_lateral.mp4", "")

        trial_names.append(filename)
        trial_numbers.append(filename.split("_")[-1])
        amplitude.append(row["avg_amplitude"])
        speed.append(row["avg_cycle_duration"])

        meta = metadata_lookup.get(pid, {"handedness": np.nan, "motor_skills": np.nan})
        handedness.append(meta["handedness"])
        motor_skills.append(meta["motor_skills"])

    return (features,
           np.array(amplitude),
           np.array(speed),
           np.array(trial_names),
           np.array(trial_numbers),
           np.array(handedness),
           np.array(motor_skills))


########################################################################################## create masks 
def get_trial_masks(trial_numbers):
    """
    Given an array of trial number strings, returns boolean masks
    for UPDRS trials (U1/U2), trial 5, and all other trials
    """
    trial_numbers_arr = np.array(trial_numbers)
    mask_updrs   = (trial_numbers_arr == "U1") | (trial_numbers_arr == "U2")
    mask_trial_5 = (trial_numbers_arr == "5")
    mask_rest    = ~(mask_updrs | mask_trial_5)
    return mask_updrs, mask_trial_5, mask_rest


def get_handedness_mask(trial_names, handedness):
    """
    Returns a boolean array (True where the trial used the participant's dominant hand
    Handedness is coded as: 1 = left-handed, 2 = right-handed
    """
    
    trial_hand = np.array(["R" if "_R_" in t else "L" for t in trial_names])
    dominant_hand_code = np.array(["L" if str(h).strip() == "1" else "R" for h in handedness])
    return trial_hand == dominant_hand_code


########################################################################################## preprocess pkl raw data (for plot)
def preprocess_signal(raw_signal, kernel_size=5):
    """
    Interpolates NaN gaps; applies a median filter to remove spike noise
    """
    
    signal = pd.Series(np.array(raw_signal, dtype=float))
    signal = signal.interpolate().bfill().ffill().values
    signal = medfilt(signal, kernel_size=kernel_size)
    return signal

########################################################################################## cobine front/side view of pkl file data
def combine_front_side_signals(front_raw, side_raw):
    """
    Combines frontal and lateral view signals into one signal
    1. Track where NaNs are (before interpolation)
    2. Interpolate + median filter both signals
    3. Z-score both signals so they are on the same scale
    4. Blend 50/50, but if one view has a gap, use only the intact view there
    5. Rescale the combined signal back into the frontal view's degree range
    6. Also return the lateral signal rescaled to the frontal range for display
    """
    
    front_nan = np.isnan(np.array(front_raw, dtype=float))
    side_nan  = np.isnan(np.array(side_raw,  dtype=float))

    front_signal = preprocess_signal(front_raw)
    side_signal  = preprocess_signal(side_raw)

    min_len      = min(len(front_signal), len(side_signal))
    front_signal = front_signal[:min_len]
    side_signal  = side_signal[:min_len]
    front_nan    = front_nan[:min_len]
    side_nan     = side_nan[:min_len]

    front_mean, front_std = np.mean(front_signal), np.std(front_signal)
    side_mean,  side_std  = np.mean(side_signal),  np.std(side_signal)

    front_scaled = (front_signal - front_mean) / front_std
    side_scaled  = (side_signal  - side_mean)  / side_std

    combined_scaled = 0.5 * front_scaled + 0.5 * side_scaled
    combined_scaled = np.where(front_nan & ~side_nan, side_scaled,  combined_scaled)
    combined_scaled = np.where(side_nan  & ~front_nan, front_scaled, combined_scaled)

    combined            = combined_scaled   * front_std + front_mean
    combined            = medfilt(combined, kernel_size=5)
    side_signal_display = side_scaled * front_std + front_mean

    return front_signal, side_signal_display, combined

########################################################################################## make smooth curve out of pareto front points
def smooth_pareto_front(x, y, num_points=200, s=200):
    """
    Fits a smooth spline through Pareto front points for visualisation
    """
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    if len(x_sorted) < 4:
        return x_sorted, y_sorted

    spline   = UnivariateSpline(x_sorted, y_sorted, s=s)
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    return x_smooth, spline(x_smooth)

########################################################################################## pareto-pooling for global evaluation
def pool_pareto_points(speed, amplitude, ids):
    """
    For each participant, extracts their true Pareto front points, pool them all together, then remove outliers using IQR 
    """
    
    ids = np.array(ids)
    pooled_speed     = []
    pooled_amplitude = []
    pooled_ids       = []

    for pid in np.unique(ids):
        mask    = (ids == pid)
        pf_speed, pf_amp = pareto_front_true(speed[mask], amplitude[mask])
        pooled_speed.extend(pf_speed)
        pooled_amplitude.extend(pf_amp)
        pooled_ids.extend([pid] * len(pf_speed))

    pooled_speed = np.array(pooled_speed)
    pooled_amplitude = np.array(pooled_amplitude)
    pooled_ids = np.array(pooled_ids)

    keep_mask = np.ones(len(pooled_speed), dtype=bool)
    for arr in [pooled_speed, pooled_amplitude]:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr     = q3 - q1
        keep_mask &= (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)

    return pooled_speed[keep_mask], pooled_amplitude[keep_mask], pooled_ids[keep_mask]

########################################################################################## make train test split
def holdout_split(speed, amplitude, ids, n_holdout=5, random_state=42):
    """
    Splits data into train and holdout sets, keeping all trials of the same participant together 
    """
    
    ids = np.array(ids)
    test_frac = n_holdout / len(np.unique(ids))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    train_idx, holdout_idx = next(gss.split(speed, amplitude, groups=ids))

    return (speed[train_idx], amplitude[train_idx], ids[train_idx],
            speed[holdout_idx], amplitude[holdout_idx], ids[holdout_idx])

def participant_train_test_split(speed, amplitude, ids, test_size_participants=3, random_state=42):
    """
    Splits data into train and test sets.
    If multiple participants: keeps all trials of the same participant together
    If only one participant: does a random 80/20 split 
    """
    ids       = np.array(ids)
    n_total   = len(np.unique(ids))

    if n_total > 1:
        test_frac = test_size_participants / n_total
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
        train_idx, test_idx = next(gss.split(speed, amplitude, groups=ids))
    else:
        all_idx = np.arange(len(speed))
        train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=random_state)

    return (speed[train_idx],  amplitude[train_idx],  ids[train_idx],
            speed[test_idx],   amplitude[test_idx],   ids[test_idx])


########################################################################################## cross validation
def evaluate_model_cv(fit_function, predict_function, speed, amplitude, cv_splitter, groups=None):
    """
    cross-validation for one model and returns average RMSE, MAE, R^2, Spearman
    cv_splitter: a sklearn splitter object (e.g. KFold, GroupKFold)
    groups: participant IDs — required for GroupKFold, not needed for plain KFold
    """
    rmse_scores, mae_scores, r2_scores, spearman_scores = [], [], [], []

    split_args = (speed,) if groups is None else (speed, amplitude, groups)
    for train_idx, test_idx in cv_splitter.split(*split_args):
        x_train, x_test = speed[train_idx], speed[test_idx]
        y_train, y_test = amplitude[train_idx], amplitude[test_idx]

        model  = fit_function(x_train, y_train)
        y_pred = predict_function(model, x_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        spearman_scores.append(spearmanr(y_test, y_pred)[0])

    return {"RMSE": (np.mean(rmse_scores), np.std(rmse_scores)),
            "MAE": (np.mean(mae_scores), np.std(mae_scores)),
            "R2": (np.mean(r2_scores), np.std(r2_scores)),
            "Spearman": (np.mean(spearman_scores), np.std(spearman_scores))}

def print_metrics(model_name, metrics):
    print(f"{model_name} ====================")
    print(f"RMSE:     {metrics['RMSE'][0]:.4f} +/- {metrics['RMSE'][1]:.4f}")
    print(f"MAE:      {metrics['MAE'][0]:.4f} +/- {metrics['MAE'][1]:.4f}")
    print(f"R^2:      {metrics['R2'][0]:.4f} +/- {metrics['R2'][1]:.4f}")
    print(f"Spearman: {metrics['Spearman'][0]:.4f} +/- {metrics['Spearman'][1]:.4f}\n")

########################################################################################## handedness curve
def fit_handedness_curves(model_name="logarithmic", exclude_ids=None):

    fit_function, predict_function = MODEL_FUNCS[model_name]

    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, None, exclude_ids=exclude_ids)
    ids = features["ids"].values
    is_dominant = get_handedness_mask(trial_names, handedness)
    x_smooth = np.linspace(np.min(speed), np.max(speed), 100)

    dom_curves = {}
    nondom_curves = {}

    for pid in np.unique(ids):
        pid_mask = (ids == pid)
        dom_mask = pid_mask & is_dominant
        nondom_mask = pid_mask & ~is_dominant

        if np.sum(dom_mask) < 3 or np.sum(nondom_mask) < 3:
            print(f"Skipping P{int(pid):03d}: not enough trials per hand")
            continue

        dom_model = fit_function(speed[dom_mask], amplitude[dom_mask])
        nondom_model = fit_function(speed[nondom_mask], amplitude[nondom_mask])

        dom_curves[pid] = predict_function(dom_model, x_smooth)
        nondom_curves[pid] = predict_function(nondom_model, x_smooth)

    return x_smooth, dom_curves, nondom_curves

########################################################################################## within-subject variability
def within_subject_variability(exclude_ids=None):
    """
    Computes mean, std, and CV (coefficient of variation = std/mean) for amplitude and speed per participant
    checks heteroscedasticity
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, None, exclude_ids=exclude_ids)
    ids = features["ids"].values
    results = []

    for pid in np.unique(ids):
        pid_mask = (ids == pid)
        amp_p = amplitude[pid_mask]
        spd_p = speed[pid_mask]

        mean_amp = np.mean(amp_p)
        mean_spd = np.mean(spd_p)

        results.append({"participant": f"P{int(pid):03d}",
                        "mean_amp": mean_amp,
                        "std_amp": np.std(amp_p),
                        "cv_amp": np.std(amp_p) / mean_amp,
                        "mean_spd": mean_spd,
                        "std_spd": np.std(spd_p),
                        "cv_spd": np.std(spd_p) / mean_spd})

    df = pd.DataFrame(results)

    print("=== Within-Subject Variability ===\n")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n--- Aggregated ---")
    print(f"CV Amplitude: mean={df['cv_amp'].mean():.4f}, std={df['cv_amp'].std():.4f}, "
          f"min={df['cv_amp'].min():.4f}, max={df['cv_amp'].max():.4f}")
    print(f"CV Speed:     mean={df['cv_spd'].mean():.4f}, std={df['cv_spd'].std():.4f}, "
          f"min={df['cv_spd'].min():.4f}, max={df['cv_spd'].max():.4f}")

    corr_amp, p_amp = spearmanr(df["mean_amp"], df["std_amp"])
    corr_spd, p_spd = spearmanr(df["mean_spd"], df["std_spd"])

    print(f"\n--- Heteroscedasticity (Spearman: mean vs std) ---")
    print(f"Amplitude: r={corr_amp:.4f}, p={p_amp:.4f} "
          f"({'heteroscedastic' if p_amp < 0.05 else 'no evidence of heteroscedasticity'})")
    print(f"Speed:     r={corr_spd:.4f}, p={p_spd:.4f} "
          f"({'heteroscedastic' if p_spd < 0.05 else 'no evidence of heteroscedasticity'})")

    return df

########################################################################################## UPDRS vs trial 5 / prediction line
def updrs_trial5_summary(exclude_ids=None):
    """
    For each participant: computes mean amplitude/speed for UPDRS and Trial 5, 
    the difference between them, and the UPDRS prediction error against the fitted Logarithmic model
    
    Positive residual means UPDRS fell above the fitted curve
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, None, exclude_ids=exclude_ids)
    ids = features["ids"].values
    mask_updrs, mask_trial_5, mask_rest = get_trial_masks(trial_numbers)

    rows = []
    for pid in np.unique(ids):
        pid_mask = (ids == pid)
        amp_updrs = amplitude[pid_mask & mask_updrs]
        spd_updrs = speed[pid_mask & mask_updrs]
        amp_trial5 = amplitude[pid_mask & mask_trial_5]
        spd_trial5 = speed[pid_mask & mask_trial_5]
        amp_rest = amplitude[pid_mask & mask_rest]
        spd_rest = speed[pid_mask & mask_rest]

        if len(amp_updrs) == 0 or len(amp_trial5) == 0 or len(amp_rest) < 3:
            continue

        model = fit_logarithmic_model(spd_rest, amp_rest)
        pred_updrs = predict_logarithmic_model(model, spd_updrs)
        residuals = amp_updrs - pred_updrs

        rows.append({"participant": f"P{int(pid):03d}",
                     "mean_amp_updrs": np.mean(amp_updrs),
                     "mean_spd_updrs": np.mean(spd_updrs),
                     "mean_amp_trial5": np.mean(amp_trial5),
                     "mean_spd_trial5": np.mean(spd_trial5),
                     "amp_diff": np.mean(amp_updrs) - np.mean(amp_trial5),
                     "spd_diff": np.mean(spd_updrs) - np.mean(spd_trial5),
                     "mae_updrs": np.mean(np.abs(residuals)),
                     "mean_residual_updrs":  np.mean(residuals),})

    df = pd.DataFrame(rows)
    print("=== UPDRS vs Trial 5 Summary ===\n")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n--- Aggregated ---")
    print(f"Mean amplitude difference (UPDRS - Trial 5): {df['amp_diff'].mean():.4f} +/- {df['amp_diff'].std():.4f}°")
    print(f"Mean speed difference (UPDRS - Trial 5): {df['spd_diff'].mean():.4f} +/- {df['spd_diff'].std():.4f}s")
    print(f"UPDRS MAE vs fitted line: {df['mae_updrs'].mean():.4f} +/- {df['mae_updrs'].std():.4f}°")
    print(f"UPDRS mean residual vs fitted line: {df['mean_residual_updrs'].mean():.4f} +/- {df['mean_residual_updrs'].std():.4f}°")
    direction = "above" if df["mean_residual_updrs"].mean() > 0 else "below"
    print(f"=> UPDRS trials tend to fall {direction} the fitted tradeoff curve")

    return df

########################################################################################## smallr helper functions
def participant_title(part_id):
    return f"P_{int(part_id):03d}" if part_id is not None else "All Participants"


def make_save_path(plot_type, part_id=None, output_dir="supplementary figures"):
    os.makedirs(output_dir, exist_ok=True)
    suffix   = f"P{int(part_id):03d}" if part_id is not None else "all"
    filename = f"{plot_type}_{suffix}.pdf"
    return os.path.join(output_dir, filename)


def set_axis_labels(ax, x_label="Average Cycle Duration [s]", y_label="Average Amplitude [deg]"):
    ax.set_xlabel(x_label, color="slategrey")
    ax.set_ylabel(y_label, color="slategrey")