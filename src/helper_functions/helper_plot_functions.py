import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from model_functions import *
from helper_functions import *


########################################################################################## plot entire sweep (raw signal)
def plot_whole_sweep(p_id, sweep_number, save=False):
    """
    Plots all 11 trials of one sweep for one participant
    ---
    p_id: participant ID (int)
    sweep_number: which sweep to plot (0–3)
    """
    
    PKL_PATH = r"PATH_TO_PKL.pkl"
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    distances = data["distances"]
    video_paths = data["video_path"]
    pid_str = f"P_{int(p_id):03d}"
    sweep_str = str(sweep_number)

    trial_order = [str(i) for i in range(1, 10)] + ["U1", "U2"]
    colors  = ["goldenrod"] + ["slateblue"] * 7 + ["crimson"] + ["limegreen"] * 2

    trial_lookup = {}
    sweep_type   = None
    for i, path in enumerate(video_paths):
        filename = os.path.basename(path)
        base = filename
        for suffix in ["_frontal.mp4", "_Frontal.mp4", ".mp4"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        parts = base.split("_")
        if len(parts) < 2:
            continue
        if base.upper().startswith(pid_str) and parts[-2] == sweep_str:
            trial_lookup[parts[-1].upper()] = distances[i]
            sweep_type = parts[-3].upper()

    processed = {}
    for trial_code in trial_order:
        if trial_code not in trial_lookup:
            continue
        processed[trial_code] = preprocess_signal(trial_lookup[trial_code])

    if not processed:
        print(f"No trials found for {pid_str}, sweep {sweep_number}")
        return

    all_signals = np.concatenate(list(processed.values()))
    y_min = np.min(all_signals) - 5
    y_max = np.max(all_signals) + 2

    sns.set_theme(style="white")
    fig, axes = plt.subplots(len(trial_order), 1, figsize=(10, 1.5 * len(trial_order)), sharex=True)

    for ax, trial_code, color in zip(axes, trial_order, colors):
        if trial_code not in processed:
            ax.set_visible(False)
            continue
        signal = processed[trial_code]
        ax.plot(np.arange(len(signal)), signal, linewidth=1.8, color=color)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Trial {trial_code}", fontweight="bold", color="darkslateblue")
        ax.set_ylabel("Avg Amp [deg]", color="slategrey")
        ax.grid(True)

    axes[-1].set_xlabel("Frame Number", color="slategrey")
    fig.suptitle(f"Finger Tapping Sweep {sweep_number} ({sweep_type}) - {pid_str}",
                 fontsize=18, fontweight="bold", color="darkslateblue")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save:
        fig.savefig(make_save_path("whole_sweep", p_id), format="pdf", bbox_inches="tight")
    plt.show()


########################################################################################## plot combination of front/side view    
def plot_single_trial(p_id, hand, condition, trial, save=False):
    """
    Plots frontal, lateral, and combined signals for one trial
    ---
    p_id: participant ID (int)
    hand: "L" or "R"
    condition: "AS" or "SA"
    trial: trial number or label string  (1-9, "U1", "U2")
    """
    PKL_PATH = r"C:\Users\emmal\VideoBased-PD-Biomarkers\data\TESTS\video_keypoints.pkl"
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    distances   = data["distances"]
    video_paths = data["video_path"]
    fps_list    = data["fps"]

    front_view_data   = {}
    lateral_view_data = {}

    for i, path in enumerate(video_paths):
        filename = os.path.basename(path)
        key      = filename.replace("_frontal.mp4", "").replace("_lateral.mp4", "")
        item     = {"distance": distances[i], "fps": fps_list[i], "path": path}
        if "frontal" in path.lower():
            front_view_data[key] = item
        elif "lateral" in path.lower():
            lateral_view_data[key] = item

    pid_str   = f"P_{int(p_id):03d}"
    trial_key = f"{pid_str}_FT_C_{hand.upper()}_{condition.upper()}_{trial}"

    if trial_key not in front_view_data or trial_key not in lateral_view_data:
        print(f"Trial not found: {trial_key}")
        return

    front_signal, side_signal_display, combined = combine_front_side_signals(
        front_view_data[trial_key]["distance"],
        lateral_view_data[trial_key]["distance"],
    )

    frames  = np.arange(len(front_signal))
    signals = [front_signal, side_signal_display, combined]
    titles  = ["Frontal View", "Lateral View", "Combined Signal"]
    colors  = ["khaki", "cornflowerblue", "plum"]

    all_signals = np.concatenate(signals)
    y_min = np.min(all_signals) - 5
    y_max = np.max(all_signals) + 5

    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)

    for ax, signal, title, color in zip(axes, signals, titles, colors):
        ax.plot(frames, signal, linewidth=1.8, color=color)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontweight="bold", color="darkslateblue")
        ax.set_ylabel("Avg Amplitude [deg]", color="slategrey")
        ax.grid(True)

    axes[-1].set_xlabel("Frame Number", color="slategrey")
    fig.suptitle(f"{pid_str} - {hand.upper()}_{condition.upper()} Trial {trial}",
                 fontsize=16, fontweight="bold", color="darkslateblue")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save:
        fig.savefig(make_save_path("raw_signal", p_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## scatter plot of avg cycle duration vs avg amp
def plot_speed_amplitude(part_id=None, show_groups=False, save=False):
    """
    Scatter plot of speed vs amplitude with marginal KDE distributions
    ---
    show_groups: True (highlights UPDRS and Trial 5 trials in different colours)
    part_id: None for all participants, int for one participant
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    mask_updrs, mask_trial_5, mask_rest = get_trial_masks(trial_numbers)

    sns.set_theme(style="white")
    fig, axs = plt.subplot_mosaic([["histx", "."], ["main", "histy"]], figsize=(7, 4), width_ratios=(2, 0.5), height_ratios=(0.5, 2), layout="constrained"
                                 )
    axs["histx"].tick_params(axis="x", labelbottom=False)
    axs["histy"].tick_params(axis="y", labelleft=False)
    axs["main"].scatter(speed[mask_rest], amplitude[mask_rest], color="khaki", alpha=0.6)
    
    if show_groups:
        axs["main"].scatter(speed[mask_trial_5], amplitude[mask_trial_5], color="palegreen", alpha=0.6, label="Trial 5")
        axs["main"].scatter(speed[mask_updrs], amplitude[mask_updrs], color="cornflowerblue", alpha=0.6, label="UPDRS (U1/U2)")

    set_axis_labels(axs["main"])
    axs["main"].grid(True)
    axs["main"].legend()

    sns.kdeplot(x=speed, ax=axs["histx"], color="plum", fill=True)
    sns.kdeplot(y=amplitude, ax=axs["histy"], color="plum", fill=True)
    
    if show_groups:
        sns.kdeplot(x=speed[mask_trial_5], ax=axs["histx"], color="lightseagreen", fill=True)
        sns.kdeplot(y=amplitude[mask_trial_5], ax=axs["histy"], color="palegreen", fill=True)
        sns.kdeplot(x=speed[mask_updrs], ax=axs["histx"], color="cornflowerblue", fill=True)
        sns.kdeplot(y=amplitude[mask_updrs], ax=axs["histy"], color="cornflowerblue", fill=True)

    axs["histx"].set_ylabel("")
    axs["histy"].set_xlabel("")
    fig.suptitle(participant_title(part_id), fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("scatter", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## plot pareto fronts
def plot_pareto_front(part_id=None, save=False):
    """
    Scatter plot with the true Pareto front and a smoothed curve
    --
    part_id: None for all participants, int for one participant
    """
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)

    x_pft, y_pft     = pareto_front_true(speed, amplitude)
    x_pft_s, y_pft_s = smooth_pareto_front(x_pft, y_pft)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.scatter(speed, amplitude, color="khaki", alpha=0.6)
    ax.plot(x_pft,   y_pft, "o-", color="palegreen", linewidth=2, label="Pareto Front (True)", alpha=0.7)
    ax.plot(x_pft_s, y_pft_s, "-", color="plum", linewidth=2, label="Smoothed Pareto Front (True)", alpha=0.7)

    set_axis_labels(ax)
    ax.set_ylim(np.min(amplitude) - 5, np.max(amplitude) + 5)
    ax.grid(True)
    ax.legend()
    fig.suptitle(participant_title(part_id), fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("pareto_front", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## plot all 4 models
def plot_prediction_models(part_id=None, save=False):
    """
    Fits all four models on a training split and plots their prediction lines
    ---
    part_id: None for all participants, int for one participant
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    ids = features["ids"].values

    speed_train, amp_train, speed_test, amp_test, ids_train, ids_test = participant_train_test_split(speed, amplitude, ids, test_size_participants=3)

    x_smooth = np.linspace(np.min(speed), np.max(speed), 200)
    model_specs = [("Linear", "cornflowerblue", "-"), ("Exponential", "lightgreen", "--"), ("Logarithmic", "plum", "-."), ("Logistic", "lightcoral", ":")]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.scatter(speed_train, amp_train, color="khaki", alpha=0.4, s=12, label="Train")
    ax.scatter(speed_test, amp_test, color="sandybrown", alpha=0.4, s=25, label="Test")

    for model_label, color, linestyle in model_specs:
        fit_fn, pred_fn = MODEL_FUNCS[model_label.lower()]
        model = fit_fn(speed_train, amp_train)
        y_pred = pred_fn(model, x_smooth)
        ax.plot(x_smooth, y_pred, linestyle, color=color, linewidth=2, label=model_label)

    set_axis_labels(ax)
    ax.grid(True)
    ax.legend()
    fig.suptitle(participant_title(part_id), fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("prediction_models", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## plot prediction curves on pareto-pooled data
def plot_pareto_prediction_models(save=False):
    """
    Fits all four models on Pareto-pooled training data and plots their prediction
    """
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, None, exclude_ids=[6, 16])
    ids = features["ids"].values

    speed_train, amp_train, speed_test, amp_test, ids_train, ids_test = participant_train_test_split(speed, amplitude, ids, test_size_participants=3)
    pareto_speed_train, pareto_amp_train, pareto_ids_train = pool_pareto_points(speed_train, amp_train, ids_train)

    x_smooth = np.linspace(min(np.min(speed_test), np.min(pareto_speed_train)), max(np.max(speed_test), np.max(pareto_speed_train)), 200)
    model_specs = [("Linear", "cornflowerblue", "-"), ("Exponential", "lightgreen", "--"), ("Logarithmic", "plum", "-."), ("Logistic", "lightcoral", ":")]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.scatter(speed_train, amp_train, color="khaki", alpha=0.4, s=12, label="Train (raw)")
    ax.scatter(speed_test, amp_test, color="sandybrown", alpha=0.4, s=25, label="Test (raw)")
    ax.scatter(pareto_speed_train, pareto_amp_train, color="salmon", alpha=0.8, s=30, label="Pareto Front (Train)")

    for model_label, color, linestyle in model_specs:
        fit_fn, pred_fn = MODEL_FUNCS[model_label.lower()]
        model  = fit_fn(pareto_speed_train, pareto_amp_train)
        y_pred = pred_fn(model, x_smooth)
        ax.plot(x_smooth, y_pred, linestyle, color=color, linewidth=2, label=model_label)

    set_axis_labels(ax)
    ax.set_ylim(np.min(amplitude) - 5, np.max(amplitude) + 5)
    ax.grid(True)
    ax.legend()
    fig.suptitle("All Participants (Pareto-Front Pooled)", fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("pareto_prediction_models"), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## plot best model summary
def plot_participant_models_mean_std(model_name="logarithmic", save=False):
    """
    Fits the chosen model on each participant's raw data individually,
    then plots all individual curves and the mean curve and ±1 SD band
    """
    
    fit_function, predict_function = MODEL_FUNCS[model_name]

    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp( CSV_FEATURES, CSV_METADATA, None, exclude_ids=[6, 16])
    ids = features["ids"].values
    x_smooth = np.linspace(np.min(speed), np.max(speed), 100)
    curves = {}

    for pid in np.unique(ids):
        pid_mask = (ids == pid)
        try:
            model = fit_function(speed[pid_mask], amplitude[pid_mask])
            curves[pid] = predict_function(model, x_smooth)
        except Exception as e:
            print(f"Skipping P{int(pid):03d}: fit failed ({e})")

    curve_matrix = np.array(list(curves.values()))
    mean_curve   = np.mean(curve_matrix, axis=0)
    std_curve    = np.std(curve_matrix,  axis=0)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for curve in curve_matrix:
        ax.plot(x_smooth, curve, color="salmon", alpha=0.2, linewidth=1.5)

    ax.scatter(speed, amplitude, color="khaki", alpha=0.4, s=12)
    ax.plot(x_smooth, mean_curve, color="cornflowerblue", linewidth=3, label="Mean Fit")
    ax.fill_between(x_smooth, mean_curve - std_curve, mean_curve + std_curve,
                    color="plum", alpha=0.4, label="+/- 1 SD")

    set_axis_labels(ax)
    ax.set_xlim(0.165, 1.125)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend()
    fig.suptitle(f"Speed-Amplitude Trade-off: {model_name.capitalize()} Fits per Participant", fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("pred_mean_std"), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## plot handedness comparison
def plot_handedness_models(part_id, save=False):
    """
    plots for dominant vs non-dominant hand
    """
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    is_dominant = get_handedness_mask(trial_names, handedness)
    x_smooth = np.linspace(np.min(speed), np.max(speed), 100)
    ymin = np.min(amplitude) - 5
    ymax = np.max(amplitude) + 5

    groups = [("Dominant Hand", is_dominant, "cornflowerblue"), ("Non-Dominant Hand", ~is_dominant, "palegreen")]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    for ax, (title, mask, color) in zip(axes, groups):
        model = fit_logarithmic_model(speed[mask], amplitude[mask])
        pred  = predict_logarithmic_model(model, x_smooth)

        ax.scatter(speed[mask], amplitude[mask], color="khaki", alpha=0.6)
        ax.plot(x_smooth, pred, color=color, linewidth=3, label="Logarithmic Fit", alpha=0.8)
        ax.set_title(title, color="darkslateblue")
        set_axis_labels(ax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"Speed-Amplitude Trade-off: Dominant vs Non-Dominant Hand (P_{int(part_id):03d})",
                 fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("handedness", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## sweep number comparison
def plot_sweep_number(part_id=None, model_name="logarithmic", save=False):
    """
    Plots prediction lines for each sweep number (0–3) 
    ---
    part_id: None for all participants, int for one participant
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    fit_function, predict_function = MODEL_FUNCS[model_name]
    sweep_numbers = np.array([t.split("_")[-2] for t in trial_names])

    sweep_colors = {"0": "royalblue", "1": "seagreen", "2": "darkorange", "3": "crimson"}
    x_smooth = np.linspace(np.min(speed), np.max(speed), 100)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    for sweep, color in sweep_colors.items():
        mask = (sweep_numbers == sweep)
        if not np.any(mask):
            continue
        model  = fit_function(speed[mask], amplitude[mask])
        pred   = predict_function(model, x_smooth)
        ax.scatter(speed[mask], amplitude[mask], color=color, alpha=0.3)
        ax.plot(x_smooth, pred, color=color, linewidth=2.5, label=f"Sweep {sweep} Fit", alpha=0.5)

    set_axis_labels(ax)
    ax.set_ylim(np.min(amplitude) - 5, np.max(amplitude) + 5)
    ax.grid(True)
    ax.legend()
    fig.suptitle(f"Speed-Amplitude Trade-off by Sweep ({participant_title(part_id)})",
                 fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("sweep_number", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## sweep type comparison 
def plot_sweep_condition(part_id=None, model_name="logarithmic", save=False):
    """
    Plots prediction lines for AS vs SA sweep conditions 
    ---
    part_id: None for all participants, int for one participant
    """
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    fit_function, predict_function = MODEL_FUNCS[model_name]
    condition = np.array([t.split("_")[-3] for t in trial_names])
    x_smooth  = np.linspace(np.min(speed), np.max(speed), 100)
    groups = [("AS", "cornflowerblue"), ("SA", "plum")]

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    for cond, color in groups:
        mask = (condition == cond)
        model = fit_function(speed[mask], amplitude[mask])
        pred = predict_function(model, x_smooth)
        ax.scatter(speed[mask], amplitude[mask], color=color, alpha=0.3)
        ax.plot(x_smooth, pred, color=color, linewidth=2.5,
                label=f"{cond} Fit ({model_name.capitalize()})", alpha=0.9)

    set_axis_labels(ax)
    ax.set_ylim(np.min(amplitude) - 5, np.max(amplitude) + 5)
    ax.grid(True)
    ax.legend()
    fig.suptitle(f"Speed-Amplitude Trade-off: AS vs SA ({participant_title(part_id)})",
                 fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("sweep_condition", part_id), format="pdf", bbox_inches="tight")
    plt.show()

########################################################################################## UPDRS comparison 
def plot_updrs(part_id=None, show_groups=True, save=False):
    """
    Scatter plot with highlighted UPDRS and Trial 5 trials
    class mean positions as diamond markers 
    ---
    show_groups: True to highlight UPDRS and Trial 5
    part_id: None for all participants, int for one participant
    """
    
    features, amplitude, speed, trial_names, trial_numbers, handedness, motor_skills = load_speed_amp(CSV_FEATURES, CSV_METADATA, part_id)
    mask_updrs, mask_trial_5, mask_rest = get_trial_masks(trial_numbers)

    model    = fit_logarithmic_model(speed[mask_rest], amplitude[mask_rest])
    x_smooth = np.linspace(np.min(speed), np.max(speed), 100)
    pred     = predict_logarithmic_model(model, x_smooth)

    mean_speed_updrs  = np.mean(speed[mask_updrs])
    mean_amp_updrs    = np.mean(amplitude[mask_updrs])
    mean_speed_trial5 = np.mean(speed[mask_trial_5])
    mean_amp_trial5   = np.mean(amplitude[mask_trial_5])

    pred_updrs         = predict_logarithmic_model(model, speed[mask_updrs])
    residuals_updrs    = amplitude[mask_updrs] - pred_updrs
    mae_updrs          = np.mean(np.abs(residuals_updrs))
    mean_residual_updrs = np.mean(residuals_updrs)

    sns.set_theme(style="white")
    fig, axs = plt.subplot_mosaic([["histx", "."], ["main", "histy"]], figsize=(7, 4), width_ratios=(2, 0.5), height_ratios=(0.5, 2), layout="constrained")
    
    axs["histx"].tick_params(axis="x", labelbottom=False)
    axs["histy"].tick_params(axis="y", labelleft=False)

    axs["main"].scatter(speed[mask_rest], amplitude[mask_rest], color="khaki", alpha=0.6)
    axs["main"].plot(x_smooth, pred, color="darkslateblue", linewidth=2.5,
                     label="Logarithmic Fit", alpha=0.9)

    if show_groups:
        axs["main"].scatter(speed[mask_trial_5], amplitude[mask_trial_5],
                            color="palegreen", alpha=0.6, label="Trial 5")
        axs["main"].scatter(speed[mask_updrs], amplitude[mask_updrs],
                            color="cornflowerblue", alpha=0.6, label="UPDRS (U1/U2)")
        axs["main"].scatter(mean_speed_updrs, mean_amp_updrs,  color="cornflowerblue",
                            s=45, marker="D", zorder=5, edgecolors="darkslateblue",
                            linewidths=1.5, label="UPDRS Mean")
        axs["main"].scatter(mean_speed_trial5, mean_amp_trial5, color="palegreen",
                            s=45, marker="D", zorder=5, edgecolors="darkslateblue",
                            linewidths=1.5, label="Trial 5 Mean")
        axs["main"].annotate(
            f"UPDRS MAE: {mae_updrs:.2f}°\nMean residual: {mean_residual_updrs:+.2f}°",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8, color="darkslateblue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="slategrey", alpha=0.8)
        )

    set_axis_labels(axs["main"])
    axs["main"].grid(True)
    axs["main"].legend(fontsize=7)

    sns.kdeplot(x=speed, ax=axs["histx"], color="plum", fill=True)
    sns.kdeplot(y=amplitude, ax=axs["histy"], color="plum", fill=True)
    if show_groups:
        sns.kdeplot(x=speed[mask_trial_5], ax=axs["histx"], color="lightseagreen", fill=True)
        sns.kdeplot(y=amplitude[mask_trial_5], ax=axs["histy"], color="palegreen", fill=True)
        sns.kdeplot(x=speed[mask_updrs], ax=axs["histx"], color="cornflowerblue", fill=True)
        sns.kdeplot(y=amplitude[mask_updrs], ax=axs["histy"], color="cornflowerblue", fill=True)

    axs["histx"].set_ylabel("")
    axs["histy"].set_xlabel("")
    fig.suptitle(participant_title(part_id), fontweight="bold", color="darkslateblue")

    if save:
        fig.savefig(make_save_path("updrs", part_id), format="pdf", bbox_inches="tight")
    plt.show()