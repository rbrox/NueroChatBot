import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_eeg_signals_with_avd(trial_number, channels_to_plot, zoom_y=1, output_dir="./images"):
    """
    Load EEG data from file and plot the signals for specified channels with arousal and valence values.

    Parameters:
        trial_number (int): The trial number to plot (0-indexed).
        channels_to_plot (list): List of channel numbers to plot (0-indexed).
        zoom_y (float): Factor to zoom the y-axis. Higher values zoom out, lower values zoom in. Defaults to 1.
        output_dir (str): Directory to save the plot.
    """
    file_path = "./data/s01.dat"

    # Load EEG data
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading the file: {str(e)}")

    # Check if 'data' and 'labels' exist in the dataset
    if 'data' not in data or 'labels' not in data:
        raise ValueError("The dataset does not contain the required EEG data or labels.")

    eeg_data = data['data']  # Shape: (trial, channel, time)
    avd_labels = data['labels']  # Shape: (trial, 4) - [valence, arousal, dominance, liking]

    # Validate trial and channel numbers
    if trial_number < 0 or trial_number >= eeg_data.shape[0]:
        raise ValueError(f"Invalid trial number. Must be between 0 and {eeg_data.shape[0] - 1}.")

    for channel_number in channels_to_plot:
        if channel_number < 0 or channel_number >= eeg_data.shape[1]:
            raise ValueError(f"Invalid channel number. Must be between 0 and {eeg_data.shape[1] - 1}.")

    # Extract arousal and valence for the trial
    valence = avd_labels[trial_number][0]
    arousal = avd_labels[trial_number][1]

    # Extract EEG data for the specified trial and channels
    selected_signals = [eeg_data[trial_number, channel, :] for channel in channels_to_plot]
    time_steps = np.arange(selected_signals[0].shape[0])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot signals in the same axes
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"EEG Signals for Trial {trial_number} - Valence: {valence:.2f}, Arousal: {arousal:.2f}", 
                 fontsize=16, color="white")

    # Define a color map for different channels
    colors = plt.cm.tab10(np.linspace(0, 1, len(channels_to_plot)))

    for i, signal in enumerate(selected_signals):
        # Normalize signal amplitude for zoom_y
        y_min, y_max = np.min(signal), np.max(signal)
        y_center = (y_min + y_max) / 2
        y_range = (y_max - y_min) / zoom_y
        normalized_signal = (signal - y_center) / y_range

        # Plot the signal
        ax.plot(time_steps, normalized_signal, label=f"Channel {channels_to_plot[i]}", color=colors[i])

    ax.set_ylabel("Normalized Amplitude", color="white")
    ax.set_xlabel("Time Steps", color="white")
    ax.set_title(f"Trial {trial_number} - Channels: {', '.join(map(str, channels_to_plot))}", color="white")
    ax.legend()
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    # Apply black background
    ax.set_facecolor("black")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Save the plot
    plot_file_name = f"trial_{trial_number}_channels_{'_'.join(map(str, channels_to_plot))}.jpg"
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_file_path, facecolor="black", dpi=300)
    plt.show()

    print(f"Plot saved successfully at: {plot_file_path}")
    print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")


if __name__ == "__main__":
    # Example usage
    trial_number = 0
    channels_to_plot = [0, 21, 22, 24]  # Specify channels as a Python list
    try:
        plot_eeg_signals_with_avd(trial_number, channels_to_plot, zoom_y=1.5)
    except Exception as e:
        print(f"Error: {str(e)}")
