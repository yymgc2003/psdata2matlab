import numpy as np
import scipy.io as sio
import torch
from typing import List, Tuple, Optional

def detect_triggers_from_signal(
    file_path: str,
    start_time: float,
    duration: float,
    amplitude_threshold: float,
    window_width: float = 0.001,
    signal_key: str = "TDX1",
    interval_key: str = "Tinterval"
) -> Tuple[List[int], np.ndarray, float]:
    """
    Load signal waveform file and detect triggers within specified time range
    
    Parameters:
    -----------
    file_path : str
        Path to the MAT file
    start_time : float
        Start time in seconds
    duration : float
        Duration of detection range in seconds
    amplitude_threshold : float
        Amplitude threshold for trigger detection
    window_width : float, optional
        Minimum interval between triggers in seconds, default is 0.001
    signal_key : str, optional
        Key name for signal data in MAT file, default is "TDX1"
    interval_key : str, optional
        Key name for sampling interval in MAT file, default is "Tinterval"
    
    Returns:
    --------
    trigger_points : List[int]
        List of detected trigger point sample numbers
    chunk : np.ndarray
        Extracted signal chunk
    Fs : float
        Sampling frequency
    """
    
    # Load data
    print("Loading data...")
    try:
        mat_data = sio.loadmat(file_path)
        print("Loading successful")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load file: {e}")
    
    # Get signal data and sampling interval
    try:
        signal_data = np.squeeze(mat_data[signal_key])
        Tinterval = float(mat_data[interval_key].item())
        Fs = 1.0 / Tinterval
    except KeyError as e:
        raise KeyError(f"Specified key '{e}' not found in file")
    
    # Select GPU/CPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract data from specified time range
    start_idx = int(start_time * Fs)
    duration_samples = int(duration * Fs)
    
    # Check index range
    if start_idx >= len(signal_data):
        raise ValueError("Start time exceeds signal data range")
    
    end_idx = min(start_idx + duration_samples, len(signal_data))
    chunk = signal_data[start_idx:end_idx]
    
    # Transfer data to GPU
    chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.float32)
    
    # Detect positions exceeding threshold (GPU parallel processing)
    window_samples = int(window_width * Fs)
    
    # Detect positions exceeding threshold
    threshold_mask = torch.abs(chunk_tensor) >= amplitude_threshold
    potential_triggers = torch.where(threshold_mask)[0].cpu().numpy()
    
    # Select trigger points (avoid duplicates)
    trigger_points = []
    last_trigger = -window_samples
    
    for trigger in potential_triggers:
        if trigger > last_trigger + window_samples:
            trigger_points.append(trigger)
            last_trigger = trigger
    
    print(f"Number of detected triggers: {len(trigger_points)}")
    
    return trigger_points, chunk, Fs


def detect_triggers_from_array(
    signal_array: np.ndarray,
    sampling_rate: float,
    start_time: float,
    duration: float,
    amplitude_threshold: float,
    window_width: float = 0.001
) -> Tuple[List[int], np.ndarray]:
    """
    Detect triggers directly from NumPy array signal data
    
    Parameters:
    -----------
    signal_array : np.ndarray
        Signal data array
    sampling_rate : float
        Sampling frequency in Hz
    start_time : float
        Start time in seconds
    duration : float
        Duration of detection range in seconds
    amplitude_threshold : float
        Amplitude threshold for trigger detection
    window_width : float, optional
        Minimum interval between triggers in seconds, default is 0.001
    
    Returns:
    --------
    trigger_points : List[int]
        List of detected trigger point sample numbers
    chunk : np.ndarray
        Extracted signal chunk
    """
    
    # Select GPU/CPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract data from specified time range
    start_idx = int(start_time * sampling_rate)
    duration_samples = int(duration * sampling_rate)
    
    # Check index range
    if start_idx >= len(signal_array):
        raise ValueError("Start time exceeds signal data range")
    
    end_idx = min(start_idx + duration_samples, len(signal_array))
    chunk = signal_array[start_idx:end_idx]
    
    # Transfer data to GPU
    chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.float32)
    
    # Detect positions exceeding threshold
    window_samples = int(window_width * sampling_rate)
    threshold_mask = torch.abs(chunk_tensor) >= amplitude_threshold
    potential_triggers = torch.where(threshold_mask)[0].cpu().numpy()
    
    # Select trigger points (avoid duplicates)
    trigger_points = []
    last_trigger = -window_samples
    
    for trigger in potential_triggers:
        if trigger > last_trigger + window_samples:
            trigger_points.append(trigger)
            last_trigger = trigger
    
    return trigger_points, chunk


# Usage example
if __name__ == "__main__":
    # Example of detection from MAT file
    try:
        file_path = "/home/matsubara/database/signal_mat/P20240726-1600.mat"
        start_time = 0.0  # Start time in seconds
        duration = 1.0   # Duration in seconds
        amplitude_threshold = 2  # Amplitude threshold
        
        triggers, signal_chunk, fs = detect_triggers_from_signal(
            file_path=file_path,
            start_time=start_time,
            duration=duration,
            amplitude_threshold=amplitude_threshold,
            window_width=0.0001  # 0.1ms
        )
        
        print(f"Sampling frequency: {fs} Hz")
        print(f"Detected triggers: {len(triggers)}")
        print(f"Signal chunk length: {len(signal_chunk)} samples")
        
        # Display trigger times in seconds
        trigger_times = [t / fs for t in triggers]
        print(f"Trigger times (seconds): {len(trigger_times)}")
        
    except Exception as e:
        print(f"Error: {e}") 