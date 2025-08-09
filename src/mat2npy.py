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

    trigger_points = np.array(trigger_points)  # listからndarrayに変換

    print(f"Number of detected triggers: {trigger_points.shape}")

    return trigger_points, chunk, Fs

def arrange_trigger_points(trigger_points, window_width, signal_chunk, fs):
    trigger_points = np.array(trigger_points)
    window_samples = int(window_width * fs)
    triggered_pulses = []
    for trigger in trigger_points:
        if trigger + window_samples <= len(signal_chunk):
            pulse = signal_chunk[trigger:trigger + window_samples]
            triggered_pulses.append(pulse)
    triggered_pulses = np.array(triggered_pulses)
    #print(triggered_pulses.shape)
    return triggered_pulses

def convert_mat2npy(file_path,start_time,duration,amplitude_threshold,window_width,signal_key):
    amplitude_threshold = 2  # Amplitude threshold
    window_width = 0.1e-3  
    signal_key = "TDX1"
    triggers, signal_chunk, fs = detect_triggers_from_signal(
        file_path=file_path,
        start_time=start_time,
        duration=duration,
        amplitude_threshold=amplitude_threshold,
        window_width=window_width,
        signal_key=signal_key
    )
    print(f"triggers.shape: {triggers.shape}")
    print(f"signal_chunk.shape: {signal_chunk.shape}")
    print(f"fs: {fs}")
    mat_data = sio.loadmat(file_path)
    #signal_chunk=np.squeeze(mat_data)
    #print(signal_chunk.shape)
    #print(mat_data['TDX1'].shape)
    #print(mat_data['TDX1_enlarged'].shape)
    arranged_pulses_tdx1 = arrange_trigger_points(triggers, window_width, mat_data['TDX1'], fs)
    print(f"arranged_pulses_tdx1.shape: {arranged_pulses_tdx1.shape}")
    arranged_pulses_tdx2 = arrange_trigger_points(triggers, window_width, mat_data['TDX2'], fs)
    #print(f"arranged_pulses_tdx2.shape: {arranged_pulses_tdx2.shape}")  
    arranged_pulses_tdx3 = arrange_trigger_points(triggers, window_width, mat_data['TDX3'], fs)
    #print(f"arranged_pulses_tdx3.shape: {arranged_pulses_tdx3.shape}")  
    arranged_pulses_tdx1_enlarged = arrange_trigger_points(triggers, window_width, mat_data['TDX1_enlarged'], fs)
    print(f"arranged_pulses_tdx1_enlarged.shape: {arranged_pulses_tdx1_enlarged.shape}")  
    arranged_pulses = np.stack((arranged_pulses_tdx1, arranged_pulses_tdx2, arranged_pulses_tdx3, arranged_pulses_tdx1_enlarged), axis=2)
    print(f"arranged_pulses.shape: {arranged_pulses.shape}")
    return arranged_pulses,fs
