import numpy as np
import scipy.io as sio
import torch
from typing import List, Tuple, Optional
import os
import json
import h5py
from scipy.signal import hilbert
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
    chunk = chunk_tensor.cpu().numpy()
    chunk=chunk.flatten()
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
    # ===== Emphasized Section: Always make trigger_points 1D =====
    trigger_points = np.array(trigger_points).flatten()  # Always make trigger_points 1D
    # ===== Emphasized Section: Remove all size-1 dimensions from signal_chunk =====
    signal_chunk = np.squeeze(signal_chunk)  # Remove all size-1 dimensions from signal_chunk
    #print(f"trigger_points.shape: {trigger_points.shape}")  # Check shape after flattening
    #print(f"signal_chunk.shape(after): {signal_chunk.shape}")  # Check shape after squeezing
    print(f"window width: {window_width}")
    window_samples = int(window_width * fs)
    triggered_pulses = []
    print(f"window_samples: {window_samples}")
    #print(len(signal_chunk))
    #print(trigger_points)
    signal_chunk_tmp=signal_chunk.copy()
    for trigger in trigger_points:
        if trigger + window_samples <= len(signal_chunk_tmp):
            pulse = signal_chunk_tmp[trigger:trigger + window_samples]
            triggered_pulses.append(pulse)
            #print(f'pulse.shape:{pulse.shape}')
        else:
            pulse = signal_chunk_tmp[trigger:]
            #triggered_pulses.append(pulse)
            #print(f'pulse.shape:{pulse.shape}')
    triggered_pulses = np.array(triggered_pulses)
    #print(f"triggered_pulses.shape: {triggered_pulses.shape}")
    #print(triggered_pulses[1,3000])
    #print(f"triggered_pulses.shape: {triggered_pulses.shape}")
    return triggered_pulses
def convert_exp(file_path,start_time,duration,amplitude_threshold,
                window_width,num_samples_per_pulse,signal_key):
    amplitude_threshold = 2  # Amplitude threshold  
    signal_key = "TDX1"
    triggers, signal_chunk, fs = detect_triggers_from_signal(
        file_path=file_path,
        start_time=start_time,
        duration=duration,
        amplitude_threshold=amplitude_threshold,
        window_width=window_width,
        signal_key=signal_key
    )
    #print(f"triggers.shape: {triggers.shape}")
    #print(f"signal_chunk.shape: {signal_chunk.shape}")
    #print(f"fs: {fs}")
    mat_data = sio.loadmat(file_path)
    #signal_chunk=np.squeeze(mat_data['TDX1'])
    #print(f"signal_chunk.shape(before): {signal_chunk.shape}")
    #print(mat_data['TDX1'])
    #print(mat_data['TDX1'].shape)
    #print(mat_data['TDX1_enlarged'].shape)
    if np.isinf(mat_data['TDX1']).any():
        print("Warning: arranged_pulses contains inf values. Replacing infs with 0.")
        mat_data['TDX1'] = np.nan_to_num(mat_data['TDX1'], nan=0.0)
    if np.isnan(mat_data['TDX1']).any():
        print("Warning: arranged_pulses contains nan values. Replacing nans with 0.")
        mat_data['TDX1'] = np.nan_to_num(mat_data['TDX1'], nan=0.0)
    #print(f"max: {np.max(mat_data['TDX1'])}")
    #print(f"min: {np.min(mat_data['TDX1'])}")
    arranged_pulses_tdx1 = arrange_trigger_points(triggers, window_width, mat_data['TDX1'],fs)
    #print(f"arranged_pulses_tdx1.shape: {arranged_pulses_tdx1.shape}")
    arranged_pulses_tdx2 = arrange_trigger_points(triggers, window_width, mat_data['TDX2'], fs)
    #print(f"arranged_pulses_tdx2.shape: {arranged_pulses_tdx2.shape}")  
    arranged_pulses_tdx3 = arrange_trigger_points(triggers, window_width, mat_data['TDX3'],fs)
    #print(f"arranged_pulses_tdx3.shape: {arranged_pulses_tdx3.shape}")  
    arranged_pulses_tdx1_enlarged = arrange_trigger_points(triggers, window_width, mat_data['TDX1_enlarged'],fs)
    #print(f"arranged_pulses_tdx1_enlarged.shape: {arranged_pulses_tdx1_enlarged.shape}")  
    arranged_pulses = np.stack((arranged_pulses_tdx1, arranged_pulses_tdx2, arranged_pulses_tdx3, arranged_pulses_tdx1_enlarged), axis=2)
    print(f"arranged_pulses.shape: {arranged_pulses.shape}")
    
    #print(np.max(arranged_pulses))
    return arranged_pulses,fs
def mat2npz_sim(file_path, config_path, output_dir):
    """
    Convert simulation .mat (HDF5) file and config.json to .npz format for further analysis.
    
    Parameters
    ----------
    file_path : str
        Path to the simulation .mat file (HDF5 format).
    config_path : str
        Path to the config.json file containing simulation metadata.
    output_dir : str
        Directory to save the processed .npz file.
    Returns
    -------
    save_path : str
        Path to the saved .npz file.
    """
    import h5py
    import json
    import numpy as np
    import os

    # Load simulation config from JSON
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Open the .mat (HDF5) file and inspect structure
    #with h5py.File(file_path, 'r') as g:
    #    z_group = g['#refs#/z']
    #    print(g.keys())  # Print top-level keys for inspection
    #    print(list(z_group.keys()))  # Print keys in z group

    # Extract simulation parameters from config
    end_time = config["simulation"]["t_end"]
    cfl = config["simulation"]["CFL"]
    sound_speed = config["medium"]["water"]["sound_speed"]
    dx = config["grid"]["dx"]
    dz = config["grid"]["dz"]
    dt = cfl * dx / sound_speed
    fs = 1 / dt
    print(fs)

    # Try to get sensor_data or fallback to z group
    with h5py.File(file_path, 'r') as g:
        if 'sensor_data' in g:
            # If 'sensor_data' exists as a dataset, use it
            sensor_data = g['sensor_data'][:]
            processed_data = sensor_data[15]
            # Get all top-level keys in the file
            keys = list(g.keys())
            print(f"keys:", keys)
        else:
            # If 'sensor_data' does not exist, use '#refs#/z' group
            z_group = g['#refs#/z']
            keys = list(z_group.keys())
            key_15 = keys[15] if len(keys) > 15 else keys[0]
            print(f"key_15:", key_15)
            processed_data = z_group[key_15][:]
            # Collect all datasets in z_group as a list
            sensor_data = [z_group[k][:] for k in keys]

    print(keys)
    # Reshape processed_data to [1, :, 1] for consistency
    # [number of measurements, sensor values, sensor index, (optional) vertical vector]
    # Todo: implement scan_line function of kwave
    processed_data = sensor_data[np.newaxis, :, 15, np.newaxis]
    processed_data_size = processed_data.shape[1]
    processed_data=processed_data[:,processed_data_size//2:,:]
    #processed_data=np.abs(hilbert(processed_data[:,50001:,:]))
    #processed_data=processed_data[::20]
    print(processed_data[0, :, 0].shape)  # Confirm the shape of the signal values

    # Prepare dictionary for saving
    save_dict = {
        "processed_data": processed_data,
        "fs": fs,
        "original_keys": keys,
        # Add other metadata here if needed
    }

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(output_dir, f"{base_filename}_processed.npz")
    np.savez(save_path, **save_dict)
    print(f"Processed data and metadata saved to: {save_path}")
    return save_path

def mat2npz_sim_2d(file_path, config_path, output_dir):
    """
    Convert simulation .mat (HDF5) file and config.json to .npz format for further analysis.
    
    Parameters
    ----------
    file_path : str
        Path to the simulation .mat file (HDF5 format).
    config_path : str
        Path to the config.json file containing simulation metadata.
    output_dir : str
        Directory to save the processed .npz file.
    Returns
    -------
    save_path : str
        Path to the saved .npz file.
    """
    import h5py
    import json
    import numpy as np
    import os
    import re
    from .utils import npz2png

    # Load simulation config from JSON
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Open the .mat (HDF5) file and inspect structure
    #with h5py.File(file_path, 'r') as g:
    #    print(g.keys())  # Print top-level keys for inspection
    #    v_group = g['#refs#/v']
    #    print(list(v_group.keys()))  # Print keys in z group

    # Extract simulation parameters from config
    end_time = config["simulation"]["t_end"]
    cfl = config["simulation"]["CFL"]
    sound_speed = config["medium"]["water"]["sound_speed"]
    dx = config["grid"]["dx"]
    dy = config["grid"]["dy"]
    dt = cfl * dx / sound_speed
    fs = 1 / dt
    print(fs)

    # Try to get sensor_data or fallback to z group
    with h5py.File(file_path, 'r') as g:
        if 'sensor_data' in g:
            # If 'sensor_data' exists as a dataset, use it
            sensor_data = g['sensor_data/p'][:]
            print(f"Shape of sensor_data:", np.size(sensor_data[15]))
            processed_data = sensor_data[15]
            # Get all top-level keys in the file
            keys = list(g.keys())
            print(f"keys:", keys)
        else:
            # If 'sensor_data' does not exist, use '#refs#/z' group
            v_group = g['#refs#/v']
            keys = list(v_group.keys())
            key_15 = keys[15] if len(keys) > 15 else keys[0]
            print(f"key_15:", key_15)
            processed_data = v_group[key_15][:]
            # Collect all datasets in z_group as a list
            sensor_data = [v_group[k][:] for k in keys]

    print(keys)
    # Reshape processed_data to [1, :, 1] for consistency
    # [number of measurements, sensor values, sensor index, (optional) vertical vector]
    # Todo: implement scan_line function of kwave
    print(f'sensor_data shape:{sensor_data.shape}')
    ref_data, trans_data = np.split(sensor_data, 2, axis=1)
    ref_data = np.mean(ref_data, axis=1)
    trans_data = np.mean(trans_data, axis=1)
    ref_processed_data = ref_data[np.newaxis, :, np.newaxis]
    ref_processed_data=ref_processed_data[:,46875:,:]
    trans_processed_data = trans_data[np.newaxis, :, np.newaxis]
    trans_processed_data=trans_processed_data[:,46875:,:]
    #processed_data=np.abs(hilbert(processed_data[:,50001:,:]))
    #processed_data=processed_data[::20]
    print(ref_processed_data[0, :, 0].shape)  # Confirm the shape of the signal values
    print(f'new processed data shape:{ref_processed_data.shape}')

    # Prepare dictionary for saving
    save_dict = {
        "processed_data": trans_processed_data,
        "fs": fs,
        "original_keys": keys,
        # Add other metadata here if needed
    }

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    num = re.findall(r'\d+', base_filename)[0]
    base_filename = base_filename.replace('reflector'+num,'')
    base_filename = base_filename + 'receiver' + num
    save_path = os.path.join(output_dir, f"{base_filename}_processed.npz")
    np.savez(save_path, **save_dict)
    print(f"Processed data and metadata saved to: {save_path}")
    #npz2png(file_path=save_path, save_path=output_dir,
    #        full=False, pulse_index=0)

    # Prepare dictionary for saving
    save_dict = {
        "processed_data": ref_processed_data,
        "fs": fs,
        "original_keys": keys,
        # Add other metadata here if needed
    }

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(output_dir, f"{base_filename}_processed.npz")
    np.savez(save_path, **save_dict)
    print(f"Processed data and metadata saved to: {save_path}")
    #npz2png(file_path=save_path, save_path=output_dir,
    #        full=False, pulse_index=0)
    return save_path

def mat2npz_exp(file_path, output_dir, start_time=0.0, duration=5.0, 
                amplitude_threshold=2, window_width=0.2e-3, 
                num_samples_per_pulse=2500, signal_key="TDX1"):
    """
    Convert experimental .mat data to .npz format and save with metadata.

    Parameters
    ----------
    file_path : str
        Path to the experimental .mat file.
    output_dir : str
        Directory to save the processed .npz file.
    start_time : float, optional
        Start time for signal extraction (default is 0.0).
    duration : float, optional
        Duration for signal extraction (default is 5.0).
    amplitude_threshold : float, optional
        Amplitude threshold for trigger detection (default is 2).
    window_width : float, optional
        Window width in seconds for pulse extraction (default is 0.1e-3).
    signal_key : str, optional
        Key for the signal channel to use for trigger detection (default is "TDX1").

    Returns
    -------
    save_path : str
        Path to the saved .npz file.
    """
    # Convert the experimental data using convert_exp
    raw_data, fs = convert_exp(
        file_path,
        start_time=start_time,
        duration=duration,
        amplitude_threshold=amplitude_threshold,
        window_width=window_width,
        num_samples_per_pulse=num_samples_per_pulse,
        signal_key=signal_key
    )
    print("convert_exp finished")
    print(f"max: {np.max(raw_data)}")
    processed_data=raw_data[100:14800,2500:,:]
    print(f"processed_data.shape: {processed_data.shape}")
    print(f"max: {np.max(processed_data)}")
    # English comment: Check for NaN values and replace them with 0 to avoid np.max returning nan
    if np.isnan(processed_data).any():
        print("Warning: processed_data contains NaN values. Replacing NaNs with 0.")
        processed_data = np.nan_to_num(processed_data, nan=0.0)
    if np.isinf(processed_data).any():
        print("Warning: processed_data contains inf values. Replacing infs with 0.")
        processed_data = np.nan_to_num(processed_data, nan=0.0)
    i=0
    print(f"processed_data[{i},:,0].shape: {processed_data[i,:,0].shape}")
    print(f"max: {np.max(processed_data[i,:,0])}")
    print(f"argmax: {np.argmax(processed_data[i,:,0])}")
    max_per_sample = np.max(processed_data, axis=1, keepdims=True)
    print(f"maxes argmax: {np.argmax(max_per_sample)},max: {np.max(max_per_sample)}")
    print(max_per_sample.shape, np.min(max_per_sample),np.max(max_per_sample))
    max_per_sample[max_per_sample == 0] = 1.0
    print(f"scaled: {processed_data.shape,np.min(processed_data),np.max(processed_data)}")
    #processed_data = processed_data / max_per_sample

    max_value = np.max(processed_data)
    if np.isnan(max_value):
        print("Error: np.max(processed_data) is still NaN after NaN replacement.")
    else:
        print(f"max_value: {max_value}")
    mat_data = sio.loadmat(file_path)
    keys = list(mat_data.keys())
    # Print the keys for inspection
    print(keys)
    # Print the shape of the first measurement for confirmation
    print(f"signal points: {processed_data[0, :, 0].shape}")  # [first measurement, sensor values, first sensor]
    # Prepare a dictionary to save both data and metadata
    save_dict = {
        "processed_data": processed_data,
        "fs": fs,
        "original_keys": keys,
        # Add other metadata here if needed
    }
    # Generate the save path
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(output_dir, f"{base_filename}_processed.npz")
    # Save the data as .npz
    np.savez(save_path, **save_dict)
    print(f"Processed data and metadata saved to: {save_path}")
    return save_path

