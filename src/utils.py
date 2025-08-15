import scipy.io as sio
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import h5py
def analyze_mat_file(file_path):
    """
    Load a .mat file and print all metadata
    """
    print(f"Analyzing: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Get file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"File size: {file_size:.2f} MB")
    
    # Load the .mat file
    try:
        mat_data = sio.loadmat(file_path)
        print("Successfully loaded .mat file")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Print file metadata
    print("\n=== File Metadata ===")
    for key in mat_data.keys():
        if key.startswith('__'):  # These are metadata keys
            print(f"{key}: {mat_data[key]}")
    
    # Print all variables in the file
    print("\n=== Variables ===")
    var_count = 0
    for key in mat_data.keys():
        if not key.startswith('__'):  # Skip metadata keys
            var_count += 1
            print(f"\nVariable {var_count}: {key}")
            value = mat_data[key]
            print(f"  Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Data type: {value.dtype}")
                
                # Print more details for object arrays
                if value.dtype == np.object_ or value.dtype.kind == 'O':
                    if value.size > 0:
                        first_elem = value.item() if value.size == 1 else value.flat[0]
                        print(f"  First element type: {type(first_elem)}")
                        if hasattr(first_elem, 'shape'):
                            print(f"  First element shape: {first_elem.shape}")
                
                # Print array statistics for numeric arrays
                if value.dtype.kind in 'ifu' and value.size > 0:  # integer, float, unsigned int
                    try:
                        print(f"  Min: {np.min(value)}")
                        print(f"  Max: {np.max(value)}")
                        print(f"  Mean: {np.mean(value)}")
                    except:
                        print("  Cannot compute statistics")
    
    print(f"\nTotal variables: {var_count}")
    
    # Deeper inspection of nested structures
    print("\n=== Nested Structures ===")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"\nStructure of: {key}")
            inspect_structure(mat_data[key])

def inspect_structure(obj, depth=0, max_depth=3):
    """
    Recursively inspect nested structures within a MATLAB object
    """
    prefix = "  " * depth
    
    if depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(obj, np.ndarray):
        print(f"{prefix}Array: shape={obj.shape}, dtype={obj.dtype}")
        
        # For object arrays, inspect the first element
        if obj.dtype == np.object_ or obj.dtype.kind == 'O':
            if obj.size > 0:
                print(f"{prefix}Contents:")
                first_elem = obj.item() if obj.size == 1 else obj.flat[0]
                inspect_structure(first_elem, depth + 1, max_depth)
    
    elif isinstance(obj, dict):
        print(f"{prefix}Dict with {len(obj)} keys")
        for k, v in obj.items():
            print(f"{prefix}Key: {k}")
            inspect_structure(v, depth + 1, max_depth)
    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} of length {len(obj)}")
        if len(obj) > 0:
            inspect_structure(obj[0], depth + 1, max_depth)
    
    else:
        print(f"{prefix}Value: {type(obj)}")


def plot_signal_waveform(file_path, start_ms=0, end_ms=100):
    """
    Display signal waveform from .mat file
    
    Parameters:
    -----------
    file_path : str
        Path to .mat file
    start_ms : float 
        Start time (milliseconds)
    end_ms : float
        End time (milliseconds)
    """
    # Load and inspect data
    mat_data = sio.loadmat(file_path)
    
    # Display available keys
    print("Available keys in file:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            print(f"- {key}: shape={mat_data[key].shape}")
    
    # Try different possible channel names
    channel_names = ['TDX1', 'A', 'B', 'Channel_A', 'Channel_1']
    signal = None
    
    for name in channel_names:
        if name in mat_data:
            signal = mat_data[name].flatten()
            print(f"Using channel: {name}")
            break
    
    if signal is None:
        raise ValueError("No valid channel found in the file")
    
    # Get sampling frequency
    if 'Tinterval' in mat_data:
        sampling_interval = mat_data['Tinterval'][0][0]  # sampling interval (s)
        sampling_rate = 1 / sampling_interval
        print(f"Sampling rate from file: {sampling_rate/1e6:.1f} MHz")
    else:
        print("Warning: Sampling rate not found, using default 62.5MHz")
        sampling_rate = 62.5e6  # Hz
    
    # Generate time axis (milliseconds)
    time_ms = np.arange(len(signal)) / sampling_rate * 1000
    
    # Limit display range
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_ms[mask], signal[mask], 'b-', linewidth=1)
    #plt.grid(True, which='both')
    #plt.minorticks_on()
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'Signal Waveform ({start_ms:.1f}-{end_ms:.1f} ms)\nSampling Rate: {sampling_rate/1e6:.1f} MHz')
    
    # Set x-axis ticks
    tick_interval = 0.1  # 0.1ms intervals
    plt.xticks(np.arange(start_ms, end_ms+tick_interval, tick_interval))
    
    # Display statistics
    signal_section = signal[mask]
    plt.text(0.02, 0.98, 
             f'Max: {np.max(signal_section):.3f}V\n'
             f'Min: {np.min(signal_section):.3f}V\n'
             f'RMS: {np.sqrt(np.mean(signal_section**2)):.3f}V',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
def npz2png(file_path, save_path, channel_index=1, start_time=0.0, end_time=None, full=True, pulse_index=0):    
    """
    Convert processed .npz signal data to PNG image.
    
    Parameters
    ----------
    file_path : str
        Path to the .npz file containing processed data.
    save_path : str
        Path to save the output PNG image.
    channel_index : int, optional
        Index of the channel to visualize (default is 0).
    start_time : float, optional
        Start time in seconds for visualization (default is 0.0).
    end_time : float or None, optional
        End time in seconds for visualization (default is None, meaning till the end).
    full : bool, optional
        If True, visualize all pulses as an image. If False, visualize only one pulse waveform (default is True).
    pulse_index : int, optional
        Index of the pulse to visualize when full=False (default is 0).
    
    Returns
    -------
    None
    """
    # .npzファイルからデータを読み込む
    data = np.load(file_path)
    processed_data = data["processed_data"]
    fs = data["fs"].item() if hasattr(data["fs"], "item") else float(data["fs"])
    
    # full=Trueの場合は全パルスを画像化
    if full:
        # processed_dataのshape: (n_pulses, n_samples, n_channels)
        # 指定チャンネルの全パルスを抽出
        if processed_data.ndim == 3:
            img_data = processed_data[:, :, channel_index]
        elif processed_data.ndim == 2:
            img_data = processed_data  # (n_pulses, n_samples)
        else:
            raise ValueError("processed_data shape is not supported.")
        
        # Determine the time axis range
        n_samples = img_data.shape[1]
        t = np.arange(n_samples) / fs
        if end_time is None:
            end_time = t[-1]
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        if end_idx > n_samples:
            end_idx = n_samples
        img_data = img_data[:, start_idx:end_idx]
        # Apply Hilbert transform to each pulse in img_data along the time axis
        # The analytic signal is computed for each pulse (row) individually
        # ヒルベルト変換をtorchで実装する
        import torch

        neglegible_time = 3e-6
        zero_samples = int(neglegible_time * fs)

        # img_data: (n_pulses, n_samples)
        # GPUにデータを転送
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_data_torch = torch.from_numpy(img_data).float().to(device)
        print(f"device: {device}")
        # 初期部分を0にする
        if zero_samples > 0:
            img_data_torch[:, :zero_samples] = 0

        # ヒルベルト変換のためのハイライザー（周波数領域での乗数）を作成
        n_samples = img_data_torch.shape[1]
        h = torch.zeros(n_samples, dtype=torch.complex64, device=device)
        if n_samples % 2 == 0:
            # 偶数長
            h[0] = 1
            h[1:n_samples//2] = 2
            h[n_samples//2] = 1
            # それ以降は0
        else:
            # 奇数長
            h[0] = 1
            h[1:(n_samples+1)//2] = 2
            # それ以降は0

        # FFT
        Xf = torch.fft.fft(img_data_torch, dim=1)
        # ハイライザーを掛ける
        Xf = Xf * h
        # IFFT
        analytic_signal = torch.fft.ifft(Xf, dim=1)
        # 振幅包絡を取得
        img_data_torch_abs = torch.abs(analytic_signal)
        # CPUに戻してnumpy配列に変換
        img_data = img_data_torch_abs.cpu().numpy()
        t = t[start_idx:end_idx]


        
        # 画像として保存
        plt.figure(figsize=(10, 4))
        plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (μs)')
        plt.ylabel('Pulse Number')
        plt.title('All Pulses (Channel {})'.format(channel_index))
        plt.tight_layout()
        import os
        base = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_save_path = os.path.join(base, f"{base_name}_{channel_index}img.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()
    else:
        # full=Falseの場合は指定パルスのみをプロット
        if processed_data.ndim == 3:
            pulse = processed_data[pulse_index, :, channel_index]
        elif processed_data.ndim == 2:
            pulse = processed_data[pulse_index, :]
        else:
            raise ValueError("processed_data shape is not supported.")
        n_samples = len(pulse)
        t = np.arange(n_samples) / fs
        if end_time is None:
            end_time = t[-1]
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        if end_idx > n_samples:
            end_idx = n_samples
        t = t[start_idx:end_idx]
        pulse = pulse[start_idx:end_idx]
        # Apply Hilbert transform to the pulse to obtain its analytic signal
        # The absolute value of the analytic signal gives the envelope of the pulse
        from scipy.signal import hilbert
        neglegible_time = 3e-6 # 3μss
        zero_samples = int(neglegible_time * fs)
        pulse[:zero_samples] = 0
        analytic_pulse = np.abs(hilbert(pulse))
        #print(pulse) 
        plt.figure(figsize=(10, 4))
        plt.plot(t*1e6, analytic_pulse, color='red', label='Envelope')
        plt.plot(t*1e6, pulse, color='blue', label='Original Pulse')
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
        plt.tight_layout()
        import os
        base = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        channel=channel_index
        new_save_path = os.path.join(base, f"{base_name}_{channel}pulse.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()



def analyze_mat_file_h5py(file_path):
    """
    Display metadata of a MATLAB v7.3 (HDF5-based) .mat file using h5py.

    Parameters
    ----------
    file_path : str
        Path to the .mat file (v7.3 format).
    Returns
    -------
    None
    """
    def print_attrs(name, obj):
        # Print the name and attributes of each object in the HDF5 file
        print(f"Name: {name}")
        for key, val in obj.attrs.items():
            print(f"  Attribute: {key} = {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group")

    with h5py.File(file_path, 'r') as f:
        print("Root keys (top-level groups/datasets):")
        for key in f.keys():
            print(f"  {key}")
        print("\nFull structure and attributes:")
        f.visititems(print_attrs)
