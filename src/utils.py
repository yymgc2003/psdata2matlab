import scipy.io as sio
from scipy import signal
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import h5py
import json
import math
from scipy.signal import hilbert
import torch
import polars as pl 
from scipy.spatial.distance import pdist, squareform
import glob
def hilbert_cuda(img_data_torch, device, if_hilbert = True,
                 low_filter_freq = 0,
                 high_filter_freq = 1.0e9,
                 filter_order = 8,
                 fs = 52e6):
            """
            Compute the Hilbert envelope of input data using torch (GPU/CPU).

            Parameters
            ----------
            img_data_torch : torch.Tensor
                Input data of shape (n_pulses, n_samples), float32.
            device : torch.device
                Device to perform computation on.

            Returns
            -------
            np.ndarray
                Envelope of the analytic signal, shape (n_pulses, n_samples).
            """
            n_samples = img_data_torch.shape[1]
            # Create the Hilbert transformer in the frequency domain
            h = torch.zeros(n_samples, dtype=torch.complex64, device=device)
            if n_samples % 2 == 0:
                # Even length
                h[0] = 1
                h[1:n_samples//2] = 2
                h[n_samples//2] = 1
                # The rest remain zero
            else:
                # Odd length
                h[0] = 1
                h[1:(n_samples+1)//2] = 2
                # The rest remain zero
            #Xf[:,0:low_filter_idx] = 0
            #Xf[:,high_filter_idx:n_samples] = 0
            Xf = torch.fft.fft(img_data_torch, dim=1)
            # Apply the Hilbert transformer
            Xf = Xf * h
            # IFFT to get the analytic signal
            analytic_signal = torch.fft.ifft(Xf, dim=1)
            # Take the amplitude envelope
            img_data_torch_abs = torch.abs(analytic_signal)
            # Move to CPU and convert to numpy array
            return img_data_torch_abs.cpu().numpy()


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
def npz2png(file_path, save_path, channel_index=0, 
            start_time=0.0, end_time=None, full=True, 
            pulse_index=0, envelope=True, pulse_len=1000,
            start_pulse=5000, fft_plot=False):    
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
    print(f"processed_data.shape:{processed_data.shape}")
    # full=Trueの場合は全パルスを画像化
    if full:
        # processed_dataのshape: (n_pulses, n_samples, n_channels)
        # 指定チャンネルの全パルスを抽出
        if processed_data.ndim == 3:
            # Check if the channel_index is within the valid range
            if channel_index < 0 or channel_index >= processed_data.shape[2]:
                raise IndexError(f"channel_index {channel_index} is out of bounds for axis 2 with size {processed_data.shape[2]}")
            img_data = processed_data[:, :, channel_index]
        elif processed_data.ndim == 2:
            img_data = processed_data  # (n_pulses, n_samples)
        # If processed_data has other dimensions, raise an error
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
        img_data = img_data[start_pulse:start_pulse+pulse_len, start_idx:end_idx]
        # Apply Hilbert transform to each pulse in img_data along the time axis
        # The analytic signal is computed for each pulse (row) individually
        # ヒルベルト変換をtorchで実装する
        import torch

        neglegible_time = 3e-6
        zero_samples = int(neglegible_time * fs)

        img_data = filter_signal([1e6, 10e6], img_data, fs)

        # img_data: (n_pulses, n_samples)
        # GPUにデータを転送
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_data_torch = torch.from_numpy(img_data).float().to(device)
        #print(f"device: {device}")
        # 初期部分を0にする
        if zero_samples > 0:
            img_data_torch[:, :zero_samples] = 0

        # ヒルベルト変換のためのハイライザー（周波数領域での乗数）を作成
        n_samples = img_data_torch.shape[1]

        img_data = hilbert_cuda(img_data_torch, device)
        #print(img_data.shape)
        t = t[start_idx:end_idx]
        #print(t.shape)
        #print(np.max(img_data),np.min(img_data))
        
        
        plt.figure(figsize=(10, 10))
        plt.rcParams['font.size'] = 20
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=0,vmax=1)
        plt.imshow(img_data, aspect='auto', interpolation='nearest',cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=0,vmax=0.4)
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0], t[-1], img_data.shape[0]-0.5, -0.5])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (μs)')
        plt.ylabel('Pulse Number')
        plt.title('All Pulses (Channel {})'.format(channel_index))
        plt.tight_layout()
        import os
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_save_path = os.path.join(save_path, f"{base_name}_{channel_index}img.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()
    else:
        # full=Falseの場合は指定パルスのみをプロット
        print(f'processed_data.ndim = {processed_data.ndim}')
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
        pulse_raw = pulse.copy()
        # Apply Hilbert transform to the pulse to obtain its analytic signal
        # The absolute value of the analytic signal gives the envelope of the pulse
        from scipy.signal import hilbert
        neglegible_time = 3e-6 # 3μs
        zero_samples = int(neglegible_time * fs)

        pulse_new = np.array([pulse])

        pulse_new = filter_signal([1e6, 10e6], pulse_new, fs)
        pulse = pulse_new[0]

        pulse[:zero_samples] = 0
        if envelope:
            analytic_pulse = np.abs(hilbert(pulse))
        #analytic_pulse = np.log1p(analytic_pulse)
        #print(pulse) 
        plt.figure(figsize=(10, 4))
        plt.rcParams['font.size'] = 16
        if envelope:
            plt.plot(t*1e6, analytic_pulse, color='red', label='Envelope')
        plt.plot(t*1e6, pulse, color='blue', label='Original Pulse')
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
        plt.tight_layout()
        import os
        #base = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        channel=channel_index
        new_save_path = os.path.join(save_path, f"{base_name}_{channel}pulse.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()
        if fft_plot:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pulse_raw = torch.from_numpy(pulse_raw).float().to(device)
            pulse_fft = torch.fft.fft(pulse_raw)
            pulse_fft = torch.pow(torch.abs(pulse_fft), 2)
            pulse_fft = pulse_fft.cpu().numpy()
            n_samples = len(pulse_raw)
            freq = np.arange(n_samples)*fs/n_samples
            if end_time is None:
                end_time = freq[-1]
            start_idx = 0
            end_idx = n_samples//2
            if end_idx > n_samples:
                end_idx = n_samples
            freq = freq[0:n_samples//2]
            pulse_fft = pulse_fft[start_idx:end_idx]
            pulse_fft = pulse_fft/len(pulse_fft)/fs/2
            # Apply Hilbert transform to the pulse to obtain its analytic signal
            # The absolute value of the analytic signal gives the envelope of the pulse
            plt.figure(figsize=(10, 4))
            plt.rcParams['font.size'] = 16
            plt.plot(freq*1e-6, pulse_fft, color='blue', label='Original Pulse')
            plt.axvline(x=4, color='r',linestyle='--', linewidth=1.5, label='4 MHz')
            plt.legend()
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.ylim(-0.1*np.max(pulse_fft[40:n_samples//2]), 1.1*np.max(pulse_fft[40:n_samples//2]))
            label_text = '4 MHz'
            plt.text(plt.xlim()[1]*0.13, plt.ylim()[1]*1.1,label_text, 
            color='r', 
            fontsize=18,
            rotation=0,         # テキストの回転 (90度にすると縦書きになる)
            ha='left',          # Horizontal Alignment: 左寄せ
            va='top'            # Vertical Alignment: 上端合わせ
            )
            plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
            plt.tight_layout()
            import os
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            channel=channel_index
            new_save_path = os.path.join(save_path, f"{base_name}_{channel}fft.png")
            print(new_save_path)
            plt.savefig(new_save_path)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.rcParams['font.size'] = 16
            plt.plot(t*1e6, pulse, color='blue', label='FFT')
            plt.legend()
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
            plt.tight_layout()
            import os
            #base = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            channel=channel_index
            new_save_path = os.path.join(save_path, f"{base_name}_{channel}pulse.png")
            print(new_save_path)
            plt.savefig(new_save_path)
            plt.close()
def ndarray2png(signal_array, fs, save_path, channel_index=0, 
            start_time=0.0, end_time=None, full=True, 
            pulse_index=0, envelope=False, fft_plot=False):
    # .npzファイルからデータを読み込む
    processed_data = signal_array
    print(f"processed_data.shape:{processed_data.shape}")
    # full=Trueの場合は全パルスを画像化
    if full:
        # processed_dataのshape: (n_pulses, n_samples, n_channels)
        # 指定チャンネルの全パルスを抽出
        if processed_data.ndim == 3:
            # Check if the channel_index is within the valid range
            if channel_index < 0 or channel_index >= processed_data.shape[2]:
                raise IndexError(f"channel_index {channel_index} is out of bounds for axis 2 with size {processed_data.shape[2]}")
            img_data = processed_data[:, :, channel_index]
            n_samples = img_data.shape[1]
        elif processed_data.ndim == 2:
            img_data = processed_data  # (n_pulses, n_samples)
            n_samples = img_data.shape[1]
        # If processed_data has other dimensions, raise an error
        else:
            raise ValueError("processed_data shape is not supported.")
        
        # Determine the time axis range
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
        #print(f"device: {device}")
        # 初期部分を0にする
        if zero_samples > 0:
            img_data_torch[:, :zero_samples] = 0

        # ヒルベルト変換のためのハイライザー（周波数領域での乗数）を作成
        n_samples = img_data_torch.shape[1]

        img_data = hilbert_cuda(img_data_torch, device)
        #print(img_data.shape)
        t = t[start_idx:end_idx]
        #print(t.shape)
        #print(np.max(img_data),np.min(img_data))
        #img_data[img_data>0.1] = 0.1
        
        plt.figure(figsize=(10, 4))
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5],vmin=0,vmax=1)
        plt.imshow(img_data, aspect='auto', cmap='jet', extent=[t[0]*1e6, t[-1]*1e6, img_data.shape[0]-0.5, -0.5])
        #困ったらこのサイト:https://beiznotes.org/matplot-cmap-list/
        #plt.imshow(img_data, aspect='auto', cmap='viridis', extent=[t[0], t[-1], img_data.shape[0]-0.5, -0.5])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (μs)')
        plt.ylabel('Pulse Number')
        plt.title('All Pulses (Channel {})'.format(channel_index))
        plt.tight_layout()
        import os
        new_save_path = save_path + '_img.png'
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()
    else:
        # full=Falseの場合は指定パルスのみをプロット
        if fft_plot:
            print(f'processed_data.ndim = {processed_data.ndim}')
            if processed_data.ndim == 3:
                pulse = processed_data[pulse_index, :, channel_index]
            
            elif processed_data.ndim == 2:
                pulse = processed_data[pulse_index, :]
            elif processed_data.ndim == 1:
                pulse = processed_data
            else:
                raise ValueError("processed_data shape is not supported.")
            n_samples = len(pulse)
            freq = np.arange(n_samples)*fs/n_samples
            if end_time is None:
                end_time = freq[-1]
            start_idx = 0
            end_idx = n_samples//2
            if end_idx > n_samples:
                end_idx = n_samples
            freq = freq[0:n_samples//2]
            pulse = pulse[start_idx:end_idx]
            # Apply Hilbert transform to the pulse to obtain its analytic signal
            # The absolute value of the analytic signal gives the envelope of the pulse
            plt.figure(figsize=(10, 4))
            plt.plot(freq, pulse, color='blue', label='Original Pulse')
            plt.legend()
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
            plt.tight_layout()
            import os
            new_save_path = save_path + '_fft.png'
            print(new_save_path)
            plt.savefig(new_save_path)
            plt.close()
        else:
            print(f'processed_data.ndim = {processed_data.ndim}')
            if processed_data.ndim == 3:
                pulse = processed_data[pulse_index, :, channel_index]
            
            elif processed_data.ndim == 2:
                pulse = processed_data[pulse_index, :]
            elif processed_data.ndim == 1:
                pulse = processed_data
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
            neglegible_time = 3e-6 # 3μs
            zero_samples = int(neglegible_time * fs)
            pulse[:zero_samples] = 0
            if envelope:
                analytic_pulse = np.abs(hilbert(pulse))
            #analytic_pulse = np.log1p(analytic_pulse)
            #print(pulse) 
            plt.figure(figsize=(10, 4))
            if envelope:
                plt.plot(t*1e6, analytic_pulse, color='red', label='Envelope')
            plt.plot(t*1e6, pulse, color='blue', label='Original Pulse')
            plt.legend()
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.title('Pulse {} (Channel {})'.format(pulse_index, channel_index))
            plt.tight_layout()
            import os
            new_save_path = save_path + '_pulse.png'
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
def calculate_gvf_and_signal(config_path, npz_path, csv_path,
                             label_dim=2):
    """
    Calculate the gas volume fraction (GVF) and extract the signal from the given files.

    Parameters
    ----------
    config_path : str
        Path to the config.json file.
    npz_path : str
        Path to the processed .npz file.

    Returns
    -------
    input_tmp : np.ndarray
        The extracted signal (1D array).
    target_tmp : float
        The calculated gas volume fraction (GVF).
    """
    from scipy.spatial.distance import pdist
    import polars as pl

    with open(config_path, "r") as f:
        config = json.load(f)
    num = config["simulation"]["num_particles"]
    r_ball = config["simulation"]["glass_radius"] * 1e3
    r_pipe = config["pipe"]["inner_radius"]
    surface = math.pi * (r_pipe ** 2)
    height = (config["grid"]["Nz"]-20) * config["grid"]["dz"] *1e3
    v_pipe = surface

    # df = pl.read_csv(csv_path,has_header=False)
    v_sphere=0
    if label_dim==2:
        for i in range(num):
            cur_r_ball = r_ball**2 - ((df[i,2]-0.5)*height)**2
            if cur_r_ball > 0:
                v_sphere += math.pi*cur_r_ball
        gvf = v_sphere / v_pipe
    if label_dim==3:
        v_sphere = num*4*math.pi/3*r_ball**3

    # ーーーーーー変更部分ここからーーーーー
        
        # loc_arr = df.to_numpy()
        # loc_arr[:,0:2] *= r_pipe
        # loc_arr[:,2] *= height
        # dist_arr = pdist(loc_arr, metric='euclidean')
        # for dist in dist_arr:
        #     if dist < 2*r_ball:
        #         v_sphere -= 2*math.pi*(2/3*r_ball**3-r_ball**2*dist/2+1/3*(dist/2)**3)
        #height = config["grid"]["Nz"] * config["grid"]["dz"] *1e3
        v_pipe = surface*height
        gvf = v_sphere / v_pipe

    # ーーーーーー変更部分ここまでーーーーー

    #print(f"gvf: {gvf}")

    signal = np.load(npz_path)['processed_data']
    #print(f'if nan:{np.isnan(signal).any()}')
    #print(f'shape of signal:{np.shape(signal)}') 1*75000*1
    signal_tdx1 = signal[0, :, 0]
    #print(f'if nan:{np.isnan(signal_tdx1).any()}')
    #print(f"signal_tdx1: {signal_tdx1.shape}")
    input_tmp = signal_tdx1
    input_tmp_size = np.shape(input_tmp)[0]
    input_index_list = list(range(2500))
    for i in range(2500):
        input_index_list[i] = int(input_tmp_size/2500*i)
    #print(f'input_index_list: {input_index_list}')
    #print(f'if nan:{np.isnan(input_tmp).any()}')
    input_tmp_new2 =  [input_tmp[i] for i in input_index_list]#計2500になるようにデータを取得
    input_tmp_new2 = np.array(input_tmp_new2)
    #
    print(f'input_tmp: {input_tmp_new2.shape}')
    target_tmp = gvf
    if target_tmp < 0.0008:
        print(f"case:{config_path}")
        print(f"ball:{v_sphere}")
        print(f"v_pipe:{v_pipe}")
        print(f"gvf:{gvf}")
    return input_tmp_new2, target_tmp
def preprocess_and_predict(path, model, plot_index=80, device='cuda:0'):
    """
    Loads data from the given path, applies Hilbert transform and normalization,
    and runs prediction using the provided model.

    Args:
        path (str): Path to the .npz file containing 'processed_data'.
        model (torch.nn.Module): Trained PyTorch model for prediction.
        plot_index (int): Index of the sample to plot.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Model predictions.
    """
    import numpy as np
    import torch
    from scipy.signal import hilbert
    import matplotlib.pyplot as plt

    # Load and preprocess data
    x_raw = np.load(path)["processed_data"][:,:,0]
    print(x_raw.shape)
    #npz2png(file_path=path,save_path=output_folder_path,full=False,pulse_index=1)
    #npz2png(file_path=path,save_path=output_folder_path,full=True,pulse_index=2)
    print(f"max: {np.max(x_raw)}")
    #x_test = np.abs(hilbert(x_raw))
    x_raw_torch = torch.from_numpy(x_raw).float()
    x_raw_torch = x_raw_torch.to(device)
    x_test = hilbert_cuda(x_raw_torch,device)
    print(f"max: {np.max(x_test)}")
    if np.isnan(x_test).any():
        print("nan")
        x_test = np.nan_to_num(x_test)
    x_test_tensor = torch.from_numpy(x_test).float()

    # Add channel dimension: (batch, 1, length, channel)
    x_test_tensor_all = x_test_tensor.unsqueeze(1)
    print(x_test_tensor_all.shape)
    # Normalize each (length, channel) column for each sample in the batch
    max_values_per_column = torch.max(x_test_tensor_all, dim=2, keepdim=True)[0]
    print(f"max_values_per_column.shape: {max_values_per_column.shape}")
    max_values_per_column[max_values_per_column == 0] = 1.0  # Prevent division by zero
    x_test_tensor_all = x_test_tensor_all / max_values_per_column
    #print(f"max: {torch.max(x_test_tensor_all)}")

    # Use only the first channel for CNN input
    x_test_tensor_cnn = x_test_tensor_all[:, :, :]
    x_test_tensor_cnn = x_test_tensor_cnn.to(device)
    x_test_tensor_cnn = torch.log1p(x_test_tensor_cnn)
    #print(x_test_tensor.shape)
    print(x_test_tensor_cnn.shape)
    print(f"max: {torch.max(x_test_tensor_cnn)}")
    #print(x_test_tensor_cnn)
    # Plot a sample signal
    plt.figure(figsize=(10, 4))
    plt.plot(x_test_tensor_cnn[5, 0,:].cpu().numpy())
    plt.title("x_test_tensor_cnn Signal")
    plt.xlabel("sample Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    print(x_test_tensor_cnn[plot_index,0,:].shape)
    # Model prediction
    model.eval()
    with torch.no_grad():
        x_test_tensor_cnn = x_test_tensor_cnn.to(device)
        predictions = model(x_test_tensor_cnn)
        mean, var = torch.mean(predictions), torch.var(predictions)
        print(f"predictions.shape: {predictions.shape}")
        print(predictions)
        print(torch.mean(predictions), torch.var(predictions))
        # Release memory after computation
        del predictions
        torch.cuda.empty_cache()
    return mean, var

def ndarr2npz(processed_data, fs, save_dir, file_name):
    import numpy as np
    import os
    save_dict = {
        "processed_data": processed_data,
        "fs": fs
    }
    save_path = os.path.join(save_dir, file_name)
    np.savez(save_path, **save_dict)
    print(f"Processed data and metadata saved to: {save_path}")
    return save_path

def add_noise_to_dataset(x_train, noise_type="white"):
    import numpy as np
    #x_trainに関して、50usを2500要素に分割しているのを前提としている
    x_train_new = np.copy(x_train)
    size_arr = x_train.shape[0]
    rng = np.random.default_rng()
    x_max = np.max(x_train)
    #print(f'x_train max: {x_max}')
    if noise_type == "white":
        noise = rng.normal(loc=0, scale=x_max/100, size=size_arr)
    if noise_type == "pink":
        noise_tmp = rng.normal(loc=0, scale=x_max/10, size=size_arr)
        S = np.fft.rfft(noise_tmp)
        #print(f'size: {np.size(S)}')
        fil = 1 / (np.arange(len(S))+1)
        S = S * fil
        noise = np.fft.irfft(S)
        #print(f'size: {np.size(noise)}')
    x_train_new += noise
    return x_train_new

def rolling_window_signal(signal_input, window_size=10, 
                          padding=5, sampling_freq=50e6): 
    #signal_input: 1*2500
    dt = 1/sampling_freq
    t_end = np.shape(signal_input)[0] * dt
    max_index = np.argmax(signal_input) #これが管壁の一つ目の反射になるはず
    max_index -= int(1e-6/dt)
    #左端と右端で32mm=約40usの差があるのでこれをもとにその範囲を取り出す
    pipe_length_second = 40e-6
    last_index = int(pipe_length_second / dt)

def detect_overlap(rawsignal_dir=None):
    case_dirs = sorted([d for d in os.listdir(rawsignal_dir) 
                        if os.path.isdir(os.path.join(rawsignal_dir, d)) 
                        and d.startswith("case")])
    for case_name in case_dirs:
        base_dir = os.path.join(rawsignal_dir, case_name)
        if os.path.exists(os.path.join(base_dir, 'location_seed')):
            csv_dir = os.path.join(base_dir, 'location_seed')
        else:
            csv_dir = os.path.join(base_dir, 'location_seed1')
        print(csv_dir)
        with open(os.path.join(base_dir, 'config.json'),'r',encoding='utf-8') as f:
            config = json.load(f)
        Nz = config["grid"]["Nz"]
        dz = config["grid"]["dz"]*1e3
        inner_radius = config["pipe"]["inner_radius"]
        glass_radius = config["simulation"]["glass_radius"]*1e3 #単位はmm
        location_path = glob.glob(os.path.join(csv_dir, 'location*.csv'))
        print(f'Overlap in {case_name}: \n')
        for location_csv in location_path:
            location_df = pl.read_csv(os.path.join(csv_dir,location_csv), 
                                      has_header=False)
            location_arr = location_df.to_numpy()
            location_arr[:,0:2] *= inner_radius
            location_arr[:,2] *= dz*Nz
            pairwise_distances = pdist(location_arr, metric='euclidean')
            is_all_far_enough = np.min(pairwise_distances) > 2*glass_radius
            if not is_all_far_enough:
                print(f'    {os.path.basename(location_csv)}')

def filter_signal(filter_freq, x_raw, fs, device='cuda:0'):
    min_freq = filter_freq[0]
    max_freq = filter_freq[1]
    x_size = x_raw.shape[1]
    #print(f'x_raw shape: {x_raw.shape}')
    x_tensor = torch.from_numpy(x_raw).float()
    x_tensor = x_tensor.to(device)
    Xf = torch.fft.fft(x_tensor,dim=1)
    #print(f'Xf shape: {Xf.shape}')
    min_freq_idx = int(x_size*min_freq/fs)
    #print(f'min freq index: {min_freq_idx}')
    #print(f'Xf shape: {Xf.shape}')
    max_freq_idx = np.min([x_size*max_freq/fs, x_size//2-1])
    max_freq_idx = int(max_freq_idx)
    if min_freq_idx!=0 or max_freq_idx<x_size//2:
        #print(f'max freq index: {max_freq_idx}')
        Xf = torch.fft.fft(x_tensor,dim=1)
        Xf[:,0:min_freq_idx]=0
        Xf[:,max_freq_idx:x_size//2]=0
        #print(f'Xf shape: {Xf.shape}')
        x_tensor_new = torch.fft.ifft(Xf,dim=1)
        #print(f'x_tensor_new shape: {x_tensor_new.shape}')
        x_tensor_new = torch.real(x_tensor_new)
        return x_tensor_new.cpu().numpy()
    else:
        return x_raw