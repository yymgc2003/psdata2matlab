import scipy.io as sio
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt

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
