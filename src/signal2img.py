import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import hilbert


def generate_echomap(file_path, start_time=0.444, duration=0.001, 
                                    window_width=100e-6, amplitude_threshold=2.0, 
                                    output_dir=None):
    """
    Function to extract waveforms for a fixed time after amplitude reaches threshold
    Accelerate processing using GPU
    
    Parameters:
    -----------
    file_path : str
        Path to .mat file
    start_time : float
        Start time (seconds)
    duration : float
        Analysis time width (seconds)
    window_width : float
        Width of extraction window (seconds), default 50μs
    amplitude_threshold : float
        Amplitude threshold for triggering
    output_dir : str
        Directory to save output images, default is None (no saving)
    
    Returns:
    --------
    triggered_pulses : list
        List of triggered pulses
    adjusted_time_us : ndarray
        Adjusted time axis (μs)
    mean_pulse : ndarray
        Mean pulse waveform
    std_pulse : ndarray
        Standard deviation of pulse waveforms
    """
    # Load data
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("loading success")
    signal_data = np.squeeze(mat_data["TDX1"])
    Tinterval = float(mat_data['Tinterval'].item())
    Fs = 1.0 / Tinterval
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Extract data from specified time range
    start_idx = int(start_time * Fs)
    duration_samples = int(duration * Fs)
    chunk = signal_data[start_idx:start_idx + duration_samples]
    
    # Transfer to GPU
    chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.float32)
    
    # Detect positions where amplitude exceeds threshold (GPU version)
    window_samples = int(window_width * Fs)
    
    # Detect positions exceeding threshold (parallel processing on GPU)
    threshold_mask = torch.abs(chunk_tensor) >= amplitude_threshold
    potential_triggers = torch.where(threshold_mask)[0].cpu().numpy()
    
    # Select trigger points (avoid duplicates)
    trigger_points = []
    last_trigger = -window_samples
    
    for trigger in potential_triggers:
        if trigger > last_trigger + window_samples:
            trigger_points.append(trigger)
            last_trigger = trigger
    
    # Extract waveforms from each trigger point
    # Initialize lists to store triggered pulses
    triggered_pulses = []  # Store raw pulse data
    
    # Prepare batch for GPU processing
    valid_triggers = [t for t in trigger_points if t + window_samples <= len(chunk)]
    n_pulses = len(valid_triggers)
    
    if n_pulses == 0:
        print("No valid pulses found")
        return [], np.array([]), np.array([]), np.array([])
    
    # Create tensor to store all pulses at once in GPU memory
    all_pulses_tensor = torch.zeros((n_pulses, window_samples), device=device)
    
    # Extract pulse from each trigger point
    for i, trigger in enumerate(valid_triggers):
        all_pulses_tensor[i] = chunk_tensor[trigger:trigger + window_samples]
        # Keep CPU list as well
        triggered_pulses.append(chunk[trigger:trigger + window_samples])
    
    # Remove initial reflection (set 0-55μs signal to 0)
    neglegible_time = 55e-6  # meaningless time
    zero_samples = int(neglegible_time * Fs)  # Calculate samples for 55μs
    
    processed_pulses_tensor = all_pulses_tensor.clone()
    processed_pulses_tensor[:, :zero_samples] = 0  # Set initial part to 0
    
    # Execute Hilbert transform on GPU (Since PyTorch doesn't have direct Hilbert transform,
    # transfer back to CPU and process with SciPy)
    processed_pulses_np = processed_pulses_tensor.cpu().numpy()
    
    # Apply Hilbert transform with batch processing
    
    hilbert_matrix = np.zeros_like(processed_pulses_np)
    
    for i in range(n_pulses):
        analytic_signal = hilbert(processed_pulses_np[i])
        hilbert_matrix[i] = np.abs(analytic_signal)
    
    # Transfer results back to GPU if needed
    hilbert_tensor = torch.tensor(hilbert_matrix, device=device)
    
    print(f"Found {n_pulses} triggered pulses")
    
    # Generate time axis (in μs)
    pulse_time_us = np.arange(window_samples) * Tinterval * 1e6
    
    # Time axis adjusted for zero point
    adjusted_time_us = np.arange(-zero_samples, window_samples-zero_samples) * Tinterval * 1e6
    
    print(f"Hilbert transform matrix shape: {hilbert_matrix.shape} (number of pulses x number of samples)")
    
    # Create Hilbert transform matrix excluding time from -60 to 0
    hilbert_matrix_trimmed = hilbert_matrix[:, zero_samples:]
    adjusted_time_us_trimmed = adjusted_time_us[adjusted_time_us >= 0]
    
    # エラー回避のためにサイズチェックを追加
    if len(adjusted_time_us_trimmed) > 0:
        # # Plot the entire matrix
        # plt.figure(figsize=(12, 6))
        # plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis', 
        #         extent=[0, adjusted_time_us_trimmed[-1], n_pulses-0.5, -0.5])
        # plt.colorbar(label='Amplitude')
        # plt.xlabel('Time (μs)')
        # plt.ylabel('Pulse Number')
        # plt.title(f'Hilbert Transform Matrix ({n_pulses} pulses x {hilbert_matrix_trimmed.shape[1]} samples)')
        # plt.tight_layout()
        
        # 画像を保存する
        if output_dir is not None:
            # 出力ディレクトリが存在しない場合は作成
            os.makedirs(output_dir, exist_ok=True)
            
            # プロットを作成（エラー回避のため、adjusted_time_us_trimmedが空でないことを確認）
            plt.figure(figsize=(12, 6))
            if len(adjusted_time_us_trimmed) > 0:
                plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis', 
                        extent=[0, adjusted_time_us_trimmed[-1], n_pulses-0.5, -0.5],
                        vmin=0, vmax=1)
            else:
                # 時間軸が空の場合は、デフォルトの範囲でプロット
                plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis',
                        vmin=0, vmax=1)
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time (μs)')
            plt.ylabel('Pulse Number')
            
            # 入力ファイル名から出力ファイル名を生成
            base_filename = os.path.basename(file_path)
            base_name = os.path.splitext(base_filename)[0]
            plt.title(f'Echo map ({base_name})')
            plt.tight_layout()
            
            # パルス数を表示
            print(f"パルス数: {n_pulses}")
            
            output_filename = f"{base_name}_tdx1.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 画像を保存
            plt.savefig(output_path)
            print(f"画像を保存しました: {output_path}")
            
            # 信号波形データを.npy形式で保存
            signal_data_filename = f"{base_name}_tdx1.npy"
            signal_data_path = os.path.join(output_dir, signal_data_filename)
            
            # 保存するデータを準備（ヒルベルト変換後の信号波形）
            signal_data_to_save = {
                'hilbert_matrix': hilbert_matrix_trimmed,
                'time_axis': adjusted_time_us_trimmed,
                'n_pulses': n_pulses
            }
            
            # .npy形式で保存
            np.save(signal_data_path, signal_data_to_save)
            print(f"信号波形データを保存しました: {signal_data_path}")
        
        plt.show()
    else:
        print("警告: 調整された時間軸が空です。プロットをスキップします。")
    
    # Calculate average waveform (executed on GPU)
    mean_pulse = torch.mean(all_pulses_tensor, dim=0).cpu().numpy()
    std_pulse = torch.std(all_pulses_tensor, dim=0).cpu().numpy()
    
    # Average waveform of Hilbert transform
    mean_hilbert = torch.mean(hilbert_tensor, dim=0).cpu().numpy()
    std_hilbert = torch.std(hilbert_tensor, dim=0).cpu().numpy()
    
    plt.figure(figsize=(15, 12))
    
    return triggered_pulses, adjusted_time_us, mean_pulse, std_pulse

def generate_bin(file_path, channels=["TDX1"], start_time=0.444, duration=0.001,
                window_width=100e-6, amplitude_threshold=2.0, output_dir=None):
    """
    開発者用ツール：.matファイルから指定されたチャンネルごとにトリガーされたパルスを抽出し、
    バイナリ形式で保存します。
    
    Parameters:
    -----------
    file_path : str
        .matファイルへのパス
    channels : list
        処理するチャンネルのリスト (例: ["TDX1", "TDX2"])
    start_time : float
        開始時間 (秒)
    duration : float
        分析時間幅 (秒)
    window_width : float
        抽出ウィンドウの幅 (秒)、デフォルト100μs
    amplitude_threshold : float
        トリガーのための振幅閾値
    output_dir : str
        出力ディレクトリ、デフォルトはNone（保存なし）
    
    Returns:
    --------
    results : dict
        チャンネルごとの結果を含む辞書
        {channel_name: {'triggered_pulses': list, 'time_axis': ndarray, 
                       'mean_pulse': ndarray, 'std_pulse': ndarray}}
    """
    # Initialize dictionary to store results
    results = {}
    
    # Load data
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("Loading successful")
    
    # Get sampling interval
    Tinterval = float(mat_data['Tinterval'].item())
    Fs = 1.0 / Tinterval
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each channel
    for channel in channels:
        print(f"Processing channel {channel}...")
        
        # Check if channel exists in data
        if channel not in mat_data:
            print(f"Warning: Channel {channel} not found in file. Skipping.")
            continue
        
        # Get signal data
        signal_data = np.squeeze(mat_data[channel])
        
        # Extract data from specified time range
        start_idx = int(start_time * Fs)
        duration_samples = int(duration * Fs)
        chunk = signal_data[start_idx:start_idx + duration_samples]
        
        # Transfer data to GPU
        chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.float32)
        
        # Calculate window samples
        window_samples = int(window_width * Fs)
        
        # Detect positions exceeding threshold (parallel processing on GPU)
        threshold_mask = torch.abs(chunk_tensor) >= amplitude_threshold
        potential_triggers = torch.where(threshold_mask)[0].cpu().numpy()
        
        # Select trigger points (avoid overlap)
        trigger_points = []
        last_trigger = -window_samples
        
        for trigger in potential_triggers:
            if trigger > last_trigger + window_samples:
                trigger_points.append(trigger)
                last_trigger = trigger
        
        # Extract waveforms from each trigger point
        triggered_pulses = []  # Store raw waveform data
        
        # Prepare batch for GPU processing
        valid_triggers = [t for t in trigger_points if t + window_samples <= len(chunk)]
        n_pulses = len(valid_triggers)
        
        if n_pulses == 0:
            print(f"No valid pulses found for channel {channel}")
            results[channel] = {
                'triggered_pulses': [],
                'time_axis': np.array([]),
                'mean_pulse': np.array([]),
                'std_pulse': np.array([])
            }
            continue
        
        # Create tensor to store all pulses in GPU memory at once
        all_pulses_tensor = torch.zeros((n_pulses, window_samples), device=device)
        
        # Extract pulses from each trigger point
        for i, trigger in enumerate(valid_triggers):
            all_pulses_tensor[i] = chunk_tensor[trigger:trigger + window_samples]
            triggered_pulses.append(chunk[trigger:trigger + window_samples])
        
        # Remove initial reflection (set 0-55μs signal to zero)
        neglegible_time = 55e-6  # Time to ignore
        zero_samples = int(neglegible_time * Fs)  # Calculate samples for 55μs
        
        processed_pulses_tensor = all_pulses_tensor.clone()
        processed_pulses_tensor[:, :zero_samples] = 0  # Set initial part to zero
        
        # Hilbert transform on GPU (Since PyTorch doesn't have direct Hilbert transform,
        # process on CPU using SciPy)
        processed_pulses_np = processed_pulses_tensor.cpu().numpy()
        
        # Apply Hilbert transform in batch
        from scipy.signal import hilbert
        hilbert_matrix = np.zeros_like(processed_pulses_np)
        
        for i in range(n_pulses):
            analytic_signal = hilbert(processed_pulses_np[i])
            hilbert_matrix[i] = np.abs(analytic_signal)
        
        # Return results to GPU if needed
        hilbert_tensor = torch.tensor(hilbert_matrix, device=device)
        
        print(f"Channel {channel}: Detected {n_pulses} trigger pulses")
        
        # Generate time axis (in μs)
        pulse_time_us = np.arange(window_samples) * Tinterval * 1e6
        
        # Time axis adjusted for zero point
        adjusted_time_us = np.arange(-zero_samples, window_samples-zero_samples) * Tinterval * 1e6
        
        # Create Hilbert transform matrix part after zero time
        hilbert_matrix_trimmed = hilbert_matrix[:, zero_samples:]
        adjusted_time_us_trimmed = adjusted_time_us[adjusted_time_us >= 0]
        
        # Calculate average waveform (execute on GPU)
        mean_pulse = torch.mean(all_pulses_tensor, dim=0).cpu().numpy()
        std_pulse = torch.std(all_pulses_tensor, dim=0).cpu().numpy()
        
        # Average waveform of Hilbert transform
        mean_hilbert = torch.mean(hilbert_tensor, dim=0).cpu().numpy()
        std_hilbert = torch.std(hilbert_tensor, dim=0).cpu().numpy()
        
        # Store results in dictionary
        results[channel] = {
            'triggered_pulses': triggered_pulses,
            'time_axis': adjusted_time_us,
            'mean_pulse': mean_pulse,
            'std_pulse': std_pulse,
            'hilbert_matrix': hilbert_matrix_trimmed,
            'time_axis_trimmed': adjusted_time_us_trimmed,
            'n_pulses': n_pulses
        }
        
        # Save data if output directory is specified
        if output_dir is not None:
            # Get base name from input file
            base_filename = os.path.basename(file_path)
            base_name = os.path.splitext(base_filename)[0]
            
            # Save binary data
            output_filename = f"{base_name}_{channel}.npy"
            output_path = os.path.join(output_dir, output_filename)
            
            # Prepare data to save
            data_to_save = {
                'raw_pulses': triggered_pulses,
                'hilbert_matrix': hilbert_matrix_trimmed,
                'time_axis': adjusted_time_us_trimmed,
                'n_pulses': n_pulses,
                'mean_pulse': mean_pulse,
                'std_pulse': std_pulse,
                'mean_hilbert': mean_hilbert,
                'std_hilbert': std_hilbert
            }
            
            # Save in binary format
            np.save(output_path, data_to_save)
            print(f"Saved data for channel {channel}: {output_path}")
            print(f"Data shape: {hilbert_matrix_trimmed.shape}")
            
            # Generate and save plot
            if len(adjusted_time_us_trimmed) > 0:
                plt.figure(figsize=(12, 6))
                plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis', 
                        extent=[0, adjusted_time_us_trimmed[-1], n_pulses-0.5, -0.5],
                        vmin=0, vmax=1)
                plt.colorbar(label='Amplitude')
                plt.xlabel('Time (μs)')
                plt.ylabel('Pulse Number')
                plt.title(f'Echo Map ({base_name} - {channel})')
                plt.tight_layout()
                
                # Save image
                img_filename = f"{base_name}_{channel}.png"
                img_path = os.path.join(output_dir, img_filename)
                plt.savefig(img_path)
                print(f"Saved image: {img_path}")
                plt.close()
    
    return results



def generate_bin_multi(file_path, channels, start_time=0.444, duration=0.001,
                starting_window=50e-6, ending_window=100e-6, amplitude_threshold=2.0, output_dir=None):
    """
    開発者用ツール：.matファイルから指定されたチャンネルごとにトリガーされたパルスを抽出し、
    マルチチャンネルのテンソルとして保存します。
    
    Parameters:
    -----------
    file_path : str
        .matファイルへのパス
    channels : list
        処理するチャンネルのリスト (例: ["TDX1", "TDX2"])
    start_time : float
        開始時間 (秒)
    duration : float
        分析時間幅 (秒)
    starting_window : float
        トリガーポイントからの開始ウィンドウ時間 (秒)、デフォルト-50μs
    ending_window : float
        トリガーポイントからの終了ウィンドウ時間 (秒)、デフォルト100μs
    amplitude_threshold : float
        トリガーのための振幅閾値
    output_dir : str
        出力ディレクトリ、デフォルトはNone（保存なし）
    
    Returns:
    --------
    results : dict
        チャンネルごとの結果を含む辞書
    """
    # Initialize dictionary to store results
    results = {}
    
    # Load data
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("Loading successful")
    
    # Get sampling interval and other metadata
    Tinterval = float(mat_data['Tinterval'].item())
    Tstart = float(mat_data['Tstart'].item())
    try:
        version = float(mat_data['Version'].item())
    except (ValueError, AttributeError):
        version = 1.0
        print("Warning: Version information not found or invalid. Using default version 1.0")
    Fs = 1.0 / Tinterval
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # トリガーチャンネル（最初のチャンネル）からトリガーポイントを検出
    trigger_channel = channels[0]
    print(f"Detecting triggers from channel {trigger_channel}...")
    
    if trigger_channel not in mat_data:
        raise ValueError(f"Trigger channel {trigger_channel} not found in file")
    
    # トリガーチャンネルのデータを取得
    trigger_signal = np.squeeze(mat_data[trigger_channel])
    
    # 指定された時間範囲のデータを抽出
    start_idx = int(start_time * Fs)
    duration_samples = int(duration * Fs)
    trigger_chunk = trigger_signal[start_idx:start_idx + duration_samples]
    
    # GPUにデータを転送
    trigger_chunk_tensor = torch.tensor(trigger_chunk, device=device, dtype=torch.float32)
    
    # 閾値を超える位置を検出（GPUで並列処理）
    threshold_mask = torch.abs(trigger_chunk_tensor) >= amplitude_threshold
    potential_triggers = torch.where(threshold_mask)[0].cpu().numpy()
    
    # トリガーポイントを選択（オーバーラップを避ける）
    trigger_points = []
    last_trigger = -int(ending_window * Fs)  # ウィンドウ幅に基づいて最小間隔を設定
    
    for trigger in potential_triggers:
        if trigger > last_trigger + int(ending_window * Fs):
            trigger_points.append(trigger)
            last_trigger = trigger
    
    trigger_times = np.array(trigger_points) * Tinterval + start_time
    print(f"Trigger times shape: {trigger_times.shape}")
    print(f"First few trigger times: {trigger_times[:5]}")
    
    # 有効なトリガーポイントを準備
    valid_triggers = [t for t in trigger_points if t + int(ending_window * Fs) <= len(trigger_chunk) and t + int(starting_window * Fs) >= 0]
    n_pulses = len(valid_triggers)
    
    if n_pulses == 0:
        raise ValueError(f"No valid pulses found in trigger channel {trigger_channel}")
    
    # 全チャンネルのデータを処理
    for channel in channels:
        print(f"Processing channel {channel}...")
        
        if channel not in mat_data:
            print(f"Warning: Channel {channel} not found in file. Skipping.")
            continue
        
        # チャンネルのデータを取得
        signal_data = np.squeeze(mat_data[channel])
        chunk = signal_data[start_idx:start_idx + duration_samples]
        chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.float32)
        
        # トリガーポイントからパルスを抽出
        window_samples = int((ending_window - starting_window) * Fs)
        all_pulses_tensor = torch.zeros((n_pulses, window_samples), device=device)
        for i, trigger in enumerate(valid_triggers):
            all_pulses_tensor[i] = chunk_tensor[trigger + int(starting_window * Fs):trigger + int(ending_window * Fs)]
        
        # 結果を辞書に格納
        results[channel] = {
            'pulses': all_pulses_tensor.cpu().numpy(),
            'trigger_times': trigger_times,
            'n_pulses': n_pulses
        }
    
    # 出力ディレクトリが指定されている場合はデータを保存
    if output_dir is not None and results:
        base_filename = os.path.basename(file_path)
        base_name = os.path.splitext(base_filename)[0]
        
        # マルチチャンネルテンソルを作成
        n_channels = len(channels)
        window_samples = int((ending_window - starting_window) * Fs)
        multi_channel_tensor = np.zeros((n_channels, n_pulses, window_samples))
        
        # 各チャンネルのデータでテンソルを埋める
        for i, channel in enumerate(channels):
            if channel in results:
                multi_channel_tensor[i] = results[channel]['pulses']
        
        # メタデータを辞書として保存
        metadata = {
            'Tstart': Tstart,
            'Tinterval': Tinterval,
            'version': version,
            'channels': channels,
            'n_pulses': n_pulses,
            'starting_window': starting_window,
            'ending_window': ending_window,
            'start_time': start_time,
            'duration': duration,
            'amplitude_threshold': amplitude_threshold
        }
        
        # Save tensor and metadata
        output_filename = f"{base_name}_multi_channel.npz"
        output_path = os.path.join(output_dir, output_filename)
        np.savez(output_path, 
                 signal_data=multi_channel_tensor,
                 metadata=metadata)
        print(f"Saved multi-channel data: {output_path}")
        print(f"Signal tensor shape: {multi_channel_tensor.shape}")
        print(f"Metadata: {metadata}")
    
    return results