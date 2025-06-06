import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from scipy import signal
from src import plot_signal_waveform
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src
from scipy.signal import hilbert
def extract_amplitude_triggered_pulses(file_path, start_time=0.444, duration=0.001, 
                                    window_width=50e-6, amplitude_threshold=2.0):
    """
    振幅が閾値に達した瞬間から一定時間の波形を切り出す関数
    
    Parameters:
    -----------
    file_path : str
        .matファイルのパス
    start_time : float
        開始時間（秒）
    duration : float
        解析時間幅（秒）
    window_width : float
        切り出す窓幅（秒）、デフォルト50μs
    amplitude_threshold : float
        トリガーとなる振幅閾値
    
    Returns:
    --------
    triggered_pulses : list
        トリガーされたパルスのリスト
    adjusted_time_us : ndarray
        調整された時間軸（μs）
    mean_pulse : ndarray
        平均パルス波形
    std_pulse : ndarray
        パルス波形の標準偏差
    """
    # データ読み込み
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("loading success")
    signal_data = np.squeeze(mat_data["TDX1"])
    Tinterval = float(mat_data['Tinterval'].item())
    Fs = 1.0 / Tinterval
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 指定時間範囲のデータ切り出し
    start_idx = int(start_time * Fs)
    duration_samples = int(duration * Fs)
    chunk = signal_data[start_idx:start_idx + duration_samples]
    
    # 振幅閾値を超えた位置の検出
    window_samples = int(window_width * Fs)
    trigger_points = []
    i = 0
    
    while i < len(chunk) - window_samples:
        if abs(chunk[i]) >= amplitude_threshold:
            trigger_points.append(i)
            i += window_samples  # 次のトリガー検索は window_width 秒後から
        else:
            i += 1
    
    # 各トリガーポイントからの波形切り出し
    # トリガーされたパルスを格納するリストを初期化
    triggered_pulses = []  # 生のパルスデータを格納
    triggered_pulses_processed = []  # 処理済みパルスデータを格納
    triggered_pulses_hilbert = []  # 処理済みパルスデータを格納
    # 各トリガーポイントに対してパルスを切り出し
    for trigger in trigger_points:
        # トリガーポイントからwindow_samples秒分のパルスを切り出し
        if trigger + window_samples <= len(chunk):
            # パルスの切り出し
            pulse = chunk[trigger:trigger + window_samples]
            triggered_pulses.append(pulse)
            
            # 初期の反射波を除去（0-3μsまでの信号を0に設定）
            neglegible_time = 55e-6 # meaningless time
            processed_pulse = pulse.copy()  # パルスのコピー作成
            zero_samples = int(neglegible_time * Fs)  # 3μs分のサンプル数計算
            processed_pulse[:zero_samples] = 0  # 初期部分を0に設定
            triggered_pulses_processed.append(processed_pulse)
            
            # ヒルベルト変換を適用して振幅情報を抽出
            # scipyのsignalモジュールからhilbert関数をインポートする必要があります
            from scipy.signal import hilbert
            analytic_signal = hilbert(processed_pulse)
            amplitude_envelope = np.abs(analytic_signal)
            
            # ヒルベルト変換後の振幅情報を格納
            if 'triggered_pulses_hilbert' not in locals():
                triggered_pulses_hilbert = []
            triggered_pulses_hilbert.append(amplitude_envelope)
    
    # 検出されたパルス数を記録
    n_pulses = len(triggered_pulses)
    print(f"Found {n_pulses} triggered pulses")
    
    # 時間軸の生成（μs単位）
    pulse_time_us = np.arange(window_samples) * Tinterval * 1e6  # 時間軸をμs単位で作成
    
    # negligible_timeが経過した時点を0とした時間軸
    zero_samples = int(neglegible_time * Fs)
    adjusted_time_us = np.arange(-zero_samples, window_samples-zero_samples) * Tinterval * 1e6
    
    # Convert Hilbert transform data to matrix format
    hilbert_matrix = np.array(triggered_pulses_hilbert)
    print(f"Hilbert transform matrix shape: {hilbert_matrix.shape} (pulses × samples)")
    
    # Plot Hilbert transform matrix as colormap
    plt.figure(figsize=(12, 6))
    im = plt.imshow(hilbert_matrix, aspect='auto', cmap='viridis', 
                   extent=[adjusted_time_us[0], adjusted_time_us[-1], n_pulses-0.5, -0.5])
    plt.colorbar(im, label='Amplitude')
    plt.xlabel('Time (μs)')
    plt.ylabel('Pulse Number')
    plt.title('Hilbert Transform Matrix Amplitude Map')
    plt.tight_layout()
    
    # 処理済みパルスを大きくプロットする
    plt.figure(figsize=(12, 6))
    
    for i, processed_pulse in enumerate(triggered_pulses_processed):
        plt.plot(adjusted_time_us, processed_pulse, alpha=0.7, label=f'Processed Pulse {i+1}' if i < 5 else '')
    
    plt.axhline(y=amplitude_threshold, color='r', linestyle='--', alpha=0.5, label='Trigger Level')
    plt.axhline(y=-amplitude_threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time (After Negligible Time)')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Processed Pulses (Initial Reflections Removed)')
    plt.grid(True)
    if n_pulses <= 5:
        plt.legend()
    plt.tight_layout()
    
    # ヒルベルト変換後のエンベロープをプロットする
    plt.figure(figsize=(12, 6))
    for i, envelope in enumerate(triggered_pulses_hilbert):
        plt.plot(adjusted_time_us, envelope, alpha=0.7, label=f'Envelope {i+1}' if i < 5 else '')
    
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time (After Negligible Time)')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Hilbert Transform Envelopes')
    plt.grid(True)
    if n_pulses <= 5:
        plt.legend()
    plt.tight_layout()
    
    # ヒルベルト変換行列をヒートマップとして表示
    plt.figure(figsize=(12, 6))
    plt.imshow(hilbert_matrix, aspect='auto', cmap='viridis', 
               extent=[adjusted_time_us[0], adjusted_time_us[-1], n_pulses-0.5, -0.5])
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (μs)')
    plt.ylabel('Pulse Number')
    plt.title(f'Hilbert Transform Matrix ({n_pulses} pulses × {hilbert_matrix.shape[1]} samples)')
    plt.tight_layout()
    plt.show()
    
    # プロット
    plt.figure(figsize=(15, 12))
    
    # 1. すべてのパルスを重ねて表示
    plt.subplot(2, 2, 1)
    for i, pulse in enumerate(triggered_pulses):
        plt.plot(adjusted_time_us, pulse, alpha=0.5, label=f'Pulse {i+1}' if i < 5 else '')
    plt.axhline(y=amplitude_threshold, color='r', linestyle='--', alpha=0.5, label='Trigger Level')
    plt.axhline(y=-amplitude_threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Overlaid Triggered Pulses')
    plt.grid(True)
    if n_pulses <= 5:
        plt.legend()
    
    # 2. 平均波形の表示
    mean_pulse = np.mean(triggered_pulses, axis=0)
    std_pulse = np.std(triggered_pulses, axis=0)
    
    plt.subplot(2, 2, 2)
    plt.plot(adjusted_time_us, mean_pulse, 'b-', label='Mean')
    plt.fill_between(adjusted_time_us, 
                    mean_pulse - std_pulse, 
                    mean_pulse + std_pulse, 
                    color='b', alpha=0.2, label='±1 STD')
    plt.axhline(y=amplitude_threshold, color='r', linestyle='--', alpha=0.5, label='Trigger Level')
    plt.axhline(y=-amplitude_threshold, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Average Pulse Shape with Standard Deviation')
    plt.grid(True)
    plt.legend()
    
    # 3. ヒルベルト変換の平均波形
    mean_hilbert = np.mean(triggered_pulses_hilbert, axis=0)
    std_hilbert = np.std(triggered_pulses_hilbert, axis=0)
    
    plt.subplot(2, 2, 3)
    plt.plot(adjusted_time_us, mean_hilbert, 'r-', label='Mean Envelope')
    plt.fill_between(adjusted_time_us, 
                    mean_hilbert - std_hilbert, 
                    mean_hilbert + std_hilbert, 
                    color='r', alpha=0.2, label='±1 STD')
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Average Hilbert Envelope with Standard Deviation')
    plt.grid(True)
    plt.legend()
    
    # 4. 生波形と包絡線の比較
    plt.subplot(2, 2, 4)
    plt.plot(adjusted_time_us, mean_pulse, 'b-', label='Mean Signal')
    plt.plot(adjusted_time_us, mean_hilbert, 'r-', label='Mean Envelope')
    plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Signal and Envelope')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 個別パルスの表示（最大3つまで）
    n_display = min(3, n_pulses)
    if n_display > 0:
        plt.figure(figsize=(15, 6))
        for i in range(n_display):
            plt.subplot(1, 3, i+1)
            plt.plot(adjusted_time_us, triggered_pulses[i], 'b-', label='Original')
            plt.plot(adjusted_time_us, triggered_pulses_hilbert[i], 'r-', label='Envelope')
            plt.axhline(y=amplitude_threshold, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=-amplitude_threshold, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='g', linestyle='-', alpha=0.5, label='Zero Time')
            plt.xlabel('Time (μs)')
            plt.ylabel('Amplitude')
            plt.title(f'Pulse {i+1}')
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    return triggered_pulses, adjusted_time_us, mean_pulse, std_pulse

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

def extract_waveforms_from_trigger_times(file_path, channels, trigger_times, 
                                       starting_window=-50e-6, ending_window=250e-6, 
                                       output_dir=None):
    """
    トリガー時刻を起点として、複数チャンネルから指定時間幅の波形を切り出し、縦に並べる処理
    
    Parameters:
    -----------
    file_path : str
        .matファイルへのパス
    channels : list
        処理するチャンネルのリスト (例: ["TDX1", "TDX2", "TDX3", "enlarged"])
    trigger_times : list or np.ndarray
        トリガー時刻のリスト（秒単位）
    starting_window : float
        トリガーポイントからの開始ウィンドウ時間 (秒)、デフォルト-50μs
    ending_window : float
        トリガーポイントからの終了ウィンドウ時間 (秒)、デフォルト250μs（合計300μs）
    output_dir : str
        出力ディレクトリ、デフォルトはNone（保存なし）
    
    Returns:
    --------
    results : dict
        チャンネルごとの結果を含む辞書
        {channel_name: {'pulses': ndarray, 'time_axis': ndarray, 'n_pulses': int}}
    """
    import numpy as np
    import scipy.io as sio
    import torch
    import matplotlib.pyplot as plt
    import os
    
    # データを読み込み
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("Loading successful")
    
    # サンプリング間隔とメタデータを取得
    Tinterval = float(mat_data['Tinterval'].item())
    Fs = 1.0 / Tinterval
    
    # GPU/CPUデバイスを確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリを作成（指定されている場合）
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # トリガー時刻をサンプル番号に変換
    trigger_samples = np.array(trigger_times) * Fs
    trigger_samples = trigger_samples.astype(int)
    
    # ウィンドウサンプル数を計算
    window_samples = int((ending_window - starting_window) * Fs)
    starting_offset = int(starting_window * Fs)
    ending_offset = int(ending_window * Fs)
    
    n_pulses = len(trigger_samples)
    print(f"Number of pulses: {n_pulses}")
    print(f"Window width: {(ending_window - starting_window) * 1e6:.1f} μs")
    
    results = {}
    
    # 各チャンネルを処理
    for channel in channels:
        print(f"Processing channel {channel}...")
        
        if channel not in mat_data:
            print(f"Warning: Channel {channel} not found in file. Skipping.")
            continue
        
        # チャンネルのデータを取得
        signal_data = np.squeeze(mat_data[channel])
        signal_tensor = torch.tensor(signal_data, device=device, dtype=torch.float32)
        
        # 有効なトリガーポイントをフィルタリング
        valid_triggers = []
        for trigger_sample in trigger_samples:
            start_sample = trigger_sample + starting_offset
            end_sample = trigger_sample + ending_offset
            if start_sample >= 0 and end_sample < len(signal_data):
                valid_triggers.append(trigger_sample)
        
        n_valid_pulses = len(valid_triggers)
        if n_valid_pulses == 0:
            print(f"No valid pulses found for channel {channel}")
            continue
        
        # 全パルスのテンソルを作成
        all_pulses_tensor = torch.zeros((n_valid_pulses, window_samples), device=device)
        
        # 各トリガーポイントからパルスを抽出
        for i, trigger_sample in enumerate(valid_triggers):
            start_sample = trigger_sample + starting_offset
            end_sample = trigger_sample + ending_offset
            all_pulses_tensor[i] = signal_tensor[start_sample:end_sample]
        
        # CPUに転送して結果に格納
        pulses_array = all_pulses_tensor.cpu().numpy()
        
        # 時間軸を生成（μs単位）
        time_axis_us = np.arange(window_samples) * Tinterval * 1e6 + starting_window * 1e6
        
        results[channel] = {
            'pulses': pulses_array,
            'time_axis': time_axis_us,
            'n_pulses': n_valid_pulses,
            'trigger_times': np.array(valid_triggers) / Fs
        }
        
        # データを保存（指定されている場合）
        if output_dir is not None:
            base_filename = os.path.basename(file_path)
            base_name = os.path.splitext(base_filename)[0]
            
            # バイナリデータを保存
            output_filename = f"{base_name}_{channel}_triggered.npy"
            output_path = os.path.join(output_dir, output_filename)
            
            data_to_save = {
                'pulses': pulses_array,
                'time_axis': time_axis_us,
                'n_pulses': n_valid_pulses,
                'trigger_times': results[channel]['trigger_times']
            }
            
            np.save(output_path, data_to_save)
            print(f"Saved data for channel {channel}: {output_path}")
            
            # 可視化
            plt.figure(figsize=(12, 8))
            plt.imshow(pulses_array, aspect='auto', cmap='viridis', 
                      extent=[time_axis_us[0], time_axis_us[-1], n_valid_pulses-0.5, -0.5])
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time (μs)')
            plt.ylabel('Pulse Number')
            plt.title(f'Triggered Waveforms ({base_name} - {channel})')
            plt.tight_layout()
            
            # 画像を保存
            img_filename = f"{base_name}_{channel}_triggered.png"
            img_path = os.path.join(output_dir, img_filename)
            plt.savefig(img_path)
            print(f"Saved image: {img_path}")
            plt.close()
    
    return results

def extract_waveforms_with_hilbert(file_path, channels, trigger_times, 
                                 starting_window=-50e-6, ending_window=250e-6, 
                                 neglect_time=55e-6, output_dir=None, savebin=False,
                                 flow_velocity=6.0):
    """
    トリガー時刻を起点として、複数チャンネルから指定時間幅の波形を切り出し、
    ヒルベルト変換も適用して縦に並べる処理
    
    Parameters:
    -----------
    file_path : str
        .matファイルへのパス
    channels : list
        処理するチャンネルのリスト (例: ["TDX1", "TDX2", "TDX3", "enlarged"])
    trigger_times : list or np.ndarray
        トリガー時刻のリスト（秒単位）
    starting_window : float
        トリガーポイントからの開始ウィンドウ時間 (秒)、デフォルト-50μs
    ending_window : float
        トリガーポイントからの終了ウィンドウ時間 (秒)、デフォルト250μs（合計300μs）
    neglect_time : float
        初期反射を除去する時間 (秒)、デフォルト55μs
    output_dir : str
        出力ディレクトリ、デフォルトはNone（保存なし）
    flow_velocity : float
        流速 (m/s)、デフォルト10.0 m/s
    
    Returns:
    --------
    results : dict
        チャンネルごとの結果を含む辞書
        {channel_name: {
            'raw_pulses': ndarray, 
            'hilbert_pulses': ndarray, 
            'hilbert_trimmed': ndarray,
            'time_axis': ndarray, 
            'time_axis_trimmed': ndarray,
            'n_pulses': int
        }}
    """
    
    # データを読み込み
    print("Loading data...")
    mat_data = sio.loadmat(file_path)
    print("Loading successful")
    
    # サンプリング間隔とメタデータを取得
    Tinterval = float(mat_data['Tinterval'].item())
    Fs = 1.0 / Tinterval
    
    # パルス繰り返し周波数（3kHz）
    PRF = 3000  # Hz
    
    # GPU/CPUデバイスを確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリを作成（指定されている場合）
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # トリガー時刻をサンプル番号に変換
    trigger_samples = np.array(trigger_times) * Fs
    trigger_samples = trigger_samples.astype(int)
    
    # ウィンドウサンプル数を計算
    window_samples = int((ending_window - starting_window) * Fs)
    starting_offset = int(starting_window * Fs)
    ending_offset = int(ending_window * Fs)
    neglect_samples = int(neglect_time * Fs)
    
    n_pulses = len(trigger_samples)
    print(f"Number of pulses: {n_pulses}")
    print(f"Window width: {(ending_window - starting_window) * 1e6:.1f} μs")
    print(f"Neglect time: {neglect_time * 1e6:.1f} μs")
    
    # 流速から距離への変換係数を計算（mm/パルス）
    distance_per_pulse = (flow_velocity * 1000) / PRF  # mm/pulse
    
    results = {}
    
    # 各チャンネルを処理
    for channel in channels:
        print(f"Processing channel {channel}...")
        
        if channel not in mat_data:
            print(f"Warning: Channel {channel} not found in file. Skipping.")
            continue
        
        # チャンネルのデータを取得
        signal_data = np.squeeze(mat_data[channel])
        signal_tensor = torch.tensor(signal_data, device=device, dtype=torch.float32)
        
        # 有効なトリガーポイントをフィルタリング
        valid_triggers = []
        for trigger_sample in trigger_samples:
            start_sample = trigger_sample + starting_offset
            end_sample = trigger_sample + ending_offset
            if start_sample >= 0 and end_sample < len(signal_data):
                valid_triggers.append(trigger_sample)
        
        n_valid_pulses = len(valid_triggers)
        if n_valid_pulses == 0:
            print(f"No valid pulses found for channel {channel}")
            continue
        
        # 全パルスのテンソルを作成
        all_pulses_tensor = torch.zeros((n_valid_pulses, window_samples), device=device)
        
        # 各トリガーポイントからパルスを抽出
        for i, trigger_sample in enumerate(valid_triggers):
            start_sample = trigger_sample + starting_offset
            end_sample = trigger_sample + ending_offset
            all_pulses_tensor[i] = signal_tensor[start_sample:end_sample]
        
        # CPUに転送
        raw_pulses_array = all_pulses_tensor.cpu().numpy()
        
        # 初期反射を除去（0-55μs信号を0に設定）
        processed_pulses = raw_pulses_array.copy()
        processed_pulses[:, :neglect_samples] = 0
        
        # ヒルベルト変換を適用（バッチ処理）
        hilbert_matrix = np.zeros_like(processed_pulses)
        
        print(f"Applying Hilbert transform to {n_valid_pulses} pulses...")
        for i in range(n_valid_pulses):
            analytic_signal = hilbert(processed_pulses[i])
            hilbert_matrix[i] = np.abs(analytic_signal)
        
        # 時間軸を生成（μs単位）
        time_axis_us = np.arange(window_samples) * Tinterval * 1e6 + starting_window * 1e6
        
        # 調整された時間軸（負の時間を除去）
        adjusted_time_us = np.arange(-neglect_samples, window_samples-neglect_samples) * Tinterval * 1e6
        
        # ヒルベルト変換行列のトリミング版（時間 >= 0 の部分のみ）
        hilbert_matrix_trimmed = hilbert_matrix[:, neglect_samples:]
        adjusted_time_us_trimmed = adjusted_time_us[adjusted_time_us >= 0]
        
        # 結果を辞書に格納
        results[channel] = {
            'raw_pulses': raw_pulses_array,
            'hilbert_pulses': hilbert_matrix,
            'hilbert_trimmed': hilbert_matrix_trimmed,
            'time_axis': time_axis_us,
            'time_axis_trimmed': adjusted_time_us_trimmed,
            'n_pulses': n_valid_pulses,
            'trigger_times': np.array(valid_triggers) / Fs
        }
        
        print(f"Channel {channel}: Processed {n_valid_pulses} pulses")
        print(f"Hilbert matrix shape: {hilbert_matrix.shape}")
        print(f"Trimmed matrix shape: {hilbert_matrix_trimmed.shape}")
        
        # データを保存および可視化（指定されている場合）
        if output_dir is not None:
            base_filename = os.path.basename(file_path)
            base_name = os.path.splitext(base_filename)[0]
            
            # バイナリデータを保存
            if savebin:
                output_filename = f"{base_name}_{channel}_hilbert.npy"
                output_path = os.path.join(output_dir, output_filename)
                
                data_to_save = {
                    'raw_pulses': raw_pulses_array,
                    'hilbert_matrix': hilbert_matrix,
                    'hilbert_trimmed': hilbert_matrix_trimmed,
                    'time_axis': time_axis_us,
                    'time_axis_trimmed': adjusted_time_us_trimmed,
                    'n_pulses': n_valid_pulses,
                    'trigger_times': results[channel]['trigger_times']
                }
                
                np.save(output_path, data_to_save)
                print(f"Saved data for channel {channel}: {output_path}")
            
            # 生信号の可視化
            plt.figure(figsize=(15, 10))
            
            # 生信号のプロット
            plt.figure(figsize=(10, 8))
            # 音速（水中）は約1500 m/s
            sound_velocity = 1500  # m/s
            # 時間を距離に変換（mm単位）
            distance_axis = time_axis_us * 1e-6 * sound_velocity * 1000  # μs → s → m → mm
            # 流速に基づいて距離を計算（mm単位）
            pulse_interval = 1.0 / 3000  # 3kHzの場合、パルス間隔は1/3000秒
            distance_per_pulse = flow_velocity * pulse_interval * 1000  # mm単位に変換
            total_distance = n_valid_pulses * distance_per_pulse
            
            plt.figure(figsize=(20, 20))  # より大きな図のサイズ
            plt.imshow(raw_pulses_array, aspect='equal', cmap='seismic', 
                      extent=[distance_axis[0], distance_axis[-1], total_distance, 0])
            plt.colorbar(label='Amplitude')
            plt.xlabel('Distance (mm)', fontsize=14)
            plt.ylabel('Distance (mm)', fontsize=14)
            plt.title(f'Raw Waveforms ({base_name} - {channel})', fontsize=16)
            plt.tight_layout()
            
            # 生信号の画像を保存
            raw_img_filename = f"{base_name}_{channel}_raw_waveforms.png"
            raw_img_path = os.path.join(output_dir, raw_img_filename)
            plt.savefig(raw_img_path, dpi=1000, bbox_inches='tight')  # 解像度を600dpiに増加
            plt.close()
            
            # ヒルベルト変換のプロット
            if len(adjusted_time_us_trimmed) > 0:
                plt.figure(figsize=(20, 20))  # より大きな図のサイズ
                # トリミングされた時間軸を距離に変換
                distance_axis_trimmed = adjusted_time_us_trimmed * 1e-6 * sound_velocity * 1000
                
                plt.imshow(hilbert_matrix_trimmed, aspect='equal', cmap='viridis', 
                          extent=[0, distance_axis_trimmed[-1], total_distance, 0],
                          vmin=0, vmax=1)
                cbar = plt.colorbar(label='Hilbert Amplitude')
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_ylabel('Hilbert Amplitude', fontsize=14)
                plt.xlabel('Distance (mm)', fontsize=14)
                plt.ylabel('Distance (mm)', fontsize=14)
                plt.title(f'Hilbert Transform (Echo Map) - Flow Velocity: {flow_velocity} m/s ({base_name} - {channel})', fontsize=16)
                plt.tight_layout()
                
                # ヒルベルト変換の画像を保存
                hilbert_img_filename = f"{base_name}_{channel}_hilbert_echo_map.png"
                hilbert_img_path = os.path.join(output_dir, hilbert_img_filename)
                plt.savefig(hilbert_img_path, dpi=1000, bbox_inches='tight')  # 解像度を600dpiに増加
                plt.close()
            
            # 画像を保存
            img_filename = f"{base_name}_{channel}_hilbert_comparison.png"
            img_path = os.path.join(output_dir, img_filename)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison image: {img_path}")
            plt.close()
            
            # ヒルベルト変換のみの単独プロット
            plt.figure(figsize=(12, 8))
            if len(adjusted_time_us_trimmed) > 0:
                plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis', 
                          extent=[0, adjusted_time_us_trimmed[-1], n_valid_pulses-0.5, -0.5],
                          vmin=0, vmax=1)
                plt.colorbar(label='Hilbert Amplitude')
                plt.xlabel('Time (μs)')
                plt.ylabel('Pulse Number')
                plt.title(f'Echo Map - Hilbert Transform ({base_name} - {channel})')
                plt.tight_layout()
                
                # ヒルベルト単独画像を保存
                hilbert_img_filename = f"{base_name}_{channel}_hilbert_only.png"
                hilbert_img_path = os.path.join(output_dir, hilbert_img_filename)
                plt.savefig(hilbert_img_path, dpi=300, bbox_inches='tight')
                print(f"Saved Hilbert image: {hilbert_img_path}")
                plt.close()
    
    return results


def plot_average_waveforms(results, output_dir=None):
    """
    各チャンネルの平均波形をプロット
    """
    
    if not results:
        print("No results to plot")
        return
    
    n_channels = len(results)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 4*n_channels))
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for i, (channel, data) in enumerate(results.items()):
        # 生信号の平均
        mean_raw = np.mean(data['raw_pulses'], axis=0)
        std_raw = np.std(data['raw_pulses'], axis=0)
        
        axes[i, 0].plot(data['time_axis'], mean_raw, 'b-', linewidth=2, label='Mean')
        axes[i, 0].fill_between(data['time_axis'], mean_raw - std_raw, mean_raw + std_raw, 
                               alpha=0.3, color='blue', label='±1σ')
        axes[i, 0].set_xlabel('Time (μs)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'{channel} - Raw Signal Average')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # ヒルベルト変換の平均
        mean_hilbert = np.mean(data['hilbert_trimmed'], axis=0)
        std_hilbert = np.std(data['hilbert_trimmed'], axis=0)
        
        # プロットの範囲を0から1までに設定
        axes[i, 1].plot(data['time_axis_trimmed'], mean_hilbert, 'r-', linewidth=2, label='Mean')
        axes[i, 1].fill_between(data['time_axis_trimmed'], mean_hilbert - std_hilbert, 
                               mean_hilbert + std_hilbert, alpha=0.3, color='red', label='±1σ')
        axes[i, 1].set_xlabel('Time (μs)')
        axes[i, 1].set_ylabel('Hilbert Amplitude')
        axes[i, 1].set_title(f'{channel} - Hilbert Transform Average')
        #axes[i, 1].set_ylim(0, 1)  # y軸の範囲を0から1に設定
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        avg_img_path = os.path.join(output_dir, "average_waveforms.png")
        plt.savefig(avg_img_path, dpi=300, bbox_inches='tight')
        print(f"Saved average waveforms: {avg_img_path}")
    
    plt.show()

