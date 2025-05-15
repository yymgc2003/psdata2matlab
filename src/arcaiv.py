import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from scipy import signal
from src import plot_signal_waveform

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