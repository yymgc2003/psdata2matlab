import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os


def extract_waveforms_with_hilbert(file_path, channels, trigger_times, 
                                 starting_window=-50e-6, ending_window=250e-6, 
                                 neglect_time=55e-6, output_dir=None):
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
            
            # 上段：生信号
            plt.subplot(2, 1, 1)
            plt.imshow(raw_pulses_array, aspect='auto', cmap='seismic', 
                      extent=[time_axis_us[0], time_axis_us[-1], n_valid_pulses-0.5, -0.5])
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time (μs)')
            plt.ylabel('Pulse Number')
            plt.title(f'Raw Waveforms ({base_name} - {channel})')
            
            # 下段：ヒルベルト変換後（トリミング版）
            plt.subplot(2, 1, 2)
            if len(adjusted_time_us_trimmed) > 0:
                plt.imshow(hilbert_matrix_trimmed, aspect='auto', cmap='viridis', 
                          extent=[0, adjusted_time_us_trimmed[-1], n_valid_pulses-0.5, -0.5],
                          vmin=0, vmax=np.percentile(hilbert_matrix_trimmed, 95))
                plt.colorbar(label='Hilbert Amplitude')
                plt.xlabel('Time (μs)')
                plt.ylabel('Pulse Number')
                plt.title(f'Hilbert Transform (Echo Map) ({base_name} - {channel})')
            
            plt.tight_layout()
            
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
                          vmin=0, vmax=np.percentile(hilbert_matrix_trimmed, 95))
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
    
    Parameters:
    -----------
    results : dict
        extract_waveforms_with_hilbert()の戻り値
    output_dir : str
        出力ディレクトリ
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
        
        axes[i, 1].plot(data['time_axis_trimmed'], mean_hilbert, 'r-', linewidth=2, label='Mean')
        axes[i, 1].fill_between(data['time_axis_trimmed'], mean_hilbert - std_hilbert, 
                               mean_hilbert + std_hilbert, alpha=0.3, color='red', label='±1σ')
        axes[i, 1].set_xlabel('Time (μs)')
        axes[i, 1].set_ylabel('Hilbert Amplitude')
        axes[i, 1].set_title(f'{channel} - Hilbert Transform Average')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        avg_img_path = os.path.join(output_dir, "average_waveforms.png")
        plt.savefig(avg_img_path, dpi=300, bbox_inches='tight')
        print(f"Saved average waveforms: {avg_img_path}")
    
    plt.show()


def plot_single_pulse_comparison(results, pulse_idx=0, output_dir=None):
    """
    単一パルスの生信号とヒルベルト変換の比較プロット
    
    Parameters:
    -----------
    results : dict
        extract_waveforms_with_hilbert()の戻り値
    pulse_idx : int
        プロットするパルスのインデックス
    output_dir : str
        出力ディレクトリ
    """
    
    if not results:
        print("No results to plot")
        return
    
    n_channels = len(results)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 4*n_channels))
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for i, (channel, data) in enumerate(results.items()):
        if pulse_idx >= data['n_pulses']:
            print(f"Warning: Pulse index {pulse_idx} exceeds available pulses in {channel}")
            continue
        
        # 生信号
        raw_pulse = data['raw_pulses'][pulse_idx]
        axes[i, 0].plot(data['time_axis'], raw_pulse, 'b-', linewidth=2)
        axes[i, 0].set_xlabel('Time (μs)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'{channel} - Raw Signal (Pulse #{pulse_idx})')
        axes[i, 0].grid(True, alpha=0.3)
        
        # ヒルベルト変換
        hilbert_pulse = data['hilbert_trimmed'][pulse_idx]
        axes[i, 1].plot(data['time_axis_trimmed'], hilbert_pulse, 'r-', linewidth=2)
        axes[i, 1].set_xlabel('Time (μs)')
        axes[i, 1].set_ylabel('Hilbert Amplitude')
        axes[i, 1].set_title(f'{channel} - Hilbert Transform (Pulse #{pulse_idx})')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is not None:
        single_img_path = os.path.join(output_dir, f"single_pulse_{pulse_idx}_comparison.png")
        plt.savefig(single_img_path, dpi=300, bbox_inches='tight')
        print(f"Saved single pulse comparison: {single_img_path}")
    
    plt.show()


# 使用例
if __name__ == "__main__":
    # TDX1からトリガーを検出済みという前提
    from trigger_detection import detect_triggers_from_signal
    
    file_path = "your_file.mat"
    start_time = 0.444
    duration = 0.001
    amplitude_threshold = 2.0
    
    # 1. TDX1からトリガーを検出
    trigger_points, chunk, Fs = detect_triggers_from_signal(
        file_path=file_path,
        start_time=start_time,
        duration=duration,
        amplitude_threshold=amplitude_threshold,
        window_width=0.0001  # 0.1ms
    )
    
    # トリガー時刻を計算
    trigger_times = np.array(trigger_points) / Fs + start_time
    
    # 2. 全チャンネルから300μsの波形を切り出し（ヒルベルト変換込み）
    channels = ["TDX1", "TDX2", "TDX3", "enlarged"]
    results = extract_waveforms_with_hilbert(
        file_path=file_path,
        channels=channels,
        trigger_times=trigger_times,
        starting_window=-50e-6,  # -50μs
        ending_window=250e-6,    # +250μs（合計300μs）
        neglect_time=55e-6,      # 初期55μsを除去
        output_dir="output_hilbert"
    )
    
    # 3. 平均波形をプロット
    plot_average_waveforms(results, output_dir="output_hilbert")
    
    # 4. 単一パルスの比較プロット
    plot_single_pulse_comparison(results, pulse_idx=0, output_dir="output_hilbert")
    
    print("\n=== Processing Complete ===")
    for channel, data in results.items():
        print(f"{channel}: {data['n_pulses']} pulses processed") 