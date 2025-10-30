from .utils import calculate_gvf_and_signal, hilbert_cuda
from scipy.signal import hilbert
def process_case_and_return_dataset(case_name, base_dir, csv_dir,
                                    output_path,
                                    rolling_window = False,
                                    log1p = True,
                                    if_hilbert = True,
                                    window_size=20, window_stride=10,
                                    label_dim=2, device='cuda:0',
                                    png_name='img'):
    """
    Process all .npz files in the specified case directory, extract signals and GVF,
    and return the resulting datasets for machine learning.

    Parameters
    ----------
    case_name : str
        The name of the case (e.g., "case6").
    base_dir : str
        The base directory where the processed case data is stored.

    Returns
    -------
    x_train : np.ndarray
        Array of input signals for machine learning.
    t_train : np.ndarray
        Array of target GVF values for machine learning.
    """
    import glob
    import os
    import numpy as np
    import json
    import polars as pl
    import re
    from matplotlib import pyplot as plt
    config_path = os.path.join(base_dir, "config.json")
    npz_files = sorted(glob.glob(os.path.join(base_dir, "*reflector*.npz")))
    print(npz_files)
    x_list = []
    t_list = []

    for npz_path in npz_files:

        # ーーーーーー変更部分ここからーーーーー

        # csv_dir : ../simulation/rawsignal/{case_name}
        # process_case_and_return_dataset に csv_dirの変数を追加

        match = re.search(r"(\d+)", os.path.basename(npz_path))
        loc_idx = int(match.group(1))
        if os.path.exists(os.path.join(csv_dir,'location_seed')):
            loc_dir = 'location_seed'
        else:
            loc_dir = 'location_seed1' 
        loc_dir = os.path.join(csv_dir,loc_dir)
        csv_path = os.path.join(loc_dir, f'location{loc_idx}.csv')
        print(f'csv_path:{csv_path}')
        print(f'npz_path:{npz_path}')

        # ーーーーーー変更部分ここまでーーーーー

        # input_tmp = calculate_gvf_and_signal(config_path, npz_path, csv_path,
        #                                                  label_dim=label_dim)
        input_tmp, target_tmp = calculate_gvf_and_signal(config_path, npz_path, csv_path,
                                                         label_dim=label_dim)
        # Apply rolling window
        if rolling_window:
            s =pl.Series(input_tmp)
            rolling_max = s.rolling_max(window_size=window_size)[window_size-1:]
            #print(f'rolling max 0: {rolling_max[:12]}')
            input_tmp = rolling_max.gather_every(window_stride).to_numpy()
        # Apply log(1 + x) transformation element-wise to input_tmp
        #print(f'input_tmp b4 log1p: {input_tmp}')
        print(f'input tmp shape{input_tmp.shape}')
        if if_hilbert:
            input_tmp= np.abs(hilbert(input_tmp))
        input_tmp = input_tmp/np.max(input_tmp)
        if log1p:
            input_tmp = np.log1p(input_tmp)
        #print(f'input_tmp after log1p: {input_tmp}')
        if np.isnan(input_tmp).any():     
            print(f'nan exists')
        x_list.append(input_tmp)
        if np.isnan(x_list).any():
            print(f'nan exists')
        input_tmp = np.array(input_tmp)
        t=np.arange(0,50e-6,50e-6/len(input_tmp))
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(t*1e6, input_tmp, color='blue', label='Original Pulse')
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        import os
        #base_name = os.path.splitext(os.path.basename(file_path))[0]
        #save_path = os.path.join(base_dir, 'graph')
        new_save_path = os.path.join(base_dir, f"{case_name}_{png_name}{loc_idx}.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()

        t_list.append(target_tmp)
    #print(len(x_list))
    #print(len(t_list))
    # Convert lists to numpy arrays for machine learning
    x_train = np.array(x_list)
    # t=np.arange(0,48e-6,48e-6/len(x_train[0]))
    # plt.figure(figsize=(10, 4))
    # plt.rcParams["font.size"] = 18
    # plt.plot(t*1e6, x_train[0], color='blue', label='Original Pulse')
    # plt.legend()
    # plt.xlabel('Time (μs)')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # import os
    # #base_name = os.path.splitext(os.path.basename(file_path))[0]
    # #save_path = os.path.join(base_dir, 'graph')
    # new_save_path = os.path.join(base_dir, f"{case_name}_{png_name}.png")
    # print(new_save_path)
    # plt.savefig(new_save_path)
    # plt.close()
    #print(f'x_train b4 convert: {x_list}')
    #print(f'x_train after convert: {x_train}')
    t_train = np.array(t_list)
    #print(x_train.shape)
    #print(t_train.shape)
    #print(np.max(x_train),np.min(x_train))
    return x_train, t_train

def process_case_and_png(case_name, base_dir, csv_dir,
                         rolling_window = False,
                         log1p = False,
                         if_hilbert = True,
                         window_size=20, window_stride=10,
                         device='cuda:0', png_name='img'):
    import glob
    import os
    import numpy as np
    import json
    import polars as pl
    import re
    import torch
    from matplotlib import pyplot as plt
    config_path = os.path.join(base_dir, "config.json")
    npz_files = sorted(glob.glob(os.path.join(base_dir, "*reflector*.npz")))
    print(npz_files)
    x_list = []

    for npz_path in npz_files:
        match = re.search(r"(\d+)", os.path.basename(npz_path))
        loc_idx = int(match.group(1))
        if os.path.exists(os.path.join(csv_dir,'location_seed')):
            loc_dir = 'location_seed'
        else:
            loc_dir = 'location_seed1' 
        loc_dir = os.path.join(csv_dir,loc_dir)
        csv_path = os.path.join(loc_dir, f'location{loc_idx}.csv')
        print(f'csv_path:{csv_path}')
        print(f'npz_path:{npz_path}')
        # input_tmp = calculate_gvf_and_signal(config_path, npz_path, csv_path,
        #                                                  label_dim=label_dim)
        npz_dict = np.load(npz_path)
        input_tmp = npz_dict["processed_data"][0,:,0]
        raw_tmp = npz_dict["processed_data"][0,:,0]
        fs = npz_dict["fs"]
        print(f'fs: {fs}')
        input_tmp_size = np.shape(input_tmp)[0]
        input_index_list = list(range(2500))
        for i in range(2500):
            input_index_list[i] = int(input_tmp_size/2500*i)
        #print(f'input_index_list: {input_index_list}')
        #print(f'if nan:{np.isnan(input_tmp).any()}')
        input_tmp_new2 =  [input_tmp[i] for i in input_index_list]#計2500になるようにデータを取得
        input_tmp_new2 = np.array(input_tmp_new2)
        input_tmp = input_tmp_new2
        #raw_tmp = raw_tmp/np.max(raw_tmp)
        # Apply rolling window
        if rolling_window:
            s =pl.Series(input_tmp)
            rolling_max = s.rolling_max(window_size=window_size)[window_size-1:]
            #print(f'rolling max 0: {rolling_max[:12]}')
            input_tmp = rolling_max.gather_every(window_stride).to_numpy()
        # Apply log(1 + x) transformation element-wise to input_tmp
        #print(f'input_tmp b4 log1p: {input_tmp}')
        print(f'input tmp shape{input_tmp.shape}')
        if if_hilbert:
            input_tmp= np.abs(hilbert(input_tmp))
        input_tmp = input_tmp/np.max(input_tmp)
        if log1p:
            input_tmp = np.log1p(input_tmp)
        #print(f'input_tmp after log1p: {input_tmp}')
        if np.isnan(input_tmp).any():     
            print(f'nan exists')
        input_tmp = np.array(input_tmp)

        len1 = len(input_tmp)
        print(f'input_tmp max: {np.argmax(input_tmp[len1//2:])}')

        t=np.arange(0e-6,50e-6,50e-6/len(input_tmp))
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(t*1e6, input_tmp, color='blue', label='Processed Signal')
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        import os
        #base_name = os.path.splitext(os.path.basename(file_path))[0]
        #save_path = os.path.join(base_dir, 'graph')
        new_save_path = os.path.join(base_dir, f"{case_name}_{png_name}{loc_idx}.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()

        len1 = len(raw_tmp)

        t=np.arange(0e-6,50e-6,50e-6/len(raw_tmp))
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(t*1e6, raw_tmp*1e-3, color='blue', label='Raw Signal')
        plt.legend()
        plt.xlabel('Time (μs)')
        plt.ylabel('Amplitude (kPa)')
        plt.tight_layout()
        import os
        #base_name = os.path.splitext(os.path.basename(file_path))[0]
        #save_path = os.path.join(base_dir, 'graph')
        new_save_path = os.path.join(base_dir, f"{case_name}_raw{loc_idx}.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()

        input_tensor = torch.from_numpy(raw_tmp).float()
        input_tensor = input_tensor.to(device)

        Xf = torch.fft.fft(input_tensor)
        Xf = torch.abs(Xf)
        Xf = torch.pow(Xf, 2)
        Xf = Xf.cpu().numpy()
        print(f'Xf: {Xf[200]}')
        Xf = Xf[:len(raw_tmp)//2]
        Xf = Xf/len(Xf)/fs/2
        freq = np.arange(0,25e6,1/50e-6)

        plt.figure(figsize=(10, 4))
        plt.rcParams["font.size"] = 18
        plt.plot(freq*1e-6, Xf[0:len(freq)], color='blue', label='FFT')
        plt.axvline(x=4, color='r',linestyle='--', linewidth=1.5, label='4 MHz')
        label_text = '4 MHz'
        plt.text(plt.xlim()[1]*0.145, plt.ylim()[1]*1.1,label_text, 
         color='r', 
         fontsize=18,
         rotation=0,         # テキストの回転 (90度にすると縦書きになる)
         ha='left',          # Horizontal Alignment: 左寄せ
         va='top'            # Vertical Alignment: 上端合わせ
        )
        plt.legend()
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        import os
        #base_name = os.path.splitext(os.path.basename(file_path))[0]
        #save_path = os.path.join(base_dir, 'graph')
        new_save_path = os.path.join(base_dir, f"{case_name}_{png_name}{loc_idx}_fft.png")
        print(new_save_path)
        plt.savefig(new_save_path)
        plt.close()