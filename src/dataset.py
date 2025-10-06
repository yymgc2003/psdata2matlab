from .utils import calculate_gvf_and_signal
def process_case_and_return_dataset(case_name, base_dir, csv_dir,
                                    output_path,
                                    rolling_window = False,
                                    log1p = False,
                                    window_size=20, window_stride=10,
                                    label_dim=2):
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
        input_tmp = input_tmp/np.max(input_tmp)
        if log1p:
            input_tmp = np.log1p(input_tmp)
        #print(f'input_tmp after log1p: {input_tmp}')
        if np.isnan(input_tmp).any():     
            print(f'nan exists')
        x_list.append(input_tmp)
        if np.isnan(x_list).any():
            print(f'nan exists')
        t_list.append(target_tmp)
    #print(len(x_list))
    #print(len(t_list))
    # Convert lists to numpy arrays for machine learning
    x_train = np.array(x_list)
    t=np.arange(0,50e-6,50e-6/len(x_train[0]))
    plt.figure(figsize=(10, 4))
    plt.plot(t*1e6, x_train[0], color='blue', label='Original Pulse')
    plt.legend()
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('Case {}'.format(case_name))
    plt.tight_layout()
    import os
    #base_name = os.path.splitext(os.path.basename(file_path))[0]
    #save_path = os.path.join(base_dir, 'graph')
    new_save_path = os.path.join(base_dir, f"{case_name}_img.png")
    print(new_save_path)
    plt.savefig(new_save_path)
    plt.close()
    #print(f'x_train b4 convert: {x_list}')
    #print(f'x_train after convert: {x_train}')
    t_train = np.array(t_list)
    #print(x_train.shape)
    #print(t_train.shape)
    #print(np.max(x_train),np.min(x_train))
    return x_train, t_train
