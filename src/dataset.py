from .utils import calculate_gvf_and_signal
def process_case_and_return_dataset(case_name, base_dir, csv_dir):
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
    import re
    npz_dir = base_dir
    config_path = os.path.join(base_dir, "config.json")
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    print(npz_files)
    x_list = []
    t_list = []

    for npz_path in npz_files:
        match = re.search(r"(\d+)", os.path.basename(npz_path))
        loc_idx = int(match.group(1))
        loc_dir = os.path.join(csv_dir,"location_seed")
        csv_path = os.path.join(loc_dir, f'location{loc_idx}.csv')
        print(f'csv_path:{csv_path}')
        print(f'npz_path:{npz_path}')
        input_tmp, target_tmp = calculate_gvf_and_signal(config_path, npz_path, csv_path)
        # Apply log(1 + x) transformation element-wise to input_tmp
        #input_tmp = np.log1p(input_tmp)
        x_list.append(input_tmp)
        t_list.append(target_tmp)
    #print(len(x_list))
    #print(len(t_list))
    # Convert lists to numpy arrays for machine learning
    x_train = np.array(x_list)
    t_train = np.array(t_list)
    #print(x_train.shape)
    #print(t_train.shape)
    print(np.max(x_train),np.min(x_train))
    return x_train, t_train