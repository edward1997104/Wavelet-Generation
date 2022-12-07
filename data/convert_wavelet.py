import os
import torch
import random
import numpy as np
import traceback
from multiprocessing import Pool
from configs import config
from models.module.dwt import DWTForward3d_Laplacian

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
padding_mode = 'zero'

def convert_file(args_list):

    results = []

    dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=padding_mode).to(
        device)

    for args in args_list:
        try:
            idx, path, resolution_index, clip_value = args
            assert path.endswith('.npy')
            voxels_np = np.load(path)
            voxels_torch = torch.from_numpy(voxels_np).unsqueeze(0).unsqueeze(0).float().to(device)

            if clip_value is not None:
                voxels_torch = torch.clip(voxels_torch, -clip_value, clip_value)

            low_lap, highs_lap = dwt_forward_3d_lap(voxels_torch)
            if resolution_index == config.max_depth:
                results.append(low_lap[0,0].detach().cpu().numpy()[None, :])
            else:
                results.append(highs_lap[resolution_index][0,0].detach().cpu().numpy()[None, :])
        except:
            traceback.print_exc()

        print(f"index {idx} Done!")

    results = np.concatenate(results, axis = 0)

    return results



def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':
    category_id = r'<CATEGORY_ID>'
    sdf_save_folder = r'<PATH_TO_SAVE_SDF_VALUES>'
    npy_save_folder = r'<PATH_TO_SAVE_WAVELET_COEEFFICIENT>'
    workers = 0
    resolution_index = config.max_depth
    clip_value = 0.1
    save_new_path =  os.path.join(npy_save_folder, category_id + f'_{clip_value if clip_value is not None else "no_clip"}_{config.wavelet_type}_{resolution_index}_{padding_mode}.npy')

    paths = [ os.path.join(sdf_save_folder, file) for file in os.listdir(sdf_save_folder) if file.endswith('.npy')]


    args = [ (idx, path, resolution_index, clip_value) for idx, path in enumerate(paths) ]
    print(f"{len(args)} left to be processed!")

    results = convert_file(args)

    np.save(save_new_path, results)
