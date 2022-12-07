from utils.debugger import MyDebugger

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
import time
from multiprocessing import Pool
import random
import traceback

def convert_sdf_grid(arg):
    mesh_path, save_folder, resolution, recompute, scale_ratio = arg
    save_path = os.path.join(save_folder, os.path.basename(mesh_path) + f'_{resolution}.npy')
    if os.path.exists(save_path) and not recompute:
        voxels = np.load(save_path)
    else:
        try:
            start_time = time.time()
            print(f"Start processing {mesh_path}")
            if os.path.isdir(mesh_path):
                mesh = trimesh.load_mesh(os.path.join(mesh_path, 'model_flipped_manifold.obj'))
            else:
                mesh = trimesh.load_mesh(mesh_path)
            print(f"mesh with {len(mesh.vertices)} vertices....")
            voxels = mesh_to_voxels(mesh, resolution, scale_ratio = scale_ratio)
            print(f"Done converting sdf for {mesh_path}!")
            np.save(save_path, voxels)
            print(f"Compelete time : {time.time() - start_time} s for {mesh_path}!")
        except:
            traceback.print_exc()
            return None

    return None

if __name__ == '__main__':

    workers = 1
    resolution = 256
    scale_ratio = 0.9
    recompute = False
    obj_names_path = r'<PATH_TO_FILENAME>'
    data_folder = r'<PATH_TO_SHAPENET>'
    save_folder = r'<PATH_TO_SAVE_SDF_VALUES>'

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    obj_txt = open(obj_names_path, "r")
    obj_list = obj_txt.readlines()
    obj_txt.close()
    obj_list = [item.strip().split('/') for item in obj_list]

    for obj in obj_list:
        if not os.path.isdir(os.path.join(save_folder, obj[0])):
            os.mkdir(os.path.join(save_folder, obj[0]))

    args = [ (os.path.join(data_folder, objs[0], objs[1]),
              os.path.join(save_folder, objs[0]),
              resolution, recompute, scale_ratio) for objs in obj_list if not os.path.exists(os.path.join(save_folder, objs[0], objs[1] + f'_{resolution}.npy'))]
    random.shuffle(args)
    print(f"{len(args)} left to be processed!")

    pool = Pool(workers)
    pool.map(convert_sdf_grid, args)


