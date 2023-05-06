import os
from fnmatch import fnmatch
from utils.other_utils import save_obj, load_obj
import numpy as np
import multiprocessing as mp
import traceback


def flip_axis(args):
    path, name = args
    try:
        vertices, _, faces, _ = load_obj(os.path.join(path, name))
        vertices = np.vstack((vertices[:, 2], vertices[:, 1], -vertices[:, 0])).T
        save_obj(os.path.join(path, 'model_flipped.obj'), vertices, faces)
        print(f"Done writing to {os.path.join(path, 'model_flipped.obj')}!")
    except:
        traceback.print_exc()


if __name__ == '__main__':

    root = '<PATH_TO_SHAPENET>'
    pattern = "*.obj"

    args = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern) and not "flipped" in name and not os.path.exists(
                    os.path.join(path, 'model_flipped.obj')):
                args.append((path, name))

    workers = 12
    pool = mp.Pool(workers)
    pool.map(flip_axis, args)
