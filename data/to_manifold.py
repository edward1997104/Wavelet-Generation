import os
from fnmatch import fnmatch
import multiprocessing as mp
import traceback

manifold_path = 'external/Manifold/build/manifold'
def to_manifold(args):
    path, name = args
    try:
        os.system(
            f"{manifold_path} {os.path.join(path, 'model_flipped.obj')} {os.path.join(path, 'model_flipped_manifold.obj')}")
        print(f"Done manifold conversion to {os.path.join(path, 'model_flipped.obj')}!")
    except:
        traceback.print_exc()


if __name__ == '__main__':

    root = '<PATH_TO_SHAPENET>'
    pattern = "*.obj"

    args = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern) and not "flipped" in name and not "manifold" in name and not os.path.exists(
                    os.path.join(path, 'model_flipped_manifold.obj')):
                args.append((path, name))
                # print(f"added path {path} {name}")

    print(f"len of the arrays {len(args)}")

    workers = 12
    pool = mp.Pool(workers)
    pool.map(to_manifold, args)
