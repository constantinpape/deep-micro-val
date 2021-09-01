import imageio
import h5py


def read_image(path):
    return imageio.imread(path)


def write_image(f, out_key, data):
    if isinstance(f, str):
        with h5py.File(f, "a") as ff:
            ds = ff.require_dataset(out_key, shape=data.shape, dtype=data.dtype, compression="gzip")
            ds[:] = data
    else:
        ds = f.require_dataset(out_key, shape=data.shape, dtype=data.dtype, compression="gzip")
        ds[:] = data
