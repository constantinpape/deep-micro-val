import argparse
import os
from glob import glob

import imageio
import h5py
import napari
import numpy as np
from tifffile import TiffWriter
from tqdm import tqdm


def fix_data(vol):
    shape = vol.shape
    x = vol.reshape((shape[0] * shape[1], shape[2], shape[3]))
    y = np.zeros_like(vol)

    nc, nz = shape[1], shape[0]
    for c in range(nc):
        start = c * nz
        stop = start + nz
        y[:, c] = x[start:stop]

    return y


def load_seg(seg_files):
    seg = []
    for segf in seg_files:
        with h5py.File(segf, "r") as f:
            seg.append(f["exported_data"][:])
    seg = np.concatenate(seg)
    return seg


def view(image_path, seg_files):
    image = imageio.volread(image_path)
    seg = load_seg(seg_files)
    name = os.path.basename(image_path)
    # the usual data screw-up
    if name != "MMStack_Pos0.ome.tif":
        print("Fix image:", name)
        image = fix_data(image)
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(seg[:, None], scale=(1, 1, 4, 4))
    v.title = name
    napari.run()


def save(image_path, image_out, seg_files, seg_out):
    image = imageio.volread(image_path)
    seg = load_seg(seg_files)
    name = os.path.basename(image_path)
    # the usual data screw-up
    if name != "MMStack_Pos0.ome.tif":
        print("Fix image:", name)
        image = fix_data(image)
    with TiffWriter(image_out) as tif:
        tif.save(image)
    with TiffWriter(seg_out) as tif:
        tif.save(seg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embryo")
    parser.add_argument("-p", "--seg_prefix", default="boundaryseg")
    parser.add_argument("-v", "--view", type=int, default=0)
    parser.add_argument("-s", "--save", type=int, default=0)
    args = parser.parse_args()
    embryo = args.embryo
    seg_prefix = args.seg_prefix
    assert embryo in ("embryo1_embryo2", "embryo3")
    assert seg_prefix in ("boundaryseg", "dapi")
    seg_type = "nuclei" if seg_prefix == "dapi" else "cell"

    data_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/{embryo}_raw"
    data_out_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/{embryo}"
    seg_in_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/segmentation_raw/{embryo}"
    seg_out_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/segmentation/{embryo}/{seg_type}"
    os.makedirs(seg_out_folder, exist_ok=True)
    os.makedirs(data_out_folder, exist_ok=True)

    images = glob(os.path.join(data_folder, "*.ome.tif"))
    images.sort()

    for image in tqdm(images):
        name = os.path.basename(image)
        pos = name[name.find("Pos"):].rstrip(".ome.tif")
        seg_pattern = os.path.join(seg_in_folder, f"{seg_prefix}_{pos}-*")
        seg_files = glob(seg_pattern)
        seg_files.sort()
        assert len(seg_files) == 6
        if args.view:
            view(image, seg_files)
        if args.save:
            image_out = os.path.join(data_out_folder, name)
            seg_out = os.path.join(seg_out_folder, name)
            save(image, image_out, seg_files, seg_out)


if __name__ == "__main__":
    main()
