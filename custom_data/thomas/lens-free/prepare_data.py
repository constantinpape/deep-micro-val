import os
from glob import glob

import h5py
import numpy as np
import napari
import pandas as pd
from imageio import imread
from PIL import Image


def prepare_image(root, out_folder=None):
    image_files = glob(os.path.join(root, "*.tif")) + glob(os.path.join(root, "*.png"))
    c0, c1, c2, mask = None, None, None, None
    for im_file in image_files:
        if "adjusted" in im_file:
            c0 = im_file
        elif "IPR_N10_phase_combined" in im_file:
            c1 = im_file
        elif "N10_phase_combined" in im_file:
            c2 = im_file
        elif "mask" in im_file:
            mask = im_file
    assert c0 is not None and c1 is not None and c2 is not None and mask is not None

    pred_files = glob(os.path.join(root, "*.h5"))
    assert len(pred_files) == 1
    pred_file = pred_files[0]

    coord_files = glob(os.path.join(root, "*.csv"))
    assert len(coord_files) == 1
    coord_file = coord_files[0]
    coords = pd.read_csv(coord_file).index.to_numpy()
    # the coordinates are in the phase contrast coordinate space, so we need to rescale to the lfi space
    scale_factor = 2.108
    coords = np.array([co[2:][::-1] for co in coords]) / scale_factor

    # can't be read with imread due to compression
    with Image.open(mask) as f:
        mask = np.array(f)

    c0, c1, c2 = imread(c0), imread(c1), imread(c2)
    with h5py.File(pred_file, "r") as f:
        pred = f["exported_data"][:]
    pred = pred.transpose((1, 0, 2))

    if out_folder is None:
        v = napari.Viewer()
        v.add_image(c0)
        v.add_image(c1)
        v.add_image(c2)
        v.add_image(mask)
        v.add_image(pred)
        v.add_points(coords)
        napari.run()
    else:
        name = os.path.split(root)[1]
        name = name.replace("Dag", "day")
        name = name.replace(" ", "") + ".h5"
        name = name.replace("SecondSet", "second_set")
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, name)
        print("Saving data to", out_path)
        with h5py.File(out_path, "a") as f:
            f.create_dataset("c0", data=c0, compression="gzip")
            f.create_dataset("c1", data=c1, compression="gzip")
            f.create_dataset("c2", data=c2, compression="gzip")
            f.create_dataset("mask", data=mask, compression="gzip")
            f.create_dataset("pred", data=pred, compression="gzip")
            f.create_dataset("seeds", data=coords, compression="gzip")


def main():
    # folder = "/g/kreshuk/Deckers/Constantine_LFI_Probabilities_CellCounts"
    folder = "/home/pape/Work/data/thomas/training_data/v2/original"
    out_folder = "/home/pape/Work/data/thomas/training_data/v2/prepared"
    root_names = os.listdir(folder)
    for root_name in root_names:
        print("Inspecting", root_name)
        prepare_image(os.path.join(folder, root_name), out_folder)


if __name__ == "__main__":
    main()
