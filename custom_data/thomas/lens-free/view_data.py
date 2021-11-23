import os
from glob import glob

import h5py
import numpy as np
import napari
import pandas as pd
from imageio import imread
from PIL import Image


def view_image(root):
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

    v = napari.Viewer()
    v.add_image(c0)
    v.add_image(c1)
    v.add_image(c2)
    v.add_image(mask)
    v.add_image(pred)
    v.add_points(coords)
    napari.run()


def main():
    folder = "/g/kreshuk/Deckers/Constantine_LFI_Probabilities_CellCounts"
    root_names = os.listdir(folder)
    for root_name in root_names:
        print("Inspecting", root_name)
        view_image(os.path.join(folder, root_name))


if __name__ == "__main__":
    main()
