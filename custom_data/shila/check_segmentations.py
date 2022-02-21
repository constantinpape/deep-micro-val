import argparse
import os
from glob import glob

import imageio
import napari
import numpy as np
import pandas as pd
from tqdm import tqdm


def _get_annotation_path(annotation_folder, root, cycle, name):
    pos = name[name.find("Pos"):].rstrip(".ome.tif")
    csv_name = f"{root}_points_{pos}.csv"
    if os.path.exists(os.path.join(annotation_folder, csv_name)):
        return os.path.join(annotation_folder, csv_name)
    else:
        prefix = os.path.join(annotation_folder, f"{root}_points_{cycle.replace('_', '')}_{pos}_z*.csv")
        files = glob(prefix)
        files.sort()
        return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-s", "--seg_root", required=True)
    args = parser.parse_args()

    # data_folder = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29"
    # segs_mws = "/g/kreshuk/data/marioni/shila/segmentation/mutex_watershed"
    # segs_ws = "/g/kreshuk/data/marioni/shila/segmentation/watershed"
    # annotation_folder = "./point_annotations/TimEmbryos-020420/HybCycle_29"

    seg_ws = os.path.join(args.seg_root, "watershed")
    root, cycle = args.input.split("/")[-2:]
    annotation_folder = f"./point_annotations/{root}/{cycle}"
    # seg_mws = os.path.join(args.seg_root, "mutex_watershed")

    names = os.listdir(seg_ws)
    names.sort()
    for name in tqdm(names):
        annotation_path = _get_annotation_path(annotation_folder, root, cycle, name)
        if isinstance(annotation_path, str):
            annotations = pd.read_csv(annotation_path).iloc[:, 1:]
            z, c = annotations.iloc[0, :2]
            z, c = int(z), int(z)
            annotations = annotations.iloc[:, 2:]
            bb_im = np.s_[z, c]
            bb_seg = np.s_[z]
        else:
            annotations = []
            for z, ff in enumerate(annotation_path):
                ann = pd.read_csv(ff).drop(columns=["index", "axis-1"])
                ann.iloc[:, 0] = z
                annotations.append(ann)
            annotations = pd.concat(annotations)
            bb_im = np.s_[:, -1]
            bb_seg = np.s_[:]

        im = imageio.volread(os.path.join(args.input, name))[bb_im]
        ws = imageio.volread(os.path.join(seg_ws, name))[bb_seg]
        assert im.shape == ws.shape
        # mws = imageio.volread(os.path.join(segs_mws, name))[bb_seg]

        v = napari.Viewer()
        v.title = name
        v.add_image(im)
        # v.add_labels(mws)
        v.add_labels(ws)
        v.add_points(annotations)
        napari.run()


if __name__ == "__main__":
    main()
