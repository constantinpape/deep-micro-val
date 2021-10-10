import argparse
import os

import imageio
import h5py
import napari
import numpy as np
import pandas as pd

from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed


def get_seeds(im, points, radius=6):
    coords = np.round(points).astype("int")
    coords = tuple(coords[:, i] for i in range(coords.shape[1]))

    # this is more correct
    seeds = np.zeros(im.shape, dtype="uint32")
    seed_ids = np.arange(1, len(points) + 1)
    seeds[coords] = seed_ids

    dist = distance_transform_edt(seeds == 0)
    seeds = watershed(dist.max() - dist, seeds, mask=dist < 4,
                      compactness=10)
    return seeds


def get_foreground_and_boundaries(seeds, threshold=0.5):
    with h5py.File("./data/nuc_probs.h5", "r") as f:
        data = f["exported_data"][:]

    foreground = data[..., 2] < threshold  # 2 is the bg channel
    # enforce seeds in foreground
    foreground = np.logical_or(foreground, seeds > 0)

    boundaries = data[..., 1]  # 1 is the boundary channel
    dists = distance_transform_edt(seeds == 0)
    dists[~foreground] = 0
    max_dist = np.percentile(dists, 95)
    dists /= max_dist
    dists = np.clip(dists, 0, 1)

    alpha = 0.3
    boundaries = alpha * boundaries + (1. - alpha) * dists

    return foreground, boundaries


def seeded_segmentation(im_path, pred_path, ann_path, view=False):
    im = imageio.imread(im_path)
    points = pd.read_csv(ann_path).values[:, 3:]

    seeds = get_seeds(im, points)
    foreground, boundaries = get_foreground_and_boundaries(pred_path, seeds)
    seg = watershed(boundaries, seeds, mask=foreground)

    if view:
        v = napari.Viewer()
        v.add_image(im)
        v.add_points(points)

        v.add_labels(seeds)
        v.add_labels(foreground, visible=False)
        v.add_image(boundaries)
        v.add_labels(seg)

        napari.run()

    return seg


def _to_tif(name):
    pos = name[name.find("Pos"):name.find("Pos")+4]
    return f"MMStack_{pos}.ome.tif"


def _to_pred(name):
    return name.replace(".ome.tif", ".ome_Probabilities Stage 2.h5")


# TODO
def export_segmentation(im_name, ann_name, seg):
    out_folder = "./data/segmentations"
    os.makedirs(out_folder, exist_ok=True)


def segment_all(save):
    annotation_folder = "./data/annotations"
    names = os.listdir(annotation_folder)
    data_folder = "./data/nuclei"
    pred_folder = "./data/predictions"
    for name in names:
        tif_name = _to_tif(name)
        pred_name = _to_pred(tif_name)
        im_path = os.path.join(data_folder, tif_name)
        pred_path = os.path.join(pred_folder, pred_name)
        ann_path = os.path.join(annotation_folder, name)
        seg = seeded_segmentation(im_path, pred_path, ann_path, view=not save)
        if save:
            export_segmentation(tif_name, name, seg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", type=int, default=0)
    args = parser.parse_args()
    segment_all(bool(args.save))


if __name__ == "__main__":
    main()
