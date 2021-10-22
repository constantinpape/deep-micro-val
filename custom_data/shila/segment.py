import argparse
import os

import bioimageio.core
import imageio
import numpy as np
import pandas as pd
import tifffile

from bioimageio.core.prediction import predict
from elf.segmentation.mutex_watershed import mutex_watershed as mws
from elf.segmentation.watershed import apply_size_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from tqdm import tqdm


def predict_image(model, image):
    image = image[None, None]
    pred = predict(model, image)[0]
    return pred[0], pred[1:]


def _get_seeds(annotations, shape, radius=6):
    coords = np.round(annotations).astype("int")
    coords = tuple(coords[:, i] for i in range(coords.shape[1]))

    seeds = np.zeros(shape, dtype="uint32")
    seed_ids = np.arange(1, len(annotations) + 1)
    seeds[coords] = seed_ids

    dist = distance_transform_edt(seeds == 0)
    seeds = watershed(dist.max() - dist, seeds, mask=dist < radius)
    return seeds


def _get_foreground(fg_pred, seeds, threshold):
    foreground = fg_pred > threshold
    # enforce seeds in foreground
    foreground = np.logical_or(foreground, seeds > 0)
    return foreground


def _get_hmap(affs, foreground, seeds, alpha=0.9):
    boundaries = np.mean(affs[:4], axis=0)
    dists = distance_transform_edt(seeds == 0)
    dists[~foreground] = 0
    max_dist = np.percentile(dists, 95)
    dists /= max_dist
    dists = np.clip(dists, 0, 1)
    boundaries = alpha * boundaries + (1. - alpha) * dists
    return boundaries


def segment_seeded_watershed(fg_pred, affs, annotations, threshold=0.5):
    seeds = _get_seeds(annotations, fg_pred.shape)
    foreground = _get_foreground(fg_pred, seeds, threshold)
    hmap = _get_hmap(affs, foreground, seeds)
    assert hmap.shape == foreground.shape
    seg = watershed(hmap, seeds, mask=foreground)
    return seg


def segment_mws(fg, affs, threshold=0.5, strides=[3, 3], min_size=50):
    offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3],
               [-9, 0], [0, -9], [-27, 0], [0, -27]]
    mask = fg > threshold
    seg = mws(affs, offsets, strides, randomize_strides=True, mask=mask).astype("uint32")
    if min_size > 0:
        seg = apply_size_filter(seg, affs[0], min_size)[0]
    return seg


def _write_res(path, seg, full_shape, cz):
    full_seg = np.zeros(full_shape, dtype="int16")
    full_seg[int(cz[0]), int(cz[1])] = seg
    with tifffile.TiffWriter(path) as tif:
        tif.save(full_seg)


def segment_image(model, data_path, annotation_path, seg_root, name):
    image = imageio.volread(data_path)
    full_shape = image.shape
    annotations = pd.read_csv(annotation_path).iloc[:, 1:]

    cz = annotations.iloc[0, :2]
    annotations = annotations.iloc[:, 2:]
    image = image[int(cz[0]), int(cz[1])]

    fg, affs = predict_image(model, image)
    seg_ws = segment_seeded_watershed(fg, affs, annotations)
    seg_mws = segment_mws(fg, affs)

    path_ws = os.path.join(seg_root, "watershed", name.replace(".csv", ".ome.tif"))
    _write_res(path_ws, seg_ws, full_shape, cz)
    path_mws = os.path.join(seg_root, "mutex_watershed", name.replace(".csv", ".ome.tif"))
    _write_res(path_mws, seg_mws, full_shape, cz)


def _to_tif(name):
    pos = name[name.find("Pos"):].rstrip(".csv")
    return f"MMStack_{pos}.ome.tif"


def _load_model(version, device):
    ckpt = "./checkpoints/nuclei_v3/bioimageio-model-affinities/custom-nucleus-segmentation.zip"
    assert os.path.exists(ckpt), ckpt
    resource = bioimageio.core.load_resource_description(ckpt)
    devices = None if device is None else [device]
    model = bioimageio.core.prediction_pipeline.create_prediction_pipeline(bioimageio_model=resource, devices=devices)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, type=int)
    parser.add_argument("-d", "--device", default=None, type=str)
    args = parser.parse_args()

    data_folder = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29"
    annotation_folder = "./point_annotations"
    names = os.listdir(annotation_folder)
    names.sort()

    seg_root = "/g/kreshuk/data/marioni/shila/segmentation"
    os.makedirs(seg_root, exist_ok=True)
    os.makedirs(os.path.join(seg_root, "watershed"), exist_ok=True)
    os.makedirs(os.path.join(seg_root, "mutex_watershed"), exist_ok=True)

    model = _load_model(args.version, args.device)

    for name in tqdm(names):
        annotation_path = os.path.join(annotation_folder, name)
        data_path = os.path.join(data_folder, _to_tif(name))
        segment_image(model, data_path, annotation_path, model, seg_root, name)


if __name__ == "__main__":
    main()
