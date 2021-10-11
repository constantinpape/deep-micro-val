from pathlib import Path

import h5py
import numpy as np

from bioimageio.core import load_resource_description
from bioimageio.core.prediction import load_image, predict, predict_with_padding, predict_with_tiling
from bioimageio.core.prediction_pipeline import create_prediction_pipeline

from elf.segmentation.utils import normalize_input
from elf.segmentation.mutex_watershed import mutex_watershed as mws
from elf.segmentation.watershed import apply_size_filter

from skimage.measure import label
from skimage.segmentation import watershed

from .common import write_image


# TODO add multicut baseline
#
# Utility Functions
#


def compute_all_baselines(in_path, out_path, affinity_model=None, boundary_model=None,
                          offsets=None, with_foreground=True, padding=None, tiling=None):
    assert sum([padding is not None, tiling is not None]) <= 1
    assert sum([affinity_model is not None, boundary_model is not None]) >= 1

    # affinty based baselines
    if affinity_model is not None:
        assert offsets is not None
        predict_affinities(affinity_model, in_path, out_path,
                           with_foreground=with_foreground,
                           padding=padding, tiling=tiling)
        threshold = 0.5 if with_foreground else None
        mutex_watershed(out_path, offsets, threshold)

    # boundary based baselines
    if boundary_model is not None:
        predict_boundaries(boundary_model, in_path, out_path,
                           with_foreground=with_foreground,
                           padding=padding, tiling=tiling)
        connected_components(out_path)
        connected_components_with_boundaries(out_path)


def load_model(path, devices=None):
    model = load_resource_description(Path(path))
    return create_prediction_pipeline(bioimageio_model=model, devices=devices)


#
# CNN Prediction
#


def _pred_impl(model, in_path, padding, tiling):
    axes = tuple(model.input_axes)
    image = load_image(in_path, axes)
    if padding is not None:
        pred = predict_with_padding(model, image, padding)
    elif tiling is not None:
        pred = predict_with_tiling(model, image, tiling)
    else:
        pred = predict(model, image)
    assert pred.shape[0] == 1, f"Have more than one batch {pred.shape}"
    return pred[0]


def predict_affinities(model, in_path, out_path,
                       with_foreground=True, padding=None, tiling=None,
                       out_key_prefix="predictions"):
    pred = _pred_impl(model, in_path, padding, tiling)
    if with_foreground:
        out_key = f"{out_key_prefix}/foreground"
        write_image(out_path, out_key, pred[0])
        pred = pred[1:]
    out_key = f"{out_key_prefix}/affinities"
    write_image(out_path, out_key, pred)


def predict_boundaries(model, in_path, out_path,
                       with_foreground=True, padding=None, tiling=None,
                       out_key_prefix="predictions"):
    pred = _pred_impl(model, in_path, padding, tiling)
    if with_foreground:
        out_key = f"{out_key_prefix}/foreground"
        write_image(out_path, out_key, pred[0])
        pred = pred[1]
    out_key = f"{out_key_prefix}/boundaries"
    write_image(out_path, out_key, pred)


#
# Segmentation Baselines
#


def connected_components(path, threshold=0.5, in_key_prefix="predictions", out_key_prefix="segmentations"):
    with h5py.File(path, "a") as f:
        foreground = f[in_key_prefix]["foreground"][:]
        seg = label(normalize_input(foreground) > threshold)
        out_key = f"{out_key_prefix}/connected_components"
        write_image(f, out_key, seg)


def connected_components_with_boundaries(path, threshold=0.5, in_key_prefix="predictions",
                                         out_key_prefix="segmentations"):
    with h5py.File(path, "a") as f:
        foreground = f[in_key_prefix]["foreground"][:]
        boundaries = f[in_key_prefix]["boundaries"][:]

        seeds = label(np.clip(foreground - boundaries, 0, 1) > threshold)
        mask = normalize_input(foreground) > threshold
        seg = watershed(boundaries, markers=seeds, mask=mask)

        out_key = f"{out_key_prefix}/connected_components_with_boundaries"
        write_image(f, out_key, seg)


# FIXME somehow the mask is not used correctly
def mutex_watershed(path, offsets, threshold=None, strides=[3, 3], min_size=25,
                    in_key_prefix="predictions", out_key_prefix="segmentations"):
    with h5py.File(path, "a") as f:
        g = f[in_key_prefix]
        affinities = g["affinities"][:]

        if threshold is None:
            mask = None
        else:
            assert "foreground" in g
            fg = g["foreground"][:]
            mask = fg > threshold
        seg = mws(affinities, offsets, strides,
                  randomize_strides=True, mask=mask).astype("uint32")
        if min_size > 0:
            seg = apply_size_filter(seg, affinities[0], min_size)[0]

        out_key = f"{out_key_prefix}/mutex_watershed"
        write_image(f, out_key, seg)
