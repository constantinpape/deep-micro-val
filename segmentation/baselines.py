from pathlib import Path

import h5py
import numpy as np

from bioimageio.core.predict import load_image, predict, pad_predict_crop
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import load_resource_description

from elf.segmentation.utils import normalize_input
from elf.segmentation.mutex_watershed import mutex_watershed as mws

from skimage.measure import label
from skimage.segmentation import watershed

from .common import write_image


#
# Utility Functions
#


def compute_all_baselines(in_path, out_path, affinity_model=None, boundary_model=None,
                          offsets=None, with_foreground=True, padding=None):
    assert sum(affinity_model is not None, boundary_model is not None) >= 1

    # affinty based baselines
    if affinity_model is not None:
        assert offsets is not None
        predict_affinities(affinity_model, in_path, out_path,
                           with_foreground=with_foreground, padding=padding)
        threshold = 0.5 if with_foreground else None
        mutex_watershed(out_path, offsets, threshold)

    # boundary based baselines
    if boundary_model is not None:
        predict_boundaries(boundary_model, in_path, out_path,
                           with_foreground=with_foreground, padding=padding)
        connected_components(out_path)
        connected_components_with_boundaries(out_path)


def load_model(path, devices=None):
    model = load_resource_description(Path(path))
    return create_prediction_pipeline(bioimageio_model=model, devices=devices)


#
# CNN Prediction
#


def _pred_impl(model, in_path, padding):
    axes = tuple(model.input_axes)
    image = load_image(in_path, axes)
    if padding is None:
        pred = predict(model, image)
    else:
        pred = pad_predict_crop(model, image, padding)
    return pred


def predict_affinities(model, in_path, out_path,
                       with_foreground=True, padding=None, out_key_prefix="predictions"):
    pred = _pred_impl(model, in_path, padding)
    if with_foreground:
        out_key = f"{out_key_prefix}/foreground"
        write_image(out_path, out_key, pred[0])
        pred = pred[1:]
    out_key = f"{out_key_prefix}/affinities"
    write_image(out_path, out_key, pred)


def predict_boundaries(model, in_path, out_path,
                       with_foreground=True, padding=None, out_key_prefix="predictions"):
    pred = _pred_impl(model, in_path, padding)
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


def mutex_watershed(path, offsets, threshold=None, strides=[3, 3],
                    in_key_prefix="predictions", out_key_prefix="segmentations"):
    with h5py.File(path, "a") as f:
        g = f[in_key_prefix]
        affinities = g["affinities"][:]

        if threshold is None:
            mask = None
        else:
            assert "foreground" in g
            mask = g["foreground"][:] > threshold
        seg = mws(affinities, offsets, strides,
                  randomize_strides=True, mask=mask)

        out_key = f"{out_key_prefix}/mutex_watershed"
        write_image(f, out_key, seg)
