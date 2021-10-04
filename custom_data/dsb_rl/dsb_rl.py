import os
import sys
from glob import glob

import imageio
import h5py
import numpy as np
import vigra
import elf.segmentation as eseg
from elf.segmentation.utils import map_background_to_zero, normalize_input
from torch_em.transform.raw import normalize_percentile

DST = "/scratch/pape/dsb/test"
SP = "/g/kreshuk/pape/Work/data/data_science_bowl/dsb2018/test/superpixels"


def copy_predictions():
    models = (1, 2)
    src = "/g/kreshuk/hilt/projects/data/dsb/medium/predictions"
    for model_id in models:
        out_key = f"segmentations/rl_model{model_id}"
        pred_folder = os.path.join(src, str(model_id))
        preds = glob(os.path.join(pred_folder, "*.h5"))
        for pred in preds:
            name = os.path.split(pred)[1].split("_")[0]
            out_path = os.path.join(DST, "predictions", f"{name}.h5")
            assert os.path.exists(out_path), out_path
            with h5py.File(pred, "r") as f:
                seg = f["predictions"][:]
            with h5py.File(out_path, "a") as f:
                f.create_dataset(out_key, data=seg, compression="gzip")


def _run_mc(bd, sp, beta=0.6):
    rag = eseg.compute_rag(sp, n_threads=8)
    feats = eseg.compute_boundary_mean_and_length(rag, bd, n_threads=8)[:, 0]
    costs = eseg.compute_edge_costs(feats)
    node_labels = eseg.multicut.multicut_kernighan_lin(rag, costs, beta=beta)
    seg = eseg.project_node_labels_to_pixels(rag, node_labels, n_threads=8)
    seg, _ = eseg.watershed.apply_size_filter(seg.astype("uint32"), bd.astype("float32"), 25)
    seg = map_background_to_zero(seg)
    return seg


def multicut_from_boundaries():
    inputs = glob(os.path.join(DST, "predictions", "*.h5"))
    out_key = "segmentations/mc_boundaries"
    for path in inputs:
        with h5py.File(path, "r") as f:
            bd = np.clip(f["predictions/boundaries"][:], 0, 1)
        sp_path = os.path.join(SP, os.path.split(path)[1])
        with h5py.File(sp_path, "r") as f:
            sp = f["gt_intersected"][:]
        seg = _run_mc(bd, sp)
        with h5py.File(path, "a") as f:
            f.create_dataset(out_key, data=seg, compression="gzip")


def multicut_from_filter(sigma=2.0):
    inputs = glob(os.path.join(DST, "images", "*.tif"))
    out_key = "segmentations/mc_filter"
    for path in inputs:

        im = imageio.imread(path)
        # TODO this normalization is a good strategy, consider using this in general for boundary / affinity predictions
        bd = normalize_percentile(
            vigra.filters.gaussianGradientMagnitude(normalize_input(im), sigma),
            lower=2.5, upper=97.5
        )
        bd = np.clip(bd, 0, 1)

        name = os.path.splitext(os.path.split(path)[1])[0] + ".h5"
        sp_path = os.path.join(SP, name)
        with h5py.File(sp_path, "r") as f:
            sp = f["gt_intersected"][:]

        seg = _run_mc(bd, sp)
        out_path = os.path.join(DST, "predictions", name)
        with h5py.File(out_path, "a") as f:
            f.create_dataset(out_key, data=seg, compression="gzip")


def debug_mc():
    import napari

    inputs = glob(os.path.join(DST, "images", "*.tif"))
    inputs.sort()
    sps = glob(os.path.join(SP, "*.h5"))
    sps.sort()

    inp, sp = inputs[0], sps[0]
    im = normalize_input(imageio.imread(inp))
    with h5py.File(sp, "r") as f:
        sp = f["gt_intersected"][:]

    sigma = 2.0
    # TODO this normalization is a good strategy, should consider using this also for boundary and affinity predictions
    bd1 = normalize_percentile(vigra.filters.gaussianGradientMagnitude(im, sigma), lower=2.5, upper=97.5)
    bd1 = np.clip(bd1, 0, 1)
    seg1 = _run_mc(bd1, sp)

    v = napari.Viewer()
    v.add_image(im, name="image")
    v.add_labels(sp, name="superpixels")

    v.add_image(bd1, name="ggm")
    v.add_labels(seg1, name="mc-ggm")
    napari.run()


def additional_segmentations():
    copy_predictions()
    multicut_from_boundaries()
    multicut_from_filter()


def _get_preds():
    pred_files = glob(os.path.join(DST, "predictions", "*.h5"))
    total = len(pred_files)
    pred_files = [ff for ff in pred_files if "segmentations/rl_model1" in h5py.File(ff, "r")]
    print(len(pred_files), "/", total, "have RL predictions")
    return pred_files


def check():
    sys.path.append("..")
    from segmentation.inspection import inspect_all
    preds = _get_preds()
    names = [os.path.splitext(os.path.split(p)[1])[0] for p in preds]
    im_files = [os.path.join(DST, "images", f"{name}.tif") for name in names]
    inspect_all(im_files, preds)


def evaluate():
    sys.path.append("..")
    from segmentation.evaluation import evaluate_all
    preds = _get_preds()
    names = [os.path.splitext(os.path.split(p)[1])[0] for p in preds]
    gt_files = [os.path.join(DST, "masks", f"{name}.tif") for name in names]
    evaluate_all(gt_files, preds, "")


if __name__ == '__main__':
    # additional_segmentations()
    # check()
    evaluate()
    # debug_mc()
