import argparse
import os
import sys
from glob import glob

import imageio
import h5py


def prepare_dataset(input_folder, output_folder):
    image_folder = os.path.join(output_folder, "images")
    pred_folder = os.path.join(output_folder, "predictions")
    if os.path.exists(image_folder) and os.path.exists(pred_folder):
        return

    raw = glob(os.path.join(input_folder, "raw/*.tif"))
    raw.sort()
    cp = glob("cellpose/*.tif")
    cp = glob(os.path.join(input_folder, "cellpose/*.tif"))
    cp.sort()
    assert len(raw) == len(cp)

    os.makedirs(image_folder, exist_ok=True)
    for ii, f in enumerate(raw):
        im = imageio.imread(f)
        out_path = os.path.join(image_folder, f"z{ii}.tif")
        imageio.imwrite(out_path, im)

    os.makedirs(pred_folder, exist_ok=True)
    for ii, f in enumerate(cp):
        im = imageio.imread(f).astype("uint32")
        out_path = os.path.join(pred_folder, f"z{ii}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("segmentations/reference", data=im, compression="gzip")


def main(folder, input_folder):
    sys.path.append("../..")
    prepare_dataset(input_folder, folder)
    assert os.path.exists(folder)

    input_files = glob(os.path.join(folder, "images", "*.tif"))
    input_files.sort()
    output_folder = os.path.join(folder, "predictions")

    if args.segment:
        from segmentation import segment_all, require_nucleus_models, get_offsets
        model_folder = os.path.join(args.folder, "models")
        affinity_model, boundary_model, stardist_model = require_nucleus_models(model_folder)
        offsets = get_offsets(affinity_model)

        tiling = {"halo": {"x": 16, "y": 16}, "tile": {"x": 512, "y": 512}}
        reshap_cellpose = (512, 512)
        segment_all(input_files, output_folder,
                    affinity_model=affinity_model,
                    boundary_model=boundary_model,
                    stardist_model=stardist_model,
                    offsets=offsets,
                    tiling=tiling,
                    reshap_cellpose=reshap_cellpose)

    output_files = glob(os.path.join(output_folder, "*.h5"))
    output_files.sort()

    if args.inspect:
        from segmentation.inspection import inspect_all
        inspect_all(input_files, output_files, include_seg_names=["cellpose", "reference"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("-s", "--segment", default=0, type=int)
    parser.add_argument("-i", "--inspect", default=0, type=int)
    parser.add_argument("--input_folder", default="/home/pape/Work/data/shila/cellpose")
    args = parser.parse_args()
    main(args.folder, args.input_folder)
