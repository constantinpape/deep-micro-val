import argparse
import os
from glob import glob
from segmentation import segment_all, require_nucleus_models, get_offsets


def prepare_dsb(folder):
    try:
        from torch_em.data.datasets.dsb import _download_dsb
    except ImportError:
        assert os.path.exists(os.path.join(folder, "train", "images")) and\
            os.path.exists(os.path.join(folder, "test", "images"))
        return
    _download_dsb(folder, source="reduced", download=True)


def main(args):
    prepare_dsb(args.folder)
    folder = os.path.join(args.folder, args.split)
    assert os.path.exists(folder)

    input_files = glob(os.path.join(folder, "images", "*.tif"))
    input_files.sort()
    output_folder = os.path.join(folder, "predictions")

    if args.segment:

        model_folder = os.path.join(args.folder, "models")
        affinity_model, boundary_model, stardist_model = require_nucleus_models(model_folder)
        offsets = get_offsets(affinity_model)

        padding = {"x": 16, "y": 16}
        segment_all(input_files, output_folder,
                    affinity_model, boundary_model, stardist_model,
                    offsets=offsets, padding=padding)

    output_files = glob(os.path.join(output_folder, "*.h5"))
    output_files.sort()

    if args.evaluate:
        from segmentation.evaluation import evaluate_all
        mask_files = glob(os.path.join(folder, "masks", "*.tif"))
        mask_files.sort()
        res_path = os.path.join(folder, "evaluation.csv")  # TODO file format?
        evaluate_all(mask_files, output_files, res_path)
        print("Evaluation results saved to:", res_path)

    if args.inspect:
        from segmentation.inspection import inspect_all
        inspect_all(input_files, output_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("-s", "--segment", default=0, type=int)
    parser.add_argument("-e", "--evaluate", default=0, type=int)
    parser.add_argument("-i", "--inspect", default=0, type=int)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args)
