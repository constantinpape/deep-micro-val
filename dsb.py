import argparse
import os
from glob import glob
from pathlib import Path
from segmentation import segment_all


def prepare_dsb(folder):
    try:
        from torch_em.data.datasets.dsb import _download_dsb
    except ImportError:
        assert os.path.exists(os.path.join(folder, "train", "images")) and\
            os.path.exists(os.path.join(folder, "test", "images"))
        return
    _download_dsb(folder, source="reduced", download=True)


# TODO try the full stardist dsb model (issues with channels)
# TODO model download
def require_models(folder):
    # model download from zenodo in python, similar to:
    # https://github.com/ilastik/bioimage-io-models/blob/main/src/ilastik-app.imjoy.html#L116
    # (maybe this already exists?!)
    model_folder = os.path.join(folder, "models")
    os.makedirs(model_folder, exist_ok=True)

    # FIXME load_resource_description fails for abs path,
    # see https://github.com/bioimage-io/spec-bioimage-io/issues/228
    # hence, we need to cast to rel paths or pathlib.Path

    # affinity_url = "todo"
    affinity_model = os.path.join(model_folder, "DSB-Nuclei-AffinityModel.zip")
    # affinity_model = os.path.relpath(affinity_model)
    affinity_model = Path(affinity_model)
    assert os.path.exists(affinity_model), affinity_model

    # boundary_url = "todo"
    boundary_model = os.path.join(model_folder, "DSB-Nuclei-BoundaryModel.zip")
    # boundary_model = os.path.relpath(boundary_model)
    boundary_model = Path(affinity_model)
    assert os.path.exists(boundary_model), boundary_model

    # stardist_url = "todo"
    stardist_model = os.path.join(model_folder, "2D_dsb2018")
    assert os.path.exists(stardist_model), stardist_model

    return affinity_model, boundary_model, stardist_model


def get_offsets(model_path):
    try:
        from bioimageio.spec import load_resource_description
    except ImportError:
        return None
    model = load_resource_description(model_path)
    return model.config["mws"]["offsets"]


def main(args):
    prepare_dsb(args.folder)
    folder = os.path.join(args.folder, args.split)
    assert os.path.exists(folder)

    input_files = glob(os.path.join(folder, "images", "*.tif"))
    input_files.sort()
    output_folder = os.path.join(folder, "predictions")

    if args.segment:

        affinity_model, boundary_model, stardist_model = require_models(args.folder)
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
