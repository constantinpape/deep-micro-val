import argparse
import os
from glob import glob
from segmentation import segment_all


def prepare_dsb(folder):
    try:
        from torch_em.data.datasets.dsb import _download_dsb
    except ImportError:
        assert os.path.exists(folder, "train", "images") and\
            os.path.exists(folder, "test", "images")
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

    # affinity_url = "todo"
    affinity_model = os.path.join(model_folder, "DSB-Nuclei-AffinityModel.zip")
    assert os.path.exists(affinity_model)

    # boundary_url = "todo"
    boundary_model = os.path.join(model_folder, "DSB-Nuclei-BoundaryModel.zip")
    assert os.path.exists(boundary_model)

    # stardist_url = "todo"
    stardist_model = os.path.join(model_folder, "2D_dsb2018")
    assert os.path.exists(stardist_model)

    return affinity_model, boundary_model, stardist_model


def get_offsets(model_path):
    try:
        from bioimageio.spec import load_resource
    except ImportError:
        return None
    model = load_resource(model_path)
    breakpoint()


def main(args):
    print("What")
    prepare_dsb(args.folder)
    print("The")
    folder = os.path.join(args.folder, args.split)
    assert os.path.exists(folder)
    print("Fuck")

    if args.segment:
        print("Globbbbbbbbbb")
        input_files = glob(os.path.join(folder, "images", "*.tif"))
        input_files.sort()
        output_folder = os.path.join(folder, "predictions")
        print("Done")

        affinity_model, boundary_model, stardist_model = require_models(args.folder)
        offsets = get_offsets(affinity_model)

        padding = {"x": 16, "y": 16}
        segment_all(input_files, output_folder,
                    affinity_model, boundary_model, stardist_model,
                    offsets=offsets, padding=padding)

    # TODO
    if args.evaluate:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("-s", "--segment", default=1, type=int)
    parser.add_argument("-e", "--evaluate", default=1, type=int)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args)
