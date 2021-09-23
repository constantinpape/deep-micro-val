import os
from pathlib import Path
from tqdm import tqdm

try:
    from . import baselines
except ImportError:
    baselines = None
try:
    from . import cellpose
except ImportError:
    cellpose = None
try:
    from . import stardist
except ImportError:
    stardist = None


def _segment_image(in_path, out_path,
                   affinity_model, boundary_model,
                   cellpose_model, stardist_model,
                   offsets, with_foreground, padding, tiling,
                   reshap_cellpose):
    if affinity_model or boundary_model:
        baselines.compute_all_baselines(in_path, out_path, affinity_model, boundary_model,
                                        offsets=offsets, with_foreground=with_foreground,
                                        padding=padding, tiling=tiling)
    if cellpose_model:
        cellpose.compute_cellpose(cellpose_model, in_path, out_path,
                                  reshape=reshap_cellpose)
    if stardist_model:
        stardist.compute_stardist(stardist_model, in_path, out_path)


# TODO cellpose model type as parameter
def segment_all(files, output_folder,
                affinity_model=None, boundary_model=None, stardist_model=None,
                offsets=None, with_foreground=True, padding=None, tiling=None,
                reshap_cellpose=None):
    os.makedirs(output_folder, exist_ok=True)

    affinity_model = None if (affinity_model is None or baselines is None) else baselines.load_model(affinity_model)
    boundary_model = None if (boundary_model is None or baselines is None) else baselines.load_model(boundary_model)
    cellpose_model = None if cellpose is None else cellpose.load_model()
    stardist_model = None if (stardist is None or stardist_model is None) else stardist.load_model(stardist_model)

    for path in tqdm(files):
        name = os.path.splitext(os.path.split(path)[1])[0]
        out_path = os.path.join(output_folder, f"{name}.h5")
        _segment_image(path, out_path,
                       affinity_model, boundary_model,
                       cellpose_model, stardist_model,
                       offsets, with_foreground,
                       padding, tiling, reshap_cellpose)


def require_nucleus_models(model_folder):
    try:
        from bioimageio.spec import export_resource_package
    except ImportError:
        export_resource_package = None

    os.makedirs(model_folder, exist_ok=True)

    # TODO proper modelzoo url
    affinity_url = ""
    affinity_model = Path(model_folder) / "DSB-Nuclei-AffinityModel.zip"
    if not affinity_model.exists() and export_resource_package is not None:
        export_resource_package(affinity_url, output_path=affinity_model)

    # TODO proper modelzoo url
    boundary_url = ""
    boundary_model = Path(model_folder) / "DSB-Nuclei-BoundaryModel.zip"
    if not boundary_model.exists() and export_resource_package is not None:
        export_resource_package(boundary_url, output_path=boundary_model)

    # TODO try the full stardist dsb model (issues with channels)
    # TODO stardist model download
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
