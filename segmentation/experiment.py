import os
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
                   offsets, with_foreground, padding):
    if affinity_model or boundary_model:
        baselines.compute_all_baselines(in_path, out_path, affinity_model, boundary_model,
                                        offsets=offsets, with_foreground=with_foreground,
                                        padding=padding)
    if cellpose_model:
        cellpose.compute_cellpose(cellpose_model, in_path, out_path)
    if stardist_model:
        stardist.compute_stardist(stardist_model, in_path, out_path)


# TODO cellpose model type as parameter
def segment_all(files, output_folder,
                affinity_model=None, boundary_model=None, stardist_model=None,
                offsets=None, with_foreground=True, padding=None):
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
                       offsets, with_foreground, padding)
