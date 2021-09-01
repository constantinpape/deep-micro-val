import os

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from .common import read_image, write_image


def load_model(path):
    model_root, model_name = os.path.split(path.rstrip('/'))
    model = StarDist2D(None, name=model_name, basedir=model_root)
    return model


def compute_stardist(model, in_path, out_path, out_key_prefix="segmentations"):
    im = read_image(in_path)
    lower_percentile = 1
    upper_percentile = 99.8
    axis_norm = (0, 1)
    im = normalize(im, lower_percentile, upper_percentile, axis=axis_norm)
    pred, _ = model.predict_instances(im)
    out_key = f"{out_key_prefix}/stardist"
    write_image(out_path, out_key, pred)
