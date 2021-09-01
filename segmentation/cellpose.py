from cellpose import models
from .common import read_image, write_image

# TODO add support for the workflow with (ilastik) foreground predictions


def load_model(model_type="nuclei", use_gpu=False):
    device, gpu = models.assign_device(True, use_gpu)
    model = models.Cellpose(gpu=gpu, model_type=model_type,
                            torch=True, device=device)
    return model


def compute_cellpose(model, in_path, out_path, channels=[0], out_key_prefix="segmentations"):
    im = read_image(in_path)
    seg = model.eval(im, diameter=None, flow_threshold=None,
                     channels=channels)
    out_key = f"{out_key_prefix}/cellpose"
    write_image(out_path, out_key, seg)
