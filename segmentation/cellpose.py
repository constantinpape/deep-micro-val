from cellpose import models
from .common import read_image, write_image

from skimage.transform import resize, downscale_local_mean

# TODO add support for the workflow with (ilastik) foreground predictions


def load_model(model_type="nuclei", use_gpu=False):
    device, gpu = models.assign_device(True, use_gpu)
    model = models.Cellpose(gpu=gpu, model_type=model_type,
                            torch=True, device=device)
    return model


def _reshape(im, new_shape, scale_mode):
    if scale_mode == "mean":
        scale_factor = tuple(sh // ns for sh, ns in zip(im.shape, new_shape))
        im = downscale_local_mean(im, scale_factor)
    elif scale_mode == "nearest":
        im = resize(im, new_shape, order=0, preserve_range=True, anti_aliasing=False).astype(im.dtype)
    elif scale_mode == "linear":
        im = resize(im, new_shape, order=1).astype(im.dtype)
    elif scale_mode == "cubic":
        im = resize(im, new_shape, order=3).astype(im.dtype)
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")
    return im


def compute_cellpose(model, in_path, out_path, channels=[0, 0], out_key_prefix="segmentations",
                     reshape=None, scale_mode="mean"):
    im = read_image(in_path)
    if reshape is not None:
        old_shape = im.shape
        im = _reshape(im, reshape, scale_mode)
    seg = model.eval(im, diameter=None, flow_threshold=None, channels=channels)[0]

    if reshape is not None:
        seg = _reshape(seg, old_shape, "nearest")

    out_key = f"{out_key_prefix}/cellpose"
    write_image(out_path, out_key, seg)
