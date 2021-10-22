import argparse
import os

import bioimageio.core
import imageio
import pandas as pd
from tqdm import tqdm
from bioimageio.core.prediction import predict


def predict_image(model, image):
    image = image[None, None]
    pred = predict(model, image)[0]
    return pred[0], pred[1:]


def segment_seeded_watershed():
    pass


def segment_mws():
    pass


def _write_res(path, seg, full_shape, cz):
    pass


def segment_image(model, data_path, annotation_path, seg_root, name):
    image = imageio.volread(data_path)
    full_shape = image.shape
    annotations = pd.read_csv(annotation_path).iloc[:, 1:]

    cz = annotations.iloc[0, :2]
    annotations = annotations.iloc[:, 2:]
    image = image[int(cz[0]), int(cz[1])]

    fg, affs = predict_image(model, image)
    seg_ws = segment_seeded_watershed(fg, affs, annotations)
    seg_mws = segment_mws(fg, affs)

    path_ws = os.path.join(seg_root, "watershed", name.replace(".csv", ".ome.tif"))
    _write_res(path_ws, seg_ws, full_shape, cz)
    path_mws = os.path.join(seg_root, "watershed", name.replace(".csv", ".ome.tif"))
    _write_res(path_mws, seg_mws, full_shape, cz)


def _to_tif(name):
    pos = name[name.find("Pos"):name.find("Pos")+4]
    return f"MMStack_{pos}.ome.tif"


# FIXME need affinity model
def _load_model(version, device):
    ckpt = "./checkpoints/nuclei_v3/bioimageio-model/custom-nucleus-segmentation.zip"
    assert os.path.exists(ckpt), ckpt
    resource = bioimageio.core.load_resource_description(ckpt)
    devices = None if device is None else [device]
    model = bioimageio.core.prediction_pipeline.create_prediction_pipeline(bioimageio_model=resource, devices=devices)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, type=int)
    parser.add_argument("-d", "--device", default=None, type=str)
    args = parser.parse_args()

    data_folder = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29"
    annotation_folder = "./point_annotations"
    names = os.listdir(annotation_folder)
    names.sort()

    seg_root = "/g/kreshuk/data/marioni/shila/segmentation"
    os.makedirs(seg_root, exist_ok=True)
    os.makedirs(os.path.join(seg_root, "watershed"), exist_ok=True)
    os.makedirs(os.path.join(seg_root, "mutex_watershed"), exist_ok=True)

    model = _load_model(args.version, args.device)

    for name in tqdm(names):
        annotation_path = os.path.join(annotation_folder, name)
        data_path = os.path.join(data_folder, _to_tif(name))
        segment_image(model, data_path, annotation_path, model, seg_root, name)


if __name__ == "__main__":
    main()
