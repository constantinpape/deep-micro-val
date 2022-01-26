import os
from glob import glob
import h5py
import napari


def check_thresholds(path, thresholds):
    with h5py.File(path, "r") as f:
        im = f["c2"][:]
        pred = f["pred"][:]
    fg_probs = 1. - pred[..., 2]
    boundaries = pred[..., 1]
    fg_probs -= boundaries
    v = napari.Viewer()
    v.add_image(im)
    v.add_image(fg_probs, visible=False)
    for thresh in thresholds:
        v.add_labels(fg_probs > thresh, name=f"treshold-{thresh}")
    napari.run()


if __name__ == "__main__":
    folder = "/home/pape/Work/data/deckers/lens-free/training_data/v2/prepared"
    files = glob(os.path.join(folder, "*.h5"))
    thresholds = [0.3, 0.4, 0.5]
    for ff in files:
        check_thresholds(ff, thresholds)
