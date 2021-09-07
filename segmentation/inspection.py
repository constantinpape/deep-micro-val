import h5py
import napari
from tqdm import tqdm
from .common import read_image


def inspect(im_file, pred_file, show_segmentations, show_predictions, include_seg_names=None):
    stop = False

    v = napari.Viewer()
    im = read_image(im_file)
    v.add_image(im, name="image")

    with h5py.File(pred_file, "r") as f:

        def visit_pred(name, node):
            v.add_image(node[:], name=name)

        if show_predictions:
            f["predictions"].visititems(visit_pred)

        def visit_seg(name, node):
            if include_seg_names is None or name in include_seg_names:
                v.add_labels(node[:], name=name)

        if show_segmentations:
            f["segmentations"].visititems(visit_seg)

    @v.bind_key("x")
    def stop_viewer(v):
        nonlocal stop
        print("Stop inspection")
        stop = True

    napari.run()
    return stop


def inspect_all(image_files, prediction_files,
                show_segmentations=True, show_predictions=False, include_seg_names=None):
    assert len(image_files) == len(prediction_files)
    for im_file, pred_file in tqdm(zip(image_files, prediction_files), total=len(image_files)):
        stop = inspect(im_file, pred_file, show_segmentations, show_predictions, include_seg_names)
        if stop:
            break
