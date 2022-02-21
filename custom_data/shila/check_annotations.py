import argparse
import os
from glob import glob

import imageio
import napari
import pandas as pd
from tqdm import tqdm


def _to_tif(name):
    pos = name[name.find("Pos"):].rstrip(".csv")
    return f"MMStack_{pos}.ome.tif"


def show_annotations(annotation_folder, data_folder, names, show_2d):
    for name in tqdm(names):
        annotations = pd.read_csv(os.path.join(annotation_folder, name)).iloc[:, 1:]
        tif_name = _to_tif(name)
        im = imageio.volread(os.path.join(data_folder, tif_name))

        if show_2d:
            cz = annotations.iloc[0, :2]
            annotations = annotations.iloc[:, 2:]
            im = im[int(cz[0]), int(cz[1])]

        v = napari.Viewer()
        v.title = name
        v.add_image(im)
        v.add_points(annotations)
        napari.run()


def show_annotations_with_slices(annotation_folder, data_folder, names):
    prefixes = list(set([
        "_".join(name.split("_")[:-1]) for name in names
    ]))
    prefixes.sort()
    for prefix in tqdm(prefixes):
        name = f"{prefix}.csv"
        tif_name = _to_tif(name)
        image_path = os.path.join(data_folder, tif_name)
        im = imageio.volread(image_path)

        print("!!!!!!!!!!!!!!!1")
        print(im.shape)
        print("!!!!!!!!!!!!!!!1")

        annotation_files = glob(f"{annotation_folder}/{prefix}_z*")
        annotation_files.sort()

        all_annotations = []
        for z, annotation_file in enumerate(annotation_files):
            annotations = pd.read_csv(annotation_file)
            annotations = annotations.drop(columns=["index", "axis-1"])
            annotations.iloc[:, 0] = z
            all_annotations.append(annotations)
        annotations = pd.concat(all_annotations)

        # print(image_path)
        # print(prefix)
        # continue

        im = im[:, -1]
        v = napari.Viewer()
        v.title = prefix
        v.add_image(im)
        v.add_points(annotations)
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--show_2d", "-s", type=int, default=1)
    parser.add_argument("--names", type=int, nargs="+", default=None)
    args = parser.parse_args()

    input_ = args.input
    timepoint, cycle = input_.split("/")[-2:]
    names = args.names
    if names is None:
        names = os.listdir(input_)
        names.sort()

    has_slices = ["z" in name for name in names]
    if any(has_slices):
        assert all(has_slices)
        has_slices = True
    else:
        has_slices = False

    data_folder = f"/g/kreshuk/data/marioni/shila/{timepoint}/{cycle}"
    # data_folder = f"/g/kreshuk/data/marioni/shila/TimEmbryos-030320/{cycle}"
    assert os.path.exists(data_folder)
    if has_slices:
        show_annotations_with_slices(input_, data_folder, names)
    else:
        show_annotations(input_, data_folder, names, bool(args.show_2d))


if __name__ == "__main__":
    main()
