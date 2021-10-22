import os
import imageio
import napari
import pandas as pd
from tqdm import tqdm


def _to_tif(name):
    pos = name[name.find("Pos"):].rstrip(".csv")
    return f"MMStack_{pos}.ome.tif"


def main(show_2d, names=None):
    annotation_folder = "./point_annotations"
    if names is None:
        names = os.listdir(annotation_folder)
        names.sort()

    data_folder = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29"
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


if __name__ == "__main__":
    wrong_annotations = [
        "TimEmbryos-020420_points_Pos10.csv",
        "TimEmbryos-020420_points_Pos11.csv",
        "TimEmbryos-020420_points_Pos12.csv",
        "TimEmbryos-020420_points_Pos13.csv",
        "TimEmbryos-020420_points_Pos15.csv",
        "TimEmbryos-020420_points_Pos14.csv",
    ]
    main(show_2d=True, names=wrong_annotations)
