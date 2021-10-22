import os
import imageio
import napari
import pandas as pd
from tqdm import tqdm


def _to_tif(name):
    pos = name[name.find("Pos"):name.find("Pos")+4]
    return f"MMStack_{pos}.ome.tif"


def main():
    annotation_folder = "./point_annotations"
    names = os.listdir(annotation_folder)
    data_folder = "/g/kreshuk/"
    for name in tqdm(names):
        tif_name = _to_tif(name)
        nuc = imageio.imread(os.path.join(data_folder, tif_name))
        annotations = pd.read_csv(os.path.join(annotation_folder, name)).iloc[:, 2:]
        v = napari.Viewer()
        v.add_image(nuc)
        v.add_points(annotations)
        napari.run()


if __name__ == "__main__":
    main()
