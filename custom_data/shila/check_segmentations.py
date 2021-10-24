import os
import napari
import pandas as pd
import imageio
from tqdm import tqdm


def _to_tif(name):
    pos = name[name.find("Pos"):].rstrip(".csv")
    return f"MMStack_{pos}.ome.tif"


def main():
    data_folder = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29"
    segs_mws = "/g/kreshuk/data/marioni/shila/segmentation/mutex_watershed"
    segs_ws = "/g/kreshuk/data/marioni/shila/segmentation/watershed"
    annotation_folder = "./point_annotations"

    names = os.listdir(annotation_folder)
    names.sort()
    for name in tqdm(names):
        annotations = pd.read_csv(os.path.join(annotation_folder, name)).iloc[:, 1:]
        tif_name = _to_tif(name)
        cz = annotations.iloc[0, :2]
        annotations = annotations.iloc[:, 2:]
        im = imageio.volread(os.path.join(data_folder, tif_name))[int(cz[0]), int(cz[1])]
        seg_name = name.replace(".csv", ".ome.tif")
        mws = imageio.volread(os.path.join(segs_mws, seg_name))[int(cz[0]), int(cz[1])]
        ws = imageio.volread(os.path.join(segs_ws, seg_name))[int(cz[0]), int(cz[1])]

        v = napari.Viewer()
        v.title = name
        v.add_image(im)
        v.add_labels(mws)
        v.add_labels(ws)
        v.add_points(annotations)
        napari.run()


if __name__ == "__main__":
    main()
