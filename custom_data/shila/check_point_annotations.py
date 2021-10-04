import imageio
import napari
import pandas as pd


def check_annotations():
    points = pd.read_csv("./data/TimEmbryos-020420_points_Pos0.csv").values[:, 1:]

    path = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29/MMStack_Pos0.ome.tif"
    im = imageio.volread(path)
    print(im.shape)

    v = napari.Viewer()
    v.add_image(im)
    v.add_points(points)
    napari.run()


def export_annotations():
    path = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/HybCycle_29/MMStack_Pos0.ome.tif"
    im = imageio.volread(path)
    im = im[1, 3]
    imageio.imsave("./data/nuc.tif", im)


if __name__ == "__main__":
    # check_annotations()
    export_annotations()
