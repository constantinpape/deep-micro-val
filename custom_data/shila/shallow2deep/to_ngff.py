import argparse
import os
from glob import glob

import imageio
import ome_zarr
import zarr
from tqdm import tqdm


def convert_image_data(in_path, out_path):
    vol = imageio.volread(in_path)
    axes = tuple("zcyx")

    scaler = ome_zarr.scale.Scaler()
    mip = scaler.local_mean(vol)
    # TODO check the MIP
    loc = ome_zarr.io.parse_url(out_path, mode="w")
    group = zarr.group(loc.store)

    # TODO
    trafos = [
        [{"type": "scale", "scale": [1.]}]
    ]

    chunks = (1, 1, 512, 512)
    assert len(chunks) == len(axes)
    storage_opts = {"compression": "gzip", "chunks": chunks}
    ome_zarr.writer.write_multiscale(
        mip, group, axes=axes,
        coordinate_transformations=trafos, storage_options=storage_opts,
    )


# TODO
def convert_label_data():
    pass


# TODO need more metadata, like the pixel size
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embryo")
    args = parser.parse_args()
    embryo = args.embryo

    input_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/{embryo}"
    output_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/{embryo}"
    os.makedirs(output_folder, exist_ok=True)
    images = glob(os.path.join(input_folder, "*.ome.tif"))

    cell_segmentation_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/segmentation/{embryo}/cells"
    nucleus_segmentation_folder = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/segmentation/{embryo}/nuclei"

    for image in tqdm(images, desc=f"Convert images from {input_folder} to ngff"):
        name = os.path.basename(image)
        out_name = name.replace(".ome.tif", ".ome.zarr")
        out_path = os.path.join(output_folder, out_name)
        convert_image_data(image, out_path)

        cell_segmentation = os.path.join(cell_segmentation_folder, name)
        assert os.path.exists(cell_segmentation)
        # TODO
        # convert_label_data()

        nucleus_segmentation = os.path.join(nucleus_segmentation_folder, name)
        assert os.path.exists(nucleus_segmentation)
        # TODO
        # convert_label_data()
        return


# starting point:
# https://gist.github.com/constantinpape/69e3cb8e0401621365d814b4d6fda0bc
if __name__ == "__main__":
    main()
