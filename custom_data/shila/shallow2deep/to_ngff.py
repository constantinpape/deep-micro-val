import argparse
import os
from glob import glob

import imageio
import ome_zarr
import ome_zarr.scale
import ome_zarr.writer
import zarr
from tqdm import tqdm


def convert_image_data(in_path, out_path):

    # load the input data from ome.tif
    vol = imageio.volread(in_path)
    # the data is stored as 'zcyx'. This is currently not allowed by ome.zarr, so we reorder to 'czyx'
    vol = vol.transpose((1, 0, 2, 3))

    # create scale pyramid
    # TODO how do we set options for the scaling?
    # (in this case the defaults are fine,
    # but it should be possible to over-ride this in general)
    scaler = ome_zarr.scale.Scaler()
    mip = scaler.local_mean(vol)

    # specify the axis and scale metadata
    axes_names = tuple("czyx")
    # TODO get this programatically / from data passed to the scaler
    is_scaled = {"c": False, "z": False, "y": True, "x": True}

    # TODO get resolution info and units from shila
    resolution = {"c": 1.0, "z": 1.0, "y": 1.0, "x": 1.0}
    units = [None, "pixel", "pixel", "pixel"]
    types = ["channel", "space", "space", "space"]

    trafos = [
        [{
            "scale": [resolution[ax] * 2**scale_level if is_scaled[ax] else resolution[ax] for ax in axes_names],
            "type": "scale"
        }]
        for scale_level in range(len(mip))
    ]

    axes = []
    for ax, type_, unit in zip(axes_names, types, units):
        axis = {"name": ax, "type": type_}
        if unit is not None:
            axis["unit"] = unit
        axes.append(axis)

    # provide additional storage options for zarr
    chunks = (1, 1, 512, 512)
    assert len(chunks) == len(axes_names)
    storage_opts = {"chunks": chunks}

    # write the data to ome.zarr
    loc = ome_zarr.io.parse_url(out_path, mode="w")
    group = zarr.group(loc.store)
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


# starting point:
# https://gist.github.com/constantinpape/69e3cb8e0401621365d814b4d6fda0bc
if __name__ == "__main__":
    main()
