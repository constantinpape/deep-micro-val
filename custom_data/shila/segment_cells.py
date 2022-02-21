import argparse
import os

import h5py
import imageio
import napari
import numpy as np
import torch
from skimage.segmentation import watershed
from torch_em.util import get_trainer
from tqdm import tqdm
# from ilastik.experimental.api import from_project_file
# from xarray import DataArray


def segment_image(input_path, seed_path, ilp, model):
    image = imageio.volread(input_path)

    print("Run prediction with ilastik ...")
    # FIXME bug in ilastik, for now we use the hard-coded prediction
    # image = DataArray(image, dims=tuple("zcyx"))
    # pred = ilp.predict(image)
    pred_path = "/g/kreshuk/data/marioni/shila/TimEmbryos-020420/data-data_Probabilities.h5"
    with h5py.File(pred_path, "r") as f:
        pred = f["exported_data"][:]
    pred = (pred[:, 1] - pred[:, 0]).astype("float32")
    print(pred.shape)

    seeds = imageio.volread(seed_path)

    bd_tmp = "./bd_tmp.h5"
    if os.path.exists(bd_tmp):
        with h5py.File(bd_tmp, "r") as f:
            boundaries = f["data"][:]
    else:
        print("Predict enhancer ...")
        boundaries = np.zeros(seeds.shape, dtype="float32")
        model = get_trainer(model, device="cuda").model
        model.eval()
        with torch.no_grad():
            for z in range(boundaries.shape[0]):
                input_ = pred[z][None, None]
                input_ = torch.from_numpy(input_).to("cuda")
                bd = model(input_)
                boundaries[z] = bd.detach().cpu().numpy().squeeze()
        with h5py.File(bd_tmp, "a") as f:
            f.create_dataset("data", data=boundaries, compression="gzip")

    print("Run foreground segmentation ...")
    # segment foreground / background
    fg_seed = (seeds > 0).astype("uint8")
    bg_seed = np.ones(fg_seed.shape, dtype="bool")
    bg_seed[:, 5:-5, 5:-5] = 0
    bg_seed[fg_seed == 1] = 0
    fg_seed[bg_seed] = 2
    fg_mask = np.zeros_like(seeds, dtype="bool")
    for z in range(fg_mask.shape[0]):
        fg_mask[z] = (watershed(boundaries[z], markers=fg_seed[z]) == 1)

    print("Run segmentation ...")
    # segment cells
    cell_seg = np.zeros_like(seeds, dtype="uint16")
    for z in range(cell_seg.shape[0]):
        cell_seg[z] = watershed(boundaries[z], markers=seeds[z], mask=fg_mask[z])

    v = napari.Viewer()
    v.add_image(image[:, -1])
    v.add_image(pred)
    v.add_image(boundaries)
    v.add_labels(seeds)
    v.add_labels(fg_mask.astype("uint8"))
    v.add_labels(cell_seg)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-n", "--nucleus_seg_folder", required=True)
    args = parser.parse_args()

    # TODO export to modelzoo format
    model = "/g/kreshuk/pape/Work/my_projects/torch-em/experiments/shallow2deep/cells/checkpoints/covid-if-cells"

    # ilp = from_project_file("./shila-boundaries.ilp")
    ilp = None

    names = os.listdir(args.nucleus_seg_folder)
    for name in tqdm(names):
        input_path = os.path.join(args.input, name)
        assert os.path.exists(input_path)
        seed_path = os.path.join(args.nucleus_seg_folder, name)
        segment_image(input_path, seed_path, ilp, model)
        break


if __name__ == "__main__":
    main()
