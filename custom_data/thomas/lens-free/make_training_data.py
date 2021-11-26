import os
from glob import glob

import h5py
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label
from skimage.segmentation import watershed


# Tmove seeds that are not inside 'fg_mask' to closest fg_mask point, remove the seeds in 'mask'
def move_seeds(points, fg_mask, mask):
    seeds = np.zeros(fg_mask.shape, dtype="uint32")
    coords = np.round(points).astype("int")
    coords = tuple(coords[:, i] for i in range(coords.shape[1]))
    seeds[coords] = 1
    seeds[mask] = 0
    seeds = label(seeds)

    seed_ids, seed_coords = np.unique(seeds, return_index=True)
    seed_ids, seed_coords = seed_ids[1:], seed_coords[1:]
    n_seeds = len(seed_ids)
    seed_indices = np.unravel_index(seed_coords, seeds.shape)
    values_at_seeds = fg_mask[seed_indices]
    assert len(values_at_seeds) == len(seed_ids)
    bg_seeds = seed_ids[~values_at_seeds]
    seed_coords = seed_coords[~values_at_seeds]
    seed_indices = np.unravel_index(seed_coords, seeds.shape)
    seeds[seed_indices] = 0
    print("Number of seeds in the background:", len(bg_seeds), "/", len(seed_ids))

    print("Moving background seeds into the foreground")
    distances, fg_indices = distance_transform_edt(~fg_mask, return_indices=True)
    fg_y_indices = fg_indices[0][seed_indices]
    fg_x_indices = fg_indices[1][seed_indices]
    assert len(fg_y_indices) == len(fg_x_indices) == len(bg_seeds)

    # NOTE could also mask the dropped seeds later and, in addition, mask out predictions for 'mask'
    # instead of just setting it to background
    seeds_dropped = 0
    distance_threshold = 12  # Thomas uses 7 as a distance but that sounds too low here
    for seed_id, dist, y, x in zip(bg_seeds, dist, fg_y_indices, fg_x_indices):
        if dist > distance_threshold:
            continue
            seeds_dropped += 1
        seeds[y, x] = seed_id
    print(seeds_dropped, "background seeds were dropped because they exceeded a distance of",
          distance_threshold, "to the foreground")

    # make sure all seeds are in the foreground now
    seed_ids, seed_coords = np.unique(seeds, return_index=True)
    seed_ids, seed_coords = seed_ids[1:], seed_coords[1:]
    assert len(seed_ids) == n_seeds
    seed_indices = np.unravel_index(seed_coords, seeds.shape)
    values_at_seeds = fg_mask[seed_indices]
    assert all(values_at_seeds)

    # enlarge the seeds
    radius = 3
    seed_mask = binary_dilation(seeds, iterations=radius)
    seeds = watershed(seed_mask.astype("float32"), seeds, mask=seed_mask)

    return seeds


def make_segmentation(im, pred, mask, seeds, view=False):
    # NOTE segments may be a bit too large with this threshold
    # due to thick bounadry predictions covering background
    thresh = 0.5

    fg_probs = 1. - pred[..., 2]
    boundaries = pred[..., 1]

    assert fg_probs.shape == boundaries.shape == mask.shape

    fg_mask = fg_probs > thresh
    fg_mask[mask] = 0

    seeds = move_seeds(seeds, fg_mask, mask)
    seg = watershed(boundaries, seeds, mask=fg_mask)

    if view:
        import napari
        v = napari.Viewer()
        v.add_image(im)
        v.add_image(fg_mask, visible=False)
        v.add_image(boundaries, visible=False)
        v.add_labels(seeds)
        v.add_labels(seg)
        napari.run()

    return seg


def make_training_data(in_path, out_path):
    with h5py.File(in_path, "r") as f, h5py.File(out_path, "w") as f_out:
        c0, c1, c2 = f["c0"][:], f["c1"][:], f["c2"][:]
        f_out.create_dataset("channels/c0", data=c0, compression="gzip")
        f_out.create_dataset("channels/c1", data=c1, compression="gzip")
        f_out.create_dataset("channels/c2", data=c2, compression="gzip")

        raw = np.concatenate([c0[None], c1[None], c2[None]], axis=0)
        f_out.create_dataset("raw", data=raw, compression="gzip")

        pred = f["pred"][:]
        seeds = f["seeds"][:]
        mask = f["mask"][:].astype("bool")

        seg = make_segmentation(c1, pred, mask, seeds, view=True)
        f_out.create_dataset("labels", data=seg, compression="gzip")


def main():
    input_folder = "/home/pape/Work/data/thomas/training_data/prepared"
    output_folder = "/home/pape/Work/data/thomas/training_data/data"
    os.makedirs(output_folder, exist_ok=True)
    input_files = glob(os.path.join(input_folder, "*.h5"))
    for inp in input_files:
        outp = os.path.join(output_folder, os.path.split(inp)[1])
        make_training_data(inp, outp)


if __name__ == "__main__":
    main()
