import os
from glob import glob

import h5py
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation, binary_closing
from skimage.measure import label
from skimage.segmentation import watershed


# Tmove seeds that are not inside 'fg_mask' to closest fg_mask point, remove the seeds in 'mask'
def move_seeds(points, fg_mask, mask):
    # distance_threshold = 12  # Thomas uses 7 as a distance but that sounds too low here
    distance_threshold = 8
    radius = 3

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
    distances = distances[seed_indices]
    fg_y_indices = fg_indices[0][seed_indices]
    fg_x_indices = fg_indices[1][seed_indices]
    assert len(fg_y_indices) == len(fg_x_indices) == len(bg_seeds)

    # NOTE could also mask the dropped seeds later and, in addition, mask out predictions for 'mask'
    # instead of just setting it to background
    dropped_seeds = []
    dropped_seed_map = np.zeros_like(seeds)

    for seed_id, dist, y, x in zip(bg_seeds, distances, fg_y_indices, fg_x_indices):
        if distance_threshold > 0 and dist > distance_threshold:
            dropped_seeds.append(seed_id)
            dropped_seed_map[y, x] = 1
            continue
        seeds[y, x] = seed_id
    print(len(dropped_seeds), "background seeds were dropped because they exceeded a distance of",
          distance_threshold, "to the foreground")
    n_seeds -= len(dropped_seeds)

    if dropped_seeds:
        dropped_seed_map = binary_dilation(dropped_seed_map, iterations=radius).astype("uint32")

    # make sure all seeds are in the foreground now
    seed_ids, seed_coords = np.unique(seeds, return_index=True)
    seed_ids, seed_coords = seed_ids[1:], seed_coords[1:]
    assert len(seed_ids) == n_seeds, f"{len(seed_ids)}, {n_seeds}"
    seed_indices = np.unravel_index(seed_coords, seeds.shape)
    values_at_seeds = fg_mask[seed_indices]
    assert all(values_at_seeds)

    # enlarge the seeds
    seed_mask = binary_dilation(seeds, iterations=radius)
    seeds = watershed(seed_mask.astype("float32"), seeds, mask=seed_mask)

    return seeds, dropped_seed_map


def make_segmentation(im, pred, mask, seeds, view=False):
    # NOTE the segmentation is stil a bit thick with this threshold; to further improve it we could try to shrink more
    # and then grow back to some gradient image
    thresh = 0.3

    fg_probs = 1. - pred[..., 2]
    boundaries = pred[..., 1]
    fg_probs -= boundaries

    assert fg_probs.shape == boundaries.shape == mask.shape

    fg_mask = fg_probs > thresh
    fg_mask = binary_closing(fg_mask, iterations=2)
    fg_mask[mask] = 0

    seeds, dropped_seeds = move_seeds(seeds, fg_mask, mask)
    seg = watershed(boundaries, seeds, mask=fg_mask)

    if view:
        import napari
        v = napari.Viewer()
        v.add_image(im)
        v.add_image(fg_mask, visible=False)
        v.add_image(boundaries, visible=False)
        v.add_labels(seeds)
        v.add_labels(dropped_seeds)
        v.add_labels(seg)
        napari.run()

    return seg


def make_training_data(in_path, out_path, view):
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

        seg = make_segmentation(c1, pred, mask, seeds, view=view)
        f_out.create_dataset("labels", data=seg, compression="gzip")


def main():
    view = False
    input_folder = "/home/pape/Work/data/deckers/lens-free/training_data/v2/prepared"
    output_folder = "/home/pape/Work/data/deckers/lens-free/training_data/v3/data"
    os.makedirs(output_folder, exist_ok=True)
    input_files = glob(os.path.join(input_folder, "*.h5"))
    for inp in input_files:
        print("Make training data for", inp)
        outp = os.path.join(output_folder, os.path.split(inp)[1])
        make_training_data(inp, outp, view=view)


if __name__ == "__main__":
    main()
