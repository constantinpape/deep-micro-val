import imageio
import h5py
import numpy as np
from elf.evaluation import (symmetric_best_dice_score, variation_of_information,
                            matching, mean_average_precision)


# TODO also save rand index
def eval_image(gt, seg):
    assert seg.shape == gt.shape
    sbd = symmetric_best_dice_score(seg, gt)
    vi_split, vi_merge = variation_of_information(seg, gt)

    iou50 = matching(seg, gt, threshold=0.5)["precision"]
    iou75 = matching(seg, gt, threshold=0.75)["precision"]
    iou90 = matching(seg, gt, threshold=0.9)["precision"]
    m_ap = mean_average_precision(seg, gt)

    return {"sbd": sbd, "vi-merge": vi_merge, "vi-split": vi_split,
            "iou50": iou50, "iou75": iou75, "iou90": iou90, "map": m_ap}


def _load_gt(gt_path, gt_key):
    if gt_key is None:
        gt = imageio.imread(gt_path)
    else:
        with h5py.File(gt_path, "r") as f:
            gt = f[gt_key][:]
    return gt


# TODO tqdm
# TODO use pandas to save the results to the save path instead
# of just looking at sbd and map
def evaluate_all(gt_files, seg_files, save_path, gt_key=None):
    maps = {}
    sbds = {}
    for gt_path, seg_path in zip(gt_files, seg_files):
        gt = _load_gt(gt_path, gt_key)
        with h5py.File(seg_path, "r") as f:
            g = f["segmentations"]
            for name, node in g.items():
                seg = node[:]
                scores = eval_image(gt, seg)

                if name in maps:
                    maps[name].append(scores["map"])
                else:
                    maps[name] = [scores["map"]]

                if name in sbds:
                    sbds[name].append(scores["sbd"])
                else:
                    sbds[name] = [scores["sbd"]]

    print("MAP-scores:")
    for name, scores in maps.items():
        print(name, ":", np.mean(scores))
    print("SBD-scores:")
    for name, scores in sbds.items():
        print(name, ":", np.mean(scores))
