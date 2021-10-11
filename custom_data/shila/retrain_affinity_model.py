import argparse
import os
import sys
from glob import glob


def retrain_for_nuclei(version, pretrained, batch_size, n_iterations):
    sys.path.append("../..")
    from segmentation.training.default_models import get_nucleus_trainer

    label_paths = glob("./gt_generation/data/ground_truth/v1/*.ome.tif")
    fnames = [os.path.split(lp)[1] for lp in label_paths]
    raw_paths = [os.path.join("./gt_generation/data/nuclei", fname) for fname in fnames]

    name = f"nuclei_v{version}"
    trainer = get_nucleus_trainer(name,
                                  raw_paths, label_paths,
                                  raw_paths, label_paths,
                                  batch_size=batch_size, patch_shape=(512, 512))
    trainer.fit(n_iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=int, required=True)
    parser.add_argument("-p", "--pretrained", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-n", "--n_iterations", type=int, default=10000)
    args = parser.parse_args()
    retrain_for_nuclei(args.version, bool(args.pretrained), args.batch_size, args.n_iterations)


if __name__ == "__main__":
    main()
