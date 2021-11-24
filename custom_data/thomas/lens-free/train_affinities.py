import os
from glob import glob

import h5py
import torch
import torch_em
from torch_em.model import UNet2d

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9]
]


def get_loader(version, split, patch_shape, batch_size):
    raw_key = "raw"
    label_key = "labels"

    root = f"/g/kreshuk/pape/Work/data/deckers/lens-free/training_data/v{version}"
    paths = glob(os.path.join(root, "*.h5"))

    shapes = []
    for path in paths:
        with h5py.File(path, "r") as f:
            shape = f[raw_key].shape
        assert shape[0] == 3, f"Expect raw data with 3 channels, got {shape[0]}"
        shape = shape[1:]
        assert len(shape) == 2
        shapes.append(shape)

    if split == "train":
        n_samples = 1000
        rois = [
            (slice(None), slice(0, shape[1] - patch_shape[1])) for shape in shapes
        ]
    elif split == "val":
        n_samples = 25
        rois = [
            (slice(None), slice(shape[1] - patch_shape[1], shape[1])) for shape in shapes
        ]
    else:
        raise ValueError(f"Invalid data split: {split}")

    assert len(rois) == len(paths)
    # print(split)
    # print(rois)

    label_transform = torch_em.transform.label.AffinityTransform(offsets=OFFSETS,
                                                                 add_binary_target=True,
                                                                 add_mask=True)
    # TODO independent per channel normalization ?!
    return torch_em.default_segmentation_loader(
        paths, raw_key, paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        ndim=2, is_seg_dataset=True,
        label_transform2=label_transform,
        rois=rois, n_samples=n_samples,
        with_channels=True, shuffle=True,
    )


def train_affinities(args):
    n_out = len(OFFSETS) + 1
    model = UNet2d(in_channels=3, out_channels=n_out, initial_features=64,
                   final_activation="Sigmoid")

    patch_shape = (512, 512)

    train_loader = get_loader(args.version, "train", patch_shape, args.batch_size)
    val_loader = get_loader(args.version, "val", patch_shape, args.batch_size)
    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )

    trainer = torch_em.default_segmentation_trainer(
        name=f"lensfree-affinity-model-v{args.version}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=args.n_iterations)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(default_batch_size=8)
    parser.add_argument("--version", "-v", default=1, type=int)
    args = parser.parse_args()
    train_affinities(args)
