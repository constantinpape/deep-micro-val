import os
from pathlib import Path

import torch
import torch_em
from bioimageio.core import load_raw_resource_description
from torch_em.model import UNet2d

from ..experiment import require_affinity_nucleus_model


def get_unet(pretrained_model, offsets, **model_kwargs):
    if pretrained_model is None:
        out_channels = 2 if offsets is None else len(offsets) + 1
        model = UNet2d(in_channels=1, out_channels=out_channels, **model_kwargs)
    else:
        os.makedirs("./checkpoints/pretrained", exist_ok=True)
        model_path = Path("./checkpoints/pretrained/dsb_affinities.zip")
        if not model_path.exists():
            model_path = require_affinity_nucleus_model(model_path, pretrained_model)
        model_description = load_raw_resource_description(model_path)
        model = UNet2d(**model_description.kwargs)
        weight_path = os.path.join(
            model_description.root_path,
            model_description.weights["pytorch_state_dict"].source.path
        )
        model.load_state_dict(torch.load(weight_path))
        offsets = model_description.config["mws"]["offsets"]
    return model, offsets


def get_nucleus_trainer(
    name,
    raw_train_paths, label_train_paths,
    raw_val_paths, label_val_paths,
    batch_size, patch_shape,
    device=None, offsets=None,
    pretrained_model="dsb", include_training_data=None,
    **model_kwargs
):

    if include_training_data is not None:
        raise NotImplementedError

    model, offsets = get_unet(pretrained_model, offsets, **model_kwargs)

    if offsets is None:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        label_transform2 = None
    else:
        label_transform = None
        label_transform2 = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                      add_binary_target=True,
                                                                      add_mask=True)

    train_loader = torch_em.default_segmentation_loader(
        raw_train_paths, None, label_train_paths, None,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, label_transform2=label_transform2,
        is_seg_dataset=False, n_samples=100
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_val_paths, None, label_val_paths, None,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, label_transform2=label_transform2,
        is_seg_dataset=False, n_samples=5
    )

    loss = torch_em.loss.LossWrapper(
        torch_em.loss.DiceLoss(),
        transform=torch_em.loss.ApplyAndRemoveMask()
    )
    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader,
        device=device, loss=loss, metric=loss
    )
    return trainer


# TODO
# def export_model(): pass
