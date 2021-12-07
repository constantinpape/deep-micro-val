import h5py
from torch_em.util import (add_weight_formats,
                           export_bioimageio_model,
                           get_default_citations,
                           export_parser_helper)


def _get_name(is_aff):
    name = "Lens-free"
    if is_aff:
        name += "-AffinityModel"
    else:
        name += "-BoundaryModel"
    return name


def _get_doc(is_aff_model):
    ndim = 2
    if is_aff_model:
        doc = f"""
## {ndim}D U-Net for Affinity Prediction

This model was trained on Lens-free images.
It predicts affinity maps and foreground probabilities for cell segmentation.
The affinities can be processed with the mutex watershed to obtain an instance segmentation.
        """
    else:
        doc = f"""
## {ndim}D U-Net for Boundary Prediction

This model was trained on the Lens-free images.
It predicts boundary maps and foreground probabilities for cell segmentation.
The boundaries can be processed with multicut segmentation to obtain an instance segmentation.
        """
    return doc


def export_to_bioimageio(checkpoint, output, input_, affs_to_bd, additional_formats):
    if input_ is None:
        input_data = None
    else:
        with h5py.File(input_, "r") as f:
            input_data = f["raw"][:, :512, :512]

    is_aff_model = True
    if is_aff_model and affs_to_bd:
        print("Export as boundary model")
        postprocessing = "affinities_with_foreground_to_boundaries2d"
        is_aff_model = False
    else:
        print("Export as affinity model")
        postprocessing = None

    name = _get_name(is_aff_model)
    tags = ["u-net", "cell-segmentation", "segmentation", "lens-free"]
    tags += ["affinity-prediction"] if is_aff_model else ["boundary-prediction"]

    # eventually we should refactor the citation logic
    cite = get_default_citations(
        model="UNet2d", model_output="affinities" if is_aff_model else "boundaries"
    )

    doc = _get_doc(is_aff_model)
    if is_aff_model:
        offsets = [
            [-1, 0], [0, -1],
            [-3, 0], [0, -3],
            [-9, 0], [0, -9]
        ]
        config = {"mws": {"offsets": offsets}}
    else:
        config = {}

    if additional_formats is None:
        additional_formats = []

    export_bioimageio_model(
        checkpoint, output,
        input_data=input_data,
        name=name,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        # need custom deepimagej fields if we have torchscript export
        for_deepimagej="torchscript" in additional_formats,
        config=config
    )
    add_weight_formats(output, additional_formats)


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output, args.input,
                         bool(args.affs_to_bd), args.additional_formats)
