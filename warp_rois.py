#!/usr/bin/env python3

"""
Transform nuclei ROIs from MNI space into participant-native T1 space.

REQUIREMENTS
------------
Install the required Python packages in your environment:

    pip install antspyx

The script expects:

1. A participant folder containing a T1 image:

    project/
    └── sub-01/
        └── T1.nii.gz

   Or a project folder containing several participant folders:

    project/
    ├── sub-01/
    │   └── T1.nii.gz
    ├── sub-02/
    │   └── T1.nii.gz
    └── sub-03/
        └── T1.nii.gz

2. A folder containing ROIs in MNI space:

    mni_nuclei/
    ├── left_nucleus.nii.gz
    └── right_nucleus.nii.gz

3. An MNI template image, for example:

    MNI152_T1_1mm.nii.gz


USAGE
-----
Save this script as:

    warp_rois.py


Process one participant:

    python warp_rois.py \
        /path/to/project/sub-01 \
        --rois /path/to/mni_nuclei \
        --template /path/to/MNI152_T1_1mm.nii.gz


Process one participant using a custom output name:

    python warp_rois.py \
        /path/to/project/sub-01 \
        --name participant_A \
        --rois /path/to/mni_nuclei \
        --template /path/to/MNI152_T1_1mm.nii.gz


Process every participant in a project folder:

    python warp_rois.py \
        /path/to/project \
        --rois /path/to/mni_nuclei \
        --template /path/to/MNI152_T1_1mm.nii.gz


Use a different T1 filename:

    python warp_rois.py \
        /path/to/project \
        --rois /path/to/mni_nuclei \
        --template /path/to/MNI152_T1_1mm.nii.gz \
        --t1-name anat_T1w.nii.gz


Show all command-line options:

    python warp_rois.py --help


OUTPUT
------
For each participant, the script creates:

    participant_folder/
    └── participant_nuclei/

Example:

    project/
    └── sub-01/
        ├── T1.nii.gz
        └── participant_nuclei/
            ├── sub-01_left_nucleus.nii.gz
            └── sub-01_right_nucleus.nii.gz

When --name is provided, that name is used as the output filename prefix.

NOTES
-----
- The input path can be either one participant folder or a whole project folder.
- A folder is recognised as a participant when it contains the selected T1 file.
- ROI masks are transformed using nearest-neighbour interpolation.

[!!!] Existing files with the same output names will be overwritten.

- Registration can take some amount of time for each participant.
"""

import argparse
from pathlib import Path

import ants


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Transform MNI-space nuclei ROIs into participant T1 space."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Participant folder or project folder containing participants.",
    )

    parser.add_argument(
        "--rois",
        type=Path,
        required=True,
        help="Folder containing MNI-space nuclei ROI files.",
    )

    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="MNI template file.",
    )

    parser.add_argument(
        "--name",
        help=(
            "Participant name used in output filenames. "
            "Only valid when processing one participant."
        ),
    )

    parser.add_argument(
        "--t1-name",
        default="T1.nii.gz",
        help="T1 filename inside each participant folder. Default: T1.nii.gz",
    )

    return parser.parse_args()


def find_participants(input_path, t1_name, custom_name=None):
    if not input_path.is_dir():
        raise NotADirectoryError(f"Folder not found: {input_path}")

    # Single-participant folder
    if (input_path / t1_name).is_file():
        participant_name = custom_name or input_path.name
        return [(input_path, participant_name)]

    # Whole project folder
    if custom_name:
        raise ValueError(
            "--name can only be used when input_path is one participant folder."
        )

    participants = [
        (folder, folder.name)
        for folder in sorted(input_path.iterdir())
        if folder.is_dir() and (folder / t1_name).is_file()
    ]

    if not participants:
        raise FileNotFoundError(
            f"No participant folders containing {t1_name} found in {input_path}"
        )

    return participants


def find_rois(roi_dir):
    if not roi_dir.is_dir():
        raise NotADirectoryError(f"ROI folder not found: {roi_dir}")

    rois = sorted(
        file
        for file in roi_dir.iterdir()
        if file.is_file()
        and (file.name.endswith(".nii") or file.name.endswith(".nii.gz"))
    )

    if not rois:
        raise FileNotFoundError(f"No NIfTI ROI files found in {roi_dir}")

    return rois


def process_participant(
    participant_path,
    participant_name,
    roi_files,
    mni_template,
    t1_name,
):
    print(f"\n--- Processing {participant_name} ---")

    t1_path = participant_path / t1_name
    output_dir = participant_path / "participant_nuclei"
    output_dir.mkdir(exist_ok=True)

    participant_t1 = ants.image_read(str(t1_path))

    registration = ants.registration(
        fixed=mni_template,
        moving=participant_t1,
        type_of_transform="SyN",
    )

    for roi_path in roi_files:
        print(f"Transforming {roi_path.name}...")

        mni_roi = ants.image_read(str(roi_path))

        native_roi = ants.apply_transforms(
            fixed=participant_t1,
            moving=mni_roi,
            transformlist=registration["invtransforms"],
            interpolator="nearestNeighbor",
        )

        output_path = output_dir / f"{participant_name}_{roi_path.name}"
        ants.image_write(native_roi, str(output_path))

        print(f"Saved: {output_path}")


def main():
    args = get_arguments()

    if not args.template.is_file():
        raise FileNotFoundError(f"MNI template not found: {args.template}")

    participants = find_participants(
        args.input_path,
        args.t1_name,
        args.name,
    )

    roi_files = find_rois(args.rois)
    mni_template = ants.image_read(str(args.template))

    print(f"Participants: {len(participants)}")
    print(f"Nuclei ROIs: {len(roi_files)}")

    for participant_path, participant_name in participants:
        try:
            process_participant(
                participant_path,
                participant_name,
                roi_files,
                mni_template,
                args.t1_name,
            )
        except Exception as error:
            print(f"Failed for {participant_name}: {error}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
