# ------------------------------------------------------------------------------
# Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation
# Challenge (M&Ms) dataset.
# ------------------------------------------------------------------------------

import os
from mp.data.datasets.dataset_segmentation import (
    SegmentationDataset,
    SegmentationInstance,
)
from mp.paths import storage_data_path


class Cardiac(SegmentationDataset):
    r"""Class for the prostate segmentation."""

    def __init__(self, subset="", target=1):

        global_name = subset
        dataset_path = os.path.join(storage_data_path, global_name)

        # Fetch all patient/study names
        study_names = set(file_name.split(".")[0] for file_name in os.listdir(dataset_path))

        # Build instances
        instances = []
        for study_name in study_names:
            instances.append(
                SegmentationInstance(
                    file_path=os.path.join(dataset_path, study_name + ".npz"),
                    name=study_name,
                    group_id=None,
                )
            )

        self.label_names = ["background", "cardiac"]
        self.nr_labels = 2
        self.hold_out_ixs = []
        self.size = len(instances)
        self.classes = "0"
        self.instances = sorted(instances, key=lambda ex: ex.name)
        self.target = target
