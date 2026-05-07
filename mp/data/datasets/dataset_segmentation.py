# ------------------------------------------------------------------------------
# All datasets descend from this SegmentationDataset class storing segmentation
# instances.
# ------------------------------------------------------------------------------

import os
import sys

import numpy as np

from mp.data.datasets.dataset import Dataset, Instance
import mp.data.datasets.dataset_utils as du
import torchio
import torch


class SegmentationInstance:
    def __init__(self, file_path, name=None, group_id=None, class_ix=0):

        assert isinstance(file_path, str)
        self.file_path = file_path
        self.name = name
        self.group_id = group_id
        self.class_ix = class_ix


class SegmentationDataset(Dataset):
    r"""A Dataset for segmentation tasks, that specific datasets descend from."""

    def __init__(
        self,
        instances,
        name,
        label_names=None,
        nr_channels=1,
        modality="unknown",
        hold_out_ixs=[],
    ):
        # Set mean input shape and mask labels, if these are not provided
        print("\nDATASET: {} with {} instances".format(name, len(instances)))

        self.label_names = label_names
        self.nr_labels = len(label_names)
        self.nr_channels = nr_channels
        self.modality = modality
        super().__init__(
            name=name,
            instances=instances,
            hold_out_ixs=hold_out_ixs,
        )
