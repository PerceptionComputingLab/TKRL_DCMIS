# ------------------------------------------------------------------------------
# From an mp.data.datasets.dataset_segmentation.SegmentationDataset, create a
# mp.data.pytorch.PytorchDataset. There are different types of datasets:
#
# PytorchSeg2DDataset: the length of the dataset is the total number of slices
# (forth dimension) in the data base. A resized slice is returned by __getitem__
#
# PytorchSeg3DDataset: __getitem__ returnes the next instance, resized to the
# specified size. The length is the number of instances.
#
# Pytorch3DQueue: 3D patches are sampled randomly from the entire dataset.
# Receives a torchio.data.Sampler. Is built on top of torchio.data.Queue
# See https://torchio.readthedocs.io/data/patch_training.html
# ------------------------------------------------------------------------------

import copy
import torch
import torchio
import numpy as np
import random
from mp.data.pytorch.pytorch_dataset import PytorchDataset
import mp.data.pytorch.transformation as trans
import mp.eval.inference.predictor as pred
import time


class PytorchSegmnetationDataset(PytorchDataset):
    def __init__(
        self,
        dataset,
        ix_lst=None,
        size=None,
        norm_key="rescaling",
        aug_key="standard",
        channel_labels=False,
    ):
        r"""A torch.utils.data.Dataset for segmentation data.
        Args:
            dataset (SegmentationDataset): a SegmentationDataset
            ix_lst (list[int)]): list specifying the instances of the dataset.
                If 'None', all not in the hold-out dataset are incuded.
            size (tuple[int]): size as (channels, width, height, Opt(depth))
            norm_key (str): Normalization strategy, from
                mp.data.pytorch.transformation
            aug_key (str): Augmentation strategy, from
                mp.data.pytorch.transformation
            channel_labels (bool): if True, the output has one channel per label
        """
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size)
        self.norm = trans.NORMALIZATION_STRATEGIES[norm_key]
        self.aug = trans.AUGMENTATION_STRATEGIES[aug_key]
        self.nr_labels = dataset.nr_labels
        self.channel_labels = channel_labels
        self.predictor = None

    def get_instance(self, ix=None, name=None):
        r"""Get a particular instance from the ix or name"""
        assert ix is None or name is None
        if ix is None:
            instance = [ex for ex in self.instances if ex.name == name]
            assert len(instance) == 1
            return instance[0]
        else:
            return self.instances[ix]

    def get_ix_from_name(self, name):
        r"""Get ix from name"""
        return next(ix for ix, ex in enumerate(self.instances) if ex.name == name)

    def transform_subject(self, subject):
        r"""Transform a subject by applying normalization and augmentation ops"""
        if self.norm is not None:
            subject = self.norm(subject)
        if self.aug is not None:
            subject = self.aug(subject)
        return subject

    def get_subject_dataloader(self, subject_ix):
        r"""Get a list of input/target pairs equivalent to those if the dataset
        was only of subject with index subject_ix. For evaluation purposes.
        """
        raise NotImplementedError


class Subject:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class PytorchSeg2DDataset(PytorchSegmnetationDataset):
    r"""Divides images into 2D slices. If resize=True, the slices are resized to
    the specified size, otherwise they are center-cropped and padded if needed.
    """

    def __init__(
        self,
        dataset,
        ix_lst=None,
        size=(1, 256, 256),
        norm_key="rescaling",
        aug_key="standard",
        channel_labels=False,
        resize=False,
    ):
        if isinstance(size, int):
            size = (1, size, size)
        super().__init__(
            dataset=dataset,
            ix_lst=ix_lst,
            size=size,
            norm_key=norm_key,
            aug_key=aug_key,
            channel_labels=channel_labels,
        )
        assert len(self.size) == 3, "Size should be 2D"
        self.resize = resize
        self.target = dataset.target
        self.predictor = pred.Predictor2D(self.instances, size=self.size, norm=self.norm, resize=resize)

        self.idxs = []
        for instance_ix, instance in enumerate(self.instances):
            data = np.load(instance.file_path)
            label = data["label"]
            for slide_ix in range(label.shape[0]):
                self.idxs.append((instance_ix, slide_ix))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h)"""

        instance_idx, slice_idx = self.idxs[idx]

        instance = self.instances[instance_idx]

        data = np.load(instance.file_path)

        x = data["image"]
        y = data["label"]

        x = torch.from_numpy(x)[slice_idx].float().unsqueeze(0)
        y = torch.from_numpy(y)[slice_idx].float().unsqueeze(0)

        if self.channel_labels:
            y = trans.label_to_channel(y, self.target)

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(x[0], cmap="gray")
        # plt.subplot(1, 3, 2)
        # plt.imshow(y[0], cmap="gray")
        # plt.subplot(1, 3, 3)
        # plt.imshow(y[1], cmap="gray")
        # plt.savefig("test.png")
        # exit()

        return x, y

    def get_subject_dataloader(self, subject_ix):

        dl_items = []
        path = self.instances[subject_ix].file_path

        data = np.load(path)

        x = data["image"]
        y = data["label"]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        for slice_idx in range(x.shape[0]):
            x_slice = x[slice_idx].float().unsqueeze(0)
            y_slice = y[slice_idx].float().unsqueeze(0)
            if self.channel_labels:
                y_slice = trans.label_to_channel(y_slice, self.target)
            dl_items.append((x_slice.unsqueeze_(0), y_slice.unsqueeze_(0)))

        return dl_items
