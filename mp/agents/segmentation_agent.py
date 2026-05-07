# ------------------------------------------------------------------------------
# A standard segmentation agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

import os
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
from mp.visualization.visualize_imgs import plot_3d_segmentation
from mp.eval.evaluate import ds_losses_metrics, ds_metrics


class SegmentationAgent(Agent):
    r"""An Agent for segmentation models."""

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        super().__init__(*args, **kwargs)
        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def get_outputs_old(self, inputs):
        outputs = self.model.forward_old(inputs)
        outputs = softmax(outputs).clamp(min=1e-08, max=1.0 - 1e-08)
        return outputs

    def track_validation_metrics(self, dataset_index, loss_f, datasets, save_path, epoch, acc):
        """
        Tracks validation metrics for the given dataset index.

        Args:
            dataset_index (int): Index of the dataset to evaluate.
            loss_f (function): Loss function used for evaluation.
            datasets (dict): Dictionary containing datasets with their names and data.
            save_path (str): Path to save the validation log.
            epoch (int): Current epoch number.
            acc (Accumulator): Accumulator object for storing evaluation results.

        Returns:
            float: Mean validation score for the specific dataset index, or None if not found.
        """
        current_index = 0  # Initialize a counter to track dataset indices
        for ds_name, ds in datasets.items():
            # Check if the current dataset is a validation set and matches the target index
            if ds_name[1] == "val" and current_index == dataset_index:
                # Calculate evaluation metrics for the validation set
                eval_dict = ds_metrics(ds, self, ["ScoreDice"])

                # Use enumerate to find the third entry (index 2) which corresponds to the foreground score
                for index, (key, value) in enumerate(eval_dict.items()):
                    if index == 2:  # If we reach the index corresponding to the foreground score
                        foreground_score = value["mean"]
                        with open(os.path.join(save_path, "val_log.txt"), "a+") as f:
                            f.write(f"Epoch {epoch + 1}: Validation score (mean): {foreground_score}\n")
                        return foreground_score

            # Update the dataset index if the current dataset is a test set
            elif ds_name[1] == "test":
                current_index += 1  # Increment the index to continue searching for the correct validation set
        # Return None if the specific validation dataset index is not found
        return None

    def multi_class_cross_entropy_no_softmax(self, prediction, target):
        r"""Stable Multiclass Cross Entropy with Softmax

        Args:
            prediction (torch.Tensor): network outputs w/ softmax
            target (torch.Tensor): label OHE

        Returns:
            (torch.Tensor) computed loss
        """
        return (-(target * torch.log(prediction)).sum(dim=1)).mean()

    def track_visualization(self, dataloader, save_path, epoch, config, phase="train"):
        r"""Creates visualizations and tracks them in tensorboard.

        Args:
            dataloader (Dataloader): dataloader to draw sample from
            save_path (string): path for the images to be saved (one folder up)
            epoch (int): current epoch
            config (dict): configuration dictionary from parsed arguments
            phase (string): either "test" or "train"
        """
        for imgs in dataloader:
            x_i, y_i = self.get_inputs_targets(imgs)
            x_i_seg = self.get_outputs(x_i)
            break

        # select sample with guaranteed segmentation label
        sample_idx = 0
        for i, y_ in enumerate(y_i):
            if len(torch.nonzero(y_)) > 0:
                sample_idx = i
                break
        x_i_img = x_i[sample_idx].unsqueeze(0)

        # segmentation
        x_i_seg = x_i_seg[sample_idx][1].unsqueeze(0).unsqueeze(0)
        threshold = 0.5
        x_i_seg_mask = (x_i_seg > threshold).int()
        y_i_seg_mask = y_i[sample_idx][1].unsqueeze(0).unsqueeze(0).int()

        save_path = os.path.join(save_path, "..", "imgs")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_pred = os.path.join(save_path, f"e_{epoch:06d}_{phase}_pred.png")
        save_path_label = os.path.join(save_path, f"e_{epoch:06d}_{phase}_label.png")

        plot_3d_segmentation(
            x_i_img,
            x_i_seg_mask,
            save_path=save_path_pred,
            img_size=(256, 256),
            alpha=0.5,
        )
        plot_3d_segmentation(
            x_i_img,
            y_i_seg_mask,
            save_path=save_path_label,
            img_size=(256, 256),
            alpha=0.5,
        )

        image = Image.open(save_path_pred)
        image = TF.to_tensor(image)
        self.writer_add_image(f"imgs_{phase}/pred", image, epoch)

        image = Image.open(save_path_label)
        image = TF.to_tensor(image)
        self.writer_add_image(f"imgs_{phase}/label", image, epoch)
