import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax


def total_variation_loss(x):
    batch_size, channels, height, width = x.size()
    tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * channels * height * width)


class PCDAgent(SegmentationAgent):
    def __init__(self, *args, **kwargs):
        """
        Initializes the TKRLAgent with default settings and parameters.
        If no metrics are specified in the arguments, sets the default metrics.
        """
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
        # Initialize mask tokens for various head dimensions
        self.probes_noise = None
        self.boundary = 0.99
        super().__init__(*args, **kwargs)

    def train(
        self,
        results,
        loss_f,
        train_dataloader,
        test_dataloader,
        config,
        init_epoch=0,
        nr_epochs=100,
        eval_datasets=dict(),
        save_path="",
        dataset_index=0,
        exp_path="",
    ):
        """
        Trains the model using the specified dataloaders, tracking metrics, and saving model states.
        """
        val_best = config["val_best"]
        self.agent_state_dict["epoch"] = init_epoch

        self.best_validation_value = 0.0
        self.best_validation_epoch = 0

        if self.probes_noise is None:
            print("Creating probes noise ...")
            self.probes_noise = (
                torch.randn(
                    int(config["batch_size"] * config["multiply_probes"]),
                    config["input_dim_channels"],
                    config["input_dim_size"],
                    config["input_dim_size"],
                    dtype=torch.float,
                )
                * 0.1
            )
        self.boundary = config["boundary"]

        start_time = time.time()

        # Handle probes noise for continual learning
        if config["continual"]:
            print("Inference probes noise")
            self.inference_probes_noise(config, save_path, dataset_index)

        for epoch in range(init_epoch, nr_epochs):
            print(f"Epoch: {epoch + 1}")
            self.agent_state_dict["epoch"] = epoch
            self.model.backbone_new.train()
            acc = self.perform_training_epoch(
                loss_f, train_dataloader, config, epoch, print_run_loss=self.verbose, save_path=save_path
            )
            with torch.no_grad():
                if val_best:
                    self.model.backbone_new.eval()
                    dice = self.track_validation_metrics(dataset_index, loss_f, eval_datasets, save_path, epoch, acc)
                    print(f"Validation Dice Score: {dice}")
                    if dice > self.best_validation_value:
                        self.best_validation_value = dice
                        self.best_validation_epoch = epoch
                        self.save_state(save_path, epoch + 1)

                self.model.backbone_scheduler.step()

                # Log losses and visualizations to tensorboard
                self.track_loss(acc, epoch + 1, config)
                self.track_visualization(train_dataloader, save_path, epoch + 1, config, "train")
                self.track_visualization(test_dataloader, save_path, epoch + 1, config, "test")

                end_time = time.time()
                print(f"Time per epoch: {round((end_time - start_time) / (epoch + 1), 4)} seconds")

        if val_best:
            self.restore_state(exp_path, self.best_validation_epoch + 1)
            with open(os.path.join(exp_path, "val_track.txt"), "a+") as file:
                file.write(f"{self.best_validation_epoch + 1}\n")
            print(f"Best epoch: {self.best_validation_epoch + 1}; Best validation Dice: {self.best_validation_value}")

        with torch.no_grad():
            self.model.backbone_new.eval()
            self.track_metrics(nr_epochs, results, loss_f, eval_datasets)

        self.model.finish()

    def inference_probes_noise(self, config, save_path, dataset_index):
        """
        Performs inference on probes noise to optimize the noise representation.
        """
        self.probes_noise = self.probes_noise.to(config["device"])
        self.probes_noise.requires_grad = True

        optimizer = torch.optim.Adam([self.probes_noise], lr=1e-3)  # 调整学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        pbar = tqdm(range(10000), desc="Optimizing Probes Noise")
        for step in pbar:
            optimizer.zero_grad()
            logits = self.model.backbone_old(self.probes_noise)
            probs = F.softmax(logits, dim=1)
            output = probs[:, 1]

            loss = -torch.mean(output) + total_variation_loss(self.probes_noise)

            loss.backward()

            torch.nn.utils.clip_grad_norm_([self.probes_noise], max_norm=1.0)

            optimizer.step()

            scheduler.step(loss.item())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            if loss.item() < -self.boundary:
                break

        for i in range(config["batch_size"]):
            save_image(
                self.probes_noise[i].cpu(), os.path.join(save_path, f"{dataset_index - 1}_{i}_probes_noise.png")
            )
        avg_noise = torch.mean(self.probes_noise, dim=0)
        save_image(avg_noise.cpu(), os.path.join(save_path, f"{dataset_index - 1}_avg_probes_noise.png"))

    def perform_training_epoch(self, loss_f, train_dataloader, config, epoch, print_run_loss=False, save_path=""):
        """
        Performs a single training epoch and tracks relevant metrics.

        Args:
            loss_f: Loss function.
            train_dataloader: Dataloader for training data.
            config: Configuration dictionary.
            epoch: Current epoch number.
            print_run_loss: Boolean indicating whether to print running loss.
            save_path: Path to save the training logs.
        """
        acc = Accumulator("loss")
        start_time = time.time()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", dynamic_ncols=True)
        for data in pbar:
            inputs, targets = self.get_inputs_targets(data)
            self.inputs_ = inputs
            outputs = self.get_outputs(inputs)
            self.model.backbone_optim.zero_grad()
            loss_seg = loss_f(outputs, targets)

            loss_distill = self.calculate_distillation_loss(inputs, outputs)

            loss = loss_seg + config["lambda_d"] * loss_distill
            loss.backward()
            self.model.backbone_optim.step()

            # Accumulate the losses for logging
            acc.add("loss", loss.item(), len(inputs))
            acc.add("loss_seg", loss_seg.item(), len(inputs))
            acc.add("loss_distill", loss_distill.item(), len(inputs))

            # Update tqdm with the latest loss values
            pbar.set_postfix(
                {
                    "Total Loss": f"{loss.item():.4f}",
                    "Segmentation Loss": f"{loss_seg.item():.4f}",
                    "Distillation Loss": f"{loss_distill.item():.4f}",
                }
            )

        pbar.close()

        if print_run_loss:
            print(
                "Running loss: {:.4f}, Distillation loss: {:.4f}, Time/epoch: {:.4f} seconds".format(
                    acc.mean("loss"), acc.mean("loss_distill"), round(time.time() - start_time, 4)
                )
            )
            with open(os.path.join(save_path, "train_track.txt"), "a+") as f:
                f.write(
                    "Epoch {}: Running loss: {:.4f}, Distillation loss: {:.4f}, Time/epoch: {:.4f} seconds\n".format(
                        self.agent_state_dict["epoch"],
                        acc.mean("loss"),
                        acc.mean("loss_distill"),
                        round(time.time() - start_time, 4),
                    )
                )

        return acc

    def calculate_distillation_loss(self, inputs, outputs):
        """
        Calculates the knowledge distillation loss based on old and new model predictions.

        Args:
            inputs: Input data for the model.
            loss_seg: Segmentation loss.

        Returns:
            torch.Tensor: The calculated distillation loss.
        """
        if self.model.backbone_old:
            # Generate outputs for the current inputs and probes noise
            outputs_noise = self.get_outputs(self.probes_noise)
            ones = torch.ones(
                self.probes_noise.shape[0], 1, self.probes_noise.shape[2], self.probes_noise.shape[3]
            ).to(self.probes_noise.device)
            zeros = torch.zeros_like(ones)
            outputs_noise_old = torch.concat([zeros, ones], axis=1)

            outputs_old = self.get_outputs_old(inputs)

            # Calculate knowledge distillation loss
            kd_loss = self.multi_class_cross_entropy_no_softmax(outputs, outputs_old.detach())
            +self.multi_class_cross_entropy_no_softmax(outputs_noise, outputs_noise_old)

            return kd_loss
        else:
            # Return a zero tensor if no distillation is performed
            return torch.zeros(1).to(inputs.device if inputs.is_cuda else "cpu")

    def track_loss(self, acc, epoch, config, phase="train"):
        r"""Tracks loss in tensorboard.

        Args:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
            epoch (int): current epoch
            config (dict): configuration dictionary from parsed arguments
            phase (string): either "test" or "train"
        """
        self.writer_add_scalar(f"loss_{phase}/loss_seg", acc.mean("loss_seg"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_distill", acc.mean("loss_distill"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_comb", acc.mean("loss"), epoch)
