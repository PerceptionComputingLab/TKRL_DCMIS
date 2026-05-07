import os
import time
from tqdm import tqdm
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator

import torch


class MASAgent(SegmentationAgent):
    r"""Extension of Segmentation Agent to support Memory Aware Synapses for brain segmentation
    as proposed in "Importance Driven Continual Learning for Segmentation Across Domains" by Oezguen et al., 2020.
    """

    def __init__(self, *args, **kwargs):
        if "metrics" not in kwargs:
            kwargs["metrics"] = ["ScoreDice", "ScoreIoU", "ScoreHausdorff"]
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

        start_time = time.time()
        for epoch in range(init_epoch, nr_epochs):
            print("Epoch:", epoch + 1)
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

                # Log losses and visualizations to TensorBoard
                self.track_loss(acc, epoch + 1, config)
                self.track_visualization(train_dataloader, save_path, epoch + 1, config, "train")
                self.track_visualization(test_dataloader, save_path, epoch + 1, config, "test")

                end_time = time.time()
                print(f"Time per epoch: {round((end_time - start_time) / (epoch + 1), 4)} seconds")

        if val_best:
            self.restore_state(exp_path, self.best_validation_epoch + 1)
            with open(os.path.join(exp_path, "val_track.txt"), "a+") as f:
                f.write(f"{self.best_validation_epoch + 1}\n")
            print(f"Best epoch: {self.best_validation_epoch + 1}; Best validation Dice: {self.best_validation_value}")

        with torch.no_grad():
            self.model.backbone_new.eval()
            self.track_metrics(nr_epochs, results, loss_f, eval_datasets)

        new_importance_weights = self.calc_importance_weights(train_dataloader, loss_f)
        self.model.update_importance_weights(new_importance_weights)
        self.model.finish()

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
            outputs = self.get_outputs(inputs)
            self.model.backbone_optim.zero_grad()

            loss_seg = loss_f(outputs, targets)
            loss_mas = self.calculate_mas_loss(loss_seg)

            loss = loss_seg + config["lambda_d"] * loss_mas
            loss.backward()
            self.model.backbone_optim.step()

            acc.add("loss", loss.item(), count=len(inputs))
            acc.add("loss_seg", loss_seg.item(), count=len(inputs))
            acc.add("loss_mas", loss_mas.item(), count=len(inputs))

            # Update tqdm with the latest loss values
            pbar.set_postfix(
                {
                    "Total Loss": f"{loss.item():.4g}",
                    "Segmentation Loss": f"{loss_seg.item():.4g}",
                    "MAS Loss": f"{loss_mas.item():.4g}",
                }
            )

        # Print running loss
        if print_run_loss:
            print(
                "Running loss: {:.4f}, MAS loss: {:.4f}, Time/epoch: {:.4f} seconds".format(
                    acc.mean("loss"), acc.mean("loss_mas"), round(time.time() - start_time, 4)
                )
            )
            with open(os.path.join(save_path, "train_track.txt"), "a+") as f:
                f.write(
                    "Epoch {}: Running loss: {:.4f}, MAS loss: {:.4f}, Time/epoch: {:.4f} seconds\n".format(
                        self.agent_state_dict["epoch"],
                        acc.mean("loss"),
                        acc.mean("loss_mas"),
                        round(time.time() - start_time, 4),
                    )
                )

        return acc

    def calculate_mas_loss(self, loss_seg):
        """
        Calculates the MAS (Memory Aware Synapses) loss to penalize deviations from old knowledge.

        Args:
            loss_seg (torch.Tensor): Segmentation loss.

        Returns:
            torch.Tensor: The calculated MAS loss.
        """
        loss_mas = torch.zeros(1).to(loss_seg.device) if loss_seg.is_cuda else torch.zeros(1)

        if self.model.importance_weights is not None:
            for param_old, param_new, weights in zip(
                self.model.backbone_old.parameters(),
                self.model.backbone_new.parameters(),
                self.model.importance_weights,
            ):
                if param_new.requires_grad:
                    loss_mas += torch.sum(weights * (param_new - param_old) ** 2)  # / self.model.n_params_backbone

        return loss_mas

    def track_loss(self, acc, epoch, config, phase="train"):
        r"""Tracks loss in tensorboard.

        Args:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
            epoch (int): current epoch
            config (dict): configuration dictionary from parsed arguments
            phase (string): either "test" or "train"
        """

        self.writer_add_scalar(f"loss_{phase}/loss_seg", acc.mean("loss_seg"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_mas", acc.mean("loss_mas"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_comb", acc.mean("loss"), epoch)

    def calc_importance_weights(self, dataloader, loss_f):
        """
        Compute importance weights for MAS (Memory Aware Synapses)

        Args:
            dataloader (Dataloader): training dataloader

        Returns:
            param_importance (List[torch.Tensor]): list of tensors representing parameter importance
        """
        # Initialize importance weights for each parameter as zeros
        param_importance = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]

        max_grad = float("-inf")
        min_grad = float("inf")

        for data in tqdm(dataloader, desc="Calculating MAS Importance Weights", dynamic_ncols=True):
            inputs, targets = self.get_inputs_targets(data)
            outputs = self.get_outputs(inputs)

            # Compute loss and gradients
            loss = loss_f(outputs, targets)
            self.model.backbone_new.zero_grad()
            loss.backward()

            # Accumulate gradients to compute importance weights
            for i, param in enumerate(self.model.backbone_new.parameters()):
                if param.grad is not None:
                    param_importance[i] += param.grad.abs() / len(dataloader)
                    max_grad = max(max_grad, param.grad.max().item())
                    min_grad = min(min_grad, param.grad.min().item())

        # Normalizing gradients to [0, 1] range for fair comparison
        for i in range(len(param_importance)):
            if max_grad != min_grad:  # Ensure there's no division by zero
                param_importance[i] = (param_importance[i] - min_grad) / (max_grad - min_grad)

        return param_importance
