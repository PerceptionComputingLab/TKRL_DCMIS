import os
import time
import torch
from tqdm import tqdm

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator


class KDAgent(SegmentationAgent):
    """Extension of Segmentation Agent to support Knowledge Distillation
    as proposed in "Incremental learning techniques for semantic segmentation" by Michieli, U., Zanuttigh, P., 2019.
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

    def perform_training_epoch(self, loss_f, train_dataloader, config, epoch, print_run_loss=False, save_path=""):
        """Performs a single training epoch.

        Args:
            loss_f (callable): Loss function for the segmenter.
            train_dataloader (DataLoader): Dataloader for training data.
            config (dict): Configuration dictionary.
            epoch (int): Current epoch number.
            print_run_loss (bool, optional): Whether to print running loss. Defaults to False.
            save_path (str, optional): Path to save training logs. Defaults to "".

        Returns:
            Accumulator: Accumulator holding losses.
        """
        acc = Accumulator("loss")
        start_time = time.time()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", dynamic_ncols=True)
        for data in pbar:
            inputs, targets = self.get_inputs_targets(data)
            outputs = self.get_outputs(inputs)
            self.model.backbone_optim.zero_grad()

            loss_seg = loss_f(outputs, targets)
            loss_distill = self.calculate_distillation_loss(inputs, outputs)

            loss = loss_seg + config["lambda_d"] * loss_distill
            loss.backward()
            self.model.backbone_optim.step()

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
        """Calculates the distillation loss based on the outputs from the current and previous models.

        Args:
            inputs (torch.Tensor): Input data for the model.
            outputs (torch.Tensor): Current model outputs.

        Returns:
            torch.Tensor: Calculated distillation loss.
        """
        if self.model.backbone_old is not None:
            outputs_old = self.get_outputs_old(inputs)
            return self.multi_class_cross_entropy_no_softmax(outputs, outputs_old)
        else:
            return torch.zeros(1, device=outputs.device)

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
