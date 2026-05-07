import os
import time
import torch
from tqdm import tqdm

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator


class EWCAgent(SegmentationAgent):
    """
    Elastic Weight Consolidation (EWC) Agent for training segmentation models.
    Extends the base SegmentationAgent with EWC-specific methods.
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
            loss_ewc = self.calculate_ewc_loss(loss_seg)

            loss = loss_seg + config["lambda_d"] * loss_ewc
            loss.backward()
            self.model.backbone_optim.step()

            # Update tqdm with the latest loss values
            pbar.set_postfix({"Loss": f"{loss.item():.4g}"})

            acc.add("loss", loss.item(), len(inputs))
            acc.add("loss_seg", loss_seg.item(), len(inputs))
            acc.add("loss_ewc", loss_ewc.item(), len(inputs))

            # Update tqdm with the latest loss values
            pbar.set_postfix(
                {
                    "Total Loss": f"{loss.item():.4g}",
                    "Segmentation Loss": f"{loss_seg.item():.4g}",
                    "EWC Loss": f"{loss_ewc.item():.4g}",
                }
            )

        if print_run_loss:
            print(
                f"Running loss: {acc.mean('loss')} - EWC loss: {acc.mean('loss_ewc')} - Time/epoch: {round(time.time() - start_time, 4)} seconds"
            )

            # Write the loss tracking information to the file with a timestamp
            with open(os.path.join(save_path, "train_track.txt"), "a+") as f:
                f.write(
                    "[{}] Epoch {}: running loss: {:.4g}, ewc loss: {:.4g} - time/epoch {:.4g} seconds\n".format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.agent_state_dict["epoch"],
                        acc.mean("loss"),
                        acc.mean("loss_ewc"),
                        round(time.time() - start_time, 4),
                    )
                )

        return acc

    def calculate_ewc_loss(self, loss_seg):
        """
        Calculates the Elastic Weight Consolidation (EWC) loss to penalize deviations from old knowledge.

        Args:
            loss_seg (torch.Tensor): Segmentation loss.

        Returns:
            torch.Tensor: The calculated EWC loss.
        """
        loss_ewc = torch.zeros(1).to(loss_seg.device)
        if self.model.importance_weights is not None:
            for param_old, param_new, weights in zip(
                self.model.backbone_old.parameters(),
                self.model.backbone_new.parameters(),
                self.model.importance_weights,
            ):
                if param_new.requires_grad:
                    loss_ewc += torch.sum(weights * (param_new - param_old) ** 2)  # / self.model.n_params_backbone
        return loss_ewc

    def calc_importance_weights(self, dataloader, loss_f):
        """
        Compute importance weights for EWC (Elastic Weight Consolidation)

        Args:
            dataloader (Dataloader): training dataloader

        Returns:
            param_importance (List[torch.Tensor]): list of tensors representing parameter importance
        """
        # Initialize importance weights for each parameter as zeros
        param_importance = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]

        for data in tqdm(dataloader, desc="Calculating Fisher Information", dynamic_ncols=True):
            inputs, targets = self.get_inputs_targets(data)
            outputs = self.get_outputs(inputs)

            # Compute loss
            loss = loss_f(outputs, targets)

            # Zero the gradients of the model parameters
            self.model.backbone_new.zero_grad()

            # Compute the gradients (first-order derivatives)
            loss.backward()

            # Accumulate the Fisher Information approximation
            for i, param in enumerate(self.model.backbone_new.parameters()):
                if param.grad is not None:
                    # Use squared gradient to approximate the diagonal of the Fisher Information matrix
                    param_importance[i] += (param.grad**2) / len(dataloader)

        return param_importance

    def track_loss(self, acc, epoch, config, phase="train"):
        r"""Tracks loss in tensorboard.

        Args:
            acc (mp.eval.accumulator.Accumulator): accumulator holding losses
            epoch (int): current epoch
            config (dict): configuration dictionary from parsed arguments
            phase (string): either "test" or "train"
        """

        self.writer_add_scalar(f"loss_{phase}/loss_seg", acc.mean("loss_seg"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_ewc", acc.mean("loss_ewc"), epoch)
        self.writer_add_scalar(f"loss_{phase}/loss_comb", acc.mean("loss"), epoch)


# def calc_importance_weights_si(self, dataloader, loss_f):
#     """
#     Compute importance weights for SI (Synaptic Intelligence)

#     Args:
#         dataloader (Dataloader): training dataloader

#     Returns:
#         param_importance (List[torch.Tensor]): list of tensors representing parameter importance
#     """
#     # Initialize importance weights for each parameter as zeros
#     param_importance = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]

#     # Initialize dictionary to store cumulative parameter changes
#     omega = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]
#     prev_params = [param.clone().detach() for param in self.model.backbone_new.parameters()]

#     for data in tqdm(dataloader, desc="Calculating Synaptic Intelligence", dynamic_ncols=True):
#         inputs, targets = self.get_inputs_targets(data)
#         outputs = self.get_outputs(inputs)

#         # Compute loss
#         loss = loss_f(outputs, targets)

#         # Zero the gradients of the model parameters
#         self.model.backbone_new.zero_grad()

#         # Compute the gradients (first-order derivatives)
#         loss.backward()

#         # Accumulate the path integral for each parameter (effective parameter change)
#         for i, param in enumerate(self.model.backbone_new.parameters()):
#             if param.grad is not None:
#                 # Calculate parameter change (difference from previous parameter state)
#                 delta_param = param - prev_params[i]
#                 # Update omega with the contribution of this parameter change
#                 omega[i] += delta_param * param.grad
#                 prev_params[i] = param.clone().detach()  # Update the previous parameter state

#     # Normalize the omega values to get the final parameter importance
#     for i in range(len(param_importance)):
#         param_importance[i] = omega[i] / (omega[i].abs().sum() + 1e-10)  # Avoid division by zero

#     return param_importance

# def calc_importance_weights_rwalk(self, dataloader, loss_f):
#     """
#     Compute importance weights for RWalk (Regularized Walk in Parameter Space)

#     Args:
#         dataloader (Dataloader): training dataloader

#     Returns:
#         param_importance (List[torch.Tensor]): list of tensors representing parameter importance
#     """
#     # Initialize importance weights for each parameter as zeros
#     param_importance = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]

#     # Part 1: Fisher Information approximation (like EWC)
#     fisher_info = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]

#     for data in tqdm(dataloader, desc="Calculating Fisher Information for RWalk", dynamic_ncols=True):
#         inputs, targets = self.get_inputs_targets(data)
#         outputs = self.get_outputs(inputs)

#         # Compute loss
#         loss = loss_f(outputs, targets)

#         # Zero the gradients of the model parameters
#         self.model.backbone_new.zero_grad()

#         # Compute the gradients (first-order derivatives)
#         loss.backward()

#         # Accumulate the Fisher Information approximation
#         for i, param in enumerate(self.model.backbone_new.parameters()):
#             if param.grad is not None:
#                 fisher_info[i] += (param.grad**2) / len(dataloader)

#     # Part 2: Path integral accumulation (like SI)
#     omega = [torch.zeros_like(param) for param in self.model.backbone_new.parameters()]
#     prev_params = [param.clone().detach() for param in self.model.backbone_new.parameters()]

#     for data in tqdm(dataloader, desc="Calculating Path Integral for RWalk", dynamic_ncols=True):
#         inputs, targets = self.get_inputs_targets(data)
#         outputs = self.get_outputs(inputs)

#         # Compute loss
#         loss = loss_f(outputs, targets)

#         # Zero the gradients of the model parameters
#         self.model.backbone_new.zero_grad()

#         # Compute the gradients (first-order derivatives)
#         loss.backward()

#         # Accumulate the path integral for each parameter
#         for i, param in enumerate(self.model.backbone_new.parameters()):
#             if param.grad is not None:
#                 delta_param = param - prev_params[i]
#                 omega[i] += delta_param * param.grad
#                 prev_params[i] = param.clone().detach()

#     # Combine Fisher Information and Omega to get the final importance weights
#     for i in range(len(param_importance)):
#         param_importance[i] = fisher_info[i] + omega[i]
#         param_importance[i] = param_importance[i] / (param_importance[i].abs().sum() + 1e-10)  # Normalize

#     return param_importance
