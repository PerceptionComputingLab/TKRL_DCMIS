import torch.utils.data
import os
from pprint import pprint

# Ensuring CUDA errors are reported in the main process
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from mp.data.datasets.ds_prostate import Prostate
from mp.data.datasets.ds_hippocampus import Hippocampus
from mp.data.datasets.ds_cardiac_mm import Cardiac
from mp.data.datasets.ds_optic import Optic
from mp.data.datasets.ds_mr_brain import Brain
from mp.data.datasets.ds_polyp import Polyp
from mp.data.data import Data
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from torch.utils.data import DataLoader

from mp.models.continual.kd import KD
from mp.models.continual.mas import MAS
from mp.models.continual.tkrl import TKRL

from mp.eval.losses.losses_segmentation import LossDiceBCE

from mp.agents.kd_agent import KDAgent
from mp.agents.mas_agent import MASAgent
from mp.agents.ewc_agent import EWCAgent
from mp.agents.mib_agent import MIBAgent
from mp.agents.plop_agent import PLOPAgent
from mp.agents.seq_agent import SEQAgent
from mp.agents.ted_agent import TEDAgent

from mp.agents.tkrl_agent import TKRLAgent
from mp.agents.pcd_agent import PCDAgent
from mp.agents.vma_agent import VMAAgent
from mp.agents.rmae_agent import RMAEAgent



def get_dataset(config, exp):
    """
    Initializes and returns the dataset for the given configuration.

    Args:
        config (dict): Configuration dictionary with dataset and training details.
        exp (object): Experiment object for setting data splits.

    Returns:
        tuple: Contains training and test dataloaders, datasets, experiment run, and label details.
    """
    data = Data()
    subset_list = []

    # Load datasets based on configuration
    if config["dataset"] == "brain":
        subset_list = ["t2", "t1ce", "flair"]
        for name in subset_list:
            dataset_domain = Brain(subset=name)
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    elif config["dataset"] == "polyp":
        subset_list = ["C1", "C2", "C3", "C4", "C5", "C6"]
        for name in subset_list:
            dataset_domain = Polyp(subset=name)
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    elif config["dataset"] == "prostate":
        subset_list = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
        for name in subset_list:
            dataset_domain = Prostate(subset=name)
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    elif config["dataset"] == "hippocampus":
        subset_list = ["DecathlonHippocampus", "DryadHippocampus", "HarP"]
        for name in subset_list:
            dataset_domain = Hippocampus(subset=name)
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    elif config["dataset"] == "mm":
        subset_list = ["Siemens", "Philips", "GE", "Canon"]
        target = {"i": 1, "o": 2, "r": 3}
        for name in subset_list:
            dataset_domain = Cardiac(subset=name, target=target[config["target_class"]])
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    elif config["dataset"] == "optic":
        subset_list = ["Domain1", "Domain2", "Domain3", "Domain4"]
        target = {"i": 1, "o": 2}
        for name in subset_list:
            dataset_domain = Optic(subset=name, target=target[config["target_class"]])
            dataset_domain.name = name
            data.add_dataset(dataset_domain)

    exp.set_data_splits(data)
    exp_run = exp.get_run(0, reload_exp_run=(config["resume_epoch"] is not None))
    datasets = {}

    # Prepare data loaders for each subset and split (train/test)
    for dataset_name, dataset in data.datasets.items():
        for split, data_indices in exp.splits[dataset_name][exp_run.run_ix].items():
            data_indices = data_indices[:None]  # Limit number of samples if debugging
            if len(data_indices) > 0:
                aug_type = config["augmentation"] if "test" not in split else "none"
                datasets[(dataset_name, split)] = PytorchSeg2DDataset(
                    dataset=dataset,
                    ix_lst=data_indices,
                    size=config["input_shape"],
                    norm_key="rescaling",
                    aug_key=aug_type,
                    resize=(not config["no_resize"]),
                    channel_labels=True,
                )
    # for ds_name, ds in datasets.items():
    #     print(f"{ds_name}: {len(ds)}")
    #     for instance_ix, instance in enumerate(ds.instances):
    #         subject_name = instance.name
    #         print(f"{subject_name}: {subject_name}")
    # Handle joint training approach separately
    if config["approach"] in ["joint"]:
        joint_train_dataset = torch.utils.data.ConcatDataset(datasets[(name, "train")] for name in subset_list)
        joint_test_dataset = torch.utils.data.ConcatDataset(datasets[(name, "test")] for name in subset_list)

        train_dataloader = DataLoader(
            dataset=joint_train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=len(config["device_ids"]) * config["n_workers"],
        )

        test_dataloader = DataLoader(
            dataset=joint_test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=len(config["device_ids"]) * config["n_workers"],
        )

        return (
            [train_dataloader],
            [test_dataloader],
            datasets,
            exp_run,
            {"label_nr": data.nr_labels, "label_names": data.label_names},
        )

    # Prepare dataloaders for individual training and test datasets
    train_dataloaders = []
    test_dataloaders = []
    for subset_name in subset_list:
        train_dataloaders.append(
            DataLoader(
                dataset=datasets[(subset_name, "train")],
                batch_size=config["batch_size"],
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=len(config["device_ids"]) * config["n_workers"],
            )
        )
        test_dataloaders.append(
            DataLoader(
                dataset=datasets[(subset_name, "test")],
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=len(config["device_ids"]) * config["n_workers"],
            )
        )

    return (
        train_dataloaders,
        test_dataloaders,
        datasets,
        exp_run,
        {"label_nr": data.nr_labels, "label_names": data.label_names},
    )


def get_model(config, nr_labels):
    """
    Initializes and returns the model for the given approach.

    Args:
        config (dict): Configuration dictionary with model and approach details.
        nr_labels (int): Number of labels in the dataset.

    Returns:
        torch.nn.Module: Initialized model.
    """
    model_mapping = {
        "mas": MAS,
        "ewc": MAS,
        "kd": KD,
        "mib": KD,
        "plop": KD,
        "seq": MAS,
        "joint": MAS,
        "ted": KD,

        "tkrl": TKRL,
        "pcd": TKRL,
        "vma": TKRL,
        "rmae": TKRL,
    }

    model_class = model_mapping[config["approach"]]
    model = model_class(input_shape=config["input_shape"], nr_labels=nr_labels)
    model.to(config["device"])

    return model


def get_loss_type(config):
    """
    Returns the loss function to be used for training.

    Args:
        config (dict): Configuration dictionary with loss details.

    Returns:
        Loss function object.
    """
    return LossDiceBCE(bce_weight=1.0, smooth=1.0, device=config["device"])


def get_agent(config, model, label_names):
    """
    Initializes and returns the training agent based on the approach.

    Args:
        config (dict): Configuration dictionary with agent details.
        model (torch.nn.Module): Model to be used by the agent.
        label_names (list): List of label names in the dataset.

    Returns:
        Agent object: Initialized agent for training.
    """
    agent_mapping = {
        "mas": MASAgent,
        "ewc": EWCAgent,
        "kd": KDAgent,
        "mib": MIBAgent,
        "plop": PLOPAgent,
        "seq": SEQAgent,
        "joint": SEQAgent,
        "ted": TEDAgent,
        
        "tkrl": TKRLAgent,
        "pcd": PCDAgent,
        "vma": VMAAgent,
        "rmae": RMAEAgent,
    }

    agent_class = agent_mapping[config["approach"]]
    agent = agent_class(model=model, label_names=label_names, device=config["device"])

    return agent
