import os
import sys
import torch
import torch.optim as optim

from mp.experiments.experiment import Experiment
from mp.eval.result import Result
from mp.utils.tensorboard import create_writer
from mp.utils.helper_functions import seed_all
from args import parse_args_as_dict
from get import get_dataset, get_model, get_loss_type, get_agent

from pprint import pprint

# Set the number of CPU threads for PyTorch
torch.set_num_threads(8)

# Parse the configuration arguments from the command line
config = parse_args_as_dict(sys.argv[1:])
seed_all(config["seed"])  # Set a fixed random seed for reproducibility
print(config)

# Initialize the experiment
exp = Experiment(
    config=config,
    name=config["experiment_name"],
    notes="",
    reload_exp=(config["resume_epoch"] is not None),
)

# Load datasets and initialize the model
train_dataloader, test_dataloader, datasets, exp_run, label_info = get_dataset(config, exp=exp)
model = get_model(config, nr_labels=label_info["label_nr"])
loss_function = get_loss_type(config)
results = Result()
pprint(exp.splits)

# Initialize the agent responsible for training
agent = get_agent(config, model=model, label_names=label_info["label_names"])
agent.summary_writer = create_writer(path=os.path.join(exp_run.paths["states"], ".."), init_epoch=0)

nr_epochs = 0
config["continual"] = False  # Set to non-continual learning by default

# Training loop over the datasets
for dataset_idx, dataloader in enumerate(train_dataloader):
    print(f"Training on dataset index: {dataset_idx}")
    if dataset_idx == 0 and config["resume_from"] is not None:
        print("Resuming from ", config["resume_from"])
        resume_states_path = os.path.join(
            exp_run.paths["states"].replace(config["experiment_name"], config["resume_from"])
        )
        best_val_log_path = os.path.join(resume_states_path, "val_track.txt")
        with open(best_val_log_path, "r") as f:
            best_states = [int(line.strip()) for line in f.readlines()]
        best_init_epoch = best_states[0]
        agent.restore_state(resume_states_path, best_init_epoch)

        with torch.no_grad():
            agent.model.eval()
            agent.track_metrics(nr_epochs, results, loss_function, datasets)
        agent.model.finish()
        config["continual"] = True
        nr_epochs = config["epochs"]
        continue
    init_epoch = nr_epochs
    nr_epochs = config["epochs"] + init_epoch

    # Set the optimizer based on whether it's continual learning or not
    if config["continual"]:
        model.set_optimizers(optim.AdamW, lr=config["lr_2"])
        model.backbone_old.eval()
    else:
        model.set_optimizers(optim.AdamW, lr=config["lr"])

    # Set up the learning rate scheduler
    model.backbone_scheduler = optim.lr_scheduler.StepLR(model.backbone_optim, step_size=5, gamma=0.5)

    # Train the model with the current dataloader
    agent.train(
        results=results,
        loss_f=loss_function,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        config=config,
        init_epoch=init_epoch,
        nr_epochs=nr_epochs,
        eval_datasets=datasets,
        save_path=exp_run.paths["states"],
        dataset_index=dataset_idx,
        exp_path=exp_run.paths["states"],
    )

    # Set continual learning to True for subsequent iterations if not using sequential approach
    if config["approach"] not in ["seq"]:
        config["continual"] = True
    else:
        agent.model.reset_bn_stats_new_model()

# Update dataset name for logging and finishing
if config["dataset"] == "mm":
    config["dataset"] = "cardiac"

# Complete the experiment and plot metrics
exp_run.finish(results=results, plot_metrics=[f"Mean_ScoreDice[{config['dataset']}]"])


print("finish")
