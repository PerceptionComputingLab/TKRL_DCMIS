import random
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import cv2

sns.set_style("whitegrid")
import os
import sys

from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import *
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
from torchvision import transforms
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm
import numpy as np

torch.set_num_threads(4)
config = parse_args_as_dict(sys.argv[1:])
seed_all(42)


def load_probe(path_dir, dataset_idx, batch_size=16):
    batch_imgs = []
    for i in range(batch_size):
        path = os.path.join(path_dir, "{}_{}_probes_noise.png".format(dataset_idx, i))
        img = cv2.imread(path)
        # transfor bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch_imgs.append(img)
    batch_imgs = np.array(batch_imgs)
    # transfer to tensor
    return torch.from_numpy(batch_imgs).permute((0, 3, 1, 2)).float()


def cluster_data(data, outpath=None, batch_size=16):
    # data = data.reshape(batch_size, -1)
    data = np.mean(data, (2, 3))
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    # tsne = TSNE(n_components=2, perplexity=10)
    # X = tsne.fit_transform(data)
    dict_ = {}
    dict_["x"] = X[:, 0]
    dict_["y"] = X[:, 1]
    if outpath is not None:
        df = pd.DataFrame(dict_)
        df.to_csv(outpath, index=False)


if __name__ == "__main__":

    transf = transforms.ToTensor()
    datasets_ = ["prostate"]
    approaches = ["pcd"]
    from mp.paths import abs_path, storage_path

    out_dir = os.path.join(abs_path, storage_path, "image_pcd")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    font_path = "analysis/times-new-roman.ttf"
    font_prop = fm.FontProperties(fname=font_path, size=16)

    for dataset_ in datasets_:
        for approach in approaches:
            config["experiment_name"] = dataset_ + "-" + approach
            config["approach"] = approach
            config["dataset"] = dataset_
            config["resume_epoch"] = 40
            config["device-ids"] = "4"
            config["ablation"] = True
            print(config)

            exp = Experiment(
                config=config,
                name=config["experiment_name"],
                notes="",
                reload_exp=(config["resume_epoch"]),
            )
            train_dataloader, _, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
            best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")
            best_states = []
            with open(best_states_file, "r") as f:
                for line in f.readlines():
                    best_states.append(int(line.replace("\n", "")))

            model = get_model(config, nr_labels=label_inf["label_nr"])

            agent = get_agent(config, model=model, label_names=label_inf["label_names"])

            states_old = best_states[0]
            agent.restore_state(exp_run.paths["states"], states_old)
            agent.model.finish()  # change new model with old one, otherwise it fails to restore the state
            states_new = best_states[1]
            agent.restore_state(exp_run.paths["states"], states_new)

            dataset_list = []
            for ds_name, _ in datasets.items():
                if ds_name[0] not in dataset_list:
                    dataset_list.append(ds_name[0])
            train_dataset = train_dataloader[1]
            batch_size = 16

            with torch.no_grad():
                for data in tqdm(train_dataset, disable=True):
                    # Get data
                    inputs, targets = agent.get_inputs_targets(data)

                    # traditional
                    outputs_old_list = agent.model.get_embedding_old(inputs)  # c1, c2, c3, c4

                    # ska augmentation
                    noise_img = torch.randn_like(inputs).to(inputs.device)
                    outputs_old_ska_list = agent.model.get_embedding_old(noise_img)  # c1, c2, c3, c4

                    # noise injection
                    noise_img = noise_img + inputs
                    outputs_old_injection_list = agent.model.get_embedding_old(noise_img)  # c1, c2, c3, c4

                    # data augmentation
                    aug_img = torch.flip(inputs, [2])
                    aug_img = torch.rot90(aug_img, k=1, dims=[2, 3])
                    aug_img = aug_img * random.uniform(0.8, 1.2)
                    aug_img = aug_img.to(inputs.device)
                    outputs_old_aug_list = agent.model.get_embedding_old(aug_img)  # c1, c2, c3, c4

                    # probe
                    probe_img = load_probe(exp_run.paths["states"], dataset_idx=1, batch_size=batch_size)
                    probe_img = probe_img.to(inputs.device)
                    outputs_old_probe_list = agent.model.get_embedding_old(probe_img)  # c1, c2, c3, c4

                    # scale range
                    for j in range(len(outputs_old_list)):
                        outputs_old = outputs_old_list[j].to("cpu")
                        outputs_old_ska = outputs_old_ska_list[j].to("cpu")
                        outputs_old_injection = outputs_old_injection_list[j].to("cpu")
                        outputs_old_aug = outputs_old_aug_list[j].to("cpu")
                        outputs_old_probe = outputs_old_probe_list[j].to("cpu")

                        X_old = outputs_old.detach().numpy()
                        X_old_ska = outputs_old_ska.detach().numpy()
                        X_old_injection = outputs_old_injection.detach().numpy()
                        X_old_aug = outputs_old_aug.detach().numpy()
                        X_old_probe = outputs_old_probe.detach().numpy()

                        # distribution of the old knowledge
                        cluster_data(X_old, os.path.join(out_dir, "old_knowledge_{}.csv".format(j)))

                        # distribution of the old knowledge with ska augmentation
                        cluster_data(X_old_ska, os.path.join(out_dir, "old_knowledge_ska_{}.csv".format(j)))

                        # distribution of the old knowledge with noise injection
                        cluster_data(X_old_injection, os.path.join(out_dir, "old_knowledge_injection_{}.csv".format(j)))

                        # distribution of the old knowledge with aug
                        cluster_data(X_old_aug, os.path.join(out_dir, "old_knowledge_aug_{}.csv".format(j)))

                        # distribution of the old knowledge with probe
                        cluster_data(X_old_probe, os.path.join(out_dir, "old_knowledge_probe_{}.csv".format(j)))

                        # distribution of all types of knowledge
                        X_old_all = np.concatenate((X_old, X_old_ska, X_old_injection, X_old_aug, X_old_probe))
                        cluster_data(
                            X_old_all,
                            os.path.join(out_dir, "all_knowledge_{}.csv".format(j)),
                            batch_size=batch_size * 5,
                        )

                        # load data from csv
                        data1 = pd.read_csv(os.path.join(out_dir, "old_knowledge_{}.csv".format(j)))
                        data2 = pd.read_csv(os.path.join(out_dir, "old_knowledge_ska_{}.csv".format(j)))
                        data3 = pd.read_csv(os.path.join(out_dir, "old_knowledge_injection_{}.csv".format(j)))
                        data4 = pd.read_csv(os.path.join(out_dir, "old_knowledge_aug_{}.csv".format(j)))
                        data5 = pd.read_csv(os.path.join(out_dir, "old_knowledge_probe_{}.csv".format(j)))

                        # create figure
                        plt.figure(figsize=(20, 4))

                        plt.subplot(1, 5, 1)
                        plt.scatter(
                            data1["x"],
                            data1["y"],
                            color="red",
                            alpha=0.8,
                            edgecolors="white",
                        )
                        plt.title("(a) Original data", fontproperties=font_prop)
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)

                        plt.subplot(1, 5, 2)
                        plt.scatter(
                            data2["x"],
                            data2["y"],
                            color="blue",
                            alpha=0.8,
                            edgecolors="white",
                        )
                        plt.title("(b) Random noise", fontproperties=font_prop)
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)

                        plt.subplot(1, 5, 3)
                        plt.scatter(
                            data3["x"],
                            data3["y"],
                            color="orange",
                            alpha=0.8,
                            edgecolors="white",
                        )
                        plt.title("(c) Noise-injected ", fontproperties=font_prop)
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)

                        plt.subplot(1, 5, 4)
                        plt.scatter(
                            data4["x"],
                            data4["y"],
                            color="green",
                            alpha=0.8,
                            edgecolors="white",
                        )
                        plt.title("(d) Augmented data ", fontproperties=font_prop)
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)

                        plt.subplot(1, 5, 5)
                        plt.scatter(
                            data5["x"],
                            data5["y"],
                            color="purple",
                            alpha=0.8,
                            edgecolors="white",
                        )
                        plt.title("(e) Our knowledge probe", fontproperties=font_prop)
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)

                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, "old_knowledge_{}_comparison.png").format(j), dpi=300)
                        plt.show()

                    plt.figure(figsize=(20, 4))
                    for j in range(len(outputs_old_list)):
                        data6 = pd.read_csv(os.path.join(out_dir, "all_knowledge_{}.csv".format(j)))
                        plt.subplot(1, 4, j + 1)
                        for k in range(5):
                            plt.scatter(
                                data6["x"][k * batch_size : (k + 1) * batch_size],
                                data6["y"][k * batch_size : (k + 1) * batch_size],
                                alpha=0.8,
                                edgecolors="white",
                                c=["red", "blue", "orange", "green", "purple"][k],
                            )
                        plt.xlabel("PC1", fontproperties=font_prop)
                        plt.ylabel("PC2", fontproperties=font_prop)
                        plt.legend(
                            ["Original data", "Random noise", "Noise-injected", "Augmented data", "Our knowledge probe"]
                        )
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, "all_knowledge.png"), dpi=300)
                    plt.show()

                    exit()
