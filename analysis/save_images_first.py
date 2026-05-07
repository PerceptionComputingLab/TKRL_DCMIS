import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import get_dataset, get_model, get_agent
from mp.utils.helper_functions import seed_all
from mp.eval.losses.losses_segmentation import LossDice
from tqdm import tqdm


def dice_coefficient(y_true, y_pred):
    """计算Dice系数"""
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + 1e-5) / (np.sum(y_true) + np.sum(y_pred) + 1e-5)


def compute_metrics(y_true, y_pred):
    """计算指标并返回一个字典"""
    return {
        "Dice": dice_coefficient(y_true, y_pred),
    }


def visualize_prediction(image, target, predicted, dice_score, title, save_path=None):
    """
    可视化预测结果并保存图片（可选）

    Args:
        image: 原始图像。
        target: Ground truth 标签。
        predicted: 模型预测结果。
        dice_score: 计算的 Dice 分数。
        title: 图片标题。
        save_path: 如果提供了路径，则将图像保存到该路径。
    """
    alpha_target = np.where(target == 1, 0.5, 0)
    alpha_predicted = np.where(predicted == 1, 0.5, 0)

    ground_truth_rgba = np.zeros((target.shape[0], target.shape[1], 4))
    ground_truth_rgba[..., 0] = 1  # Red channel for ground truth
    ground_truth_rgba[..., 3] = alpha_target

    predicted_rgba = np.zeros((predicted.shape[0], predicted.shape[1], 4))
    predicted_rgba[..., 2] = 1  # Blue channel for prediction
    predicted_rgba[..., 3] = alpha_predicted

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(image, 3), cmap="gray")
    ax.imshow(np.rot90(ground_truth_rgba, 3))
    ax.imshow(np.rot90(predicted_rgba, 3))

    # 在图像上添加 Dice 分数
    plt.title(f"{title} (Dice: {dice_score:.4f})", size=16)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def extract_model_outputs(agent, dataloader):
    image_all = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting model outputs"):
            sample_data = {}
            inputs, target = agent.get_inputs_targets(data)
            sample_data["inputs"] = inputs.detach().cpu().numpy()
            sample_data["target"] = target.detach().cpu().numpy()
            outputs = agent.get_outputs(inputs)
            sample_data["outputs"] = outputs.detach().cpu().numpy()
            image_all.append(sample_data)
            break
    return image_all


def process_experiment(config, approach, dataset_, out_dir):
    """处理一个实验，包括模型恢复、预测和结果可视化"""
    exp = Experiment(config=config, name=config["experiment_name"], reload_exp=(config["resume_epoch"]))
    train_dataloader, test_dataloader, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")

    with open(best_states_file, "r") as f:
        best_states = [int(line.strip()) for line in f.readlines()]

    model = get_model(config, nr_labels=label_inf["label_nr"])
    agent = get_agent(config, model=model, label_names=label_inf["label_names"])
    loader = test_dataloader[0]

    for best_state in best_states:
        agent.restore_state(exp_run.paths["states"], best_state)
        features = extract_model_outputs(agent, loader)

        for _, feature in enumerate(features):
            idx = 5
            image = feature["inputs"][idx][0]
            target = feature["target"][idx][1]
            new_prob = feature["outputs"][idx]
            predicted = np.argmax(new_prob, axis=0)

            metrics = compute_metrics(target, predicted)
            dice_score = metrics["Dice"]
            print(f"Metrics for approach {approach}, {dataset_} {best_state}: {metrics}")
            visualize_prediction(
                image,
                target,
                predicted,
                dice_score=dice_score,  # 传递 Dice 分数
                title=f"{approach} - {dataset_} {best_state}",
                save_path=os.path.join(out_dir, f"{dataset_}_{approach}_{best_state}_{dice_score:.4g}.png"),
            )
            break
        if config["approach"] != "seq":
            agent.model.finish()


def main():
    config = parse_args_as_dict(sys.argv[1:])
    seed_all(42)
    from mp.paths import abs_path, storage_path

    datasets_ = ["hippocampus"]
    approaches = ["seq", "kd", "pcd", "vma", "tkrl", "mas", "ewc", "mib", "plop", "ted"]

    out_dir = os.path.join(abs_path, storage_path, "save_images_first")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    target = "i"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dataset_ in datasets_:
        for approach in approaches:
            if dataset_ == "mm" or dataset_ == "optic":
                config["experiment_name"] = f"{dataset_}{target}-{approach}"
            else:
                config["experiment_name"] = f"{dataset_}-{approach}"
            config["target_class"] = target
            config["approach"] = approach
            config["dataset"] = dataset_
            config["resume_epoch"] = 40
            config["device-ids"] = "5"
            config["device"] = "cuda:5"
            print(f"Running experiment with config: {config}")
            process_experiment(config, approach, dataset_, out_dir)


if __name__ == "__main__":
    main()
