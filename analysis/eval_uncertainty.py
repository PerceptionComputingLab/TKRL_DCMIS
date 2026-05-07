import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from mp.experiments.experiment import Experiment
from args import parse_args_as_dict
from get import get_dataset, get_model, get_agent
from mp.utils.helper_functions import seed_all
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


def extract_model_outputs(agent, dataloaders):
    """
    提取模型的输出并计算每个样本的错误分类像素的总熵值

    Args:
        agent: 模型代理。
        dataloaders: 包含多个 DataLoader 的列表，每个 DataLoader 都需要被处理。

    Returns:
        all_incorrect_entropy_values: 每个 DataLoader 的错误分类像素总熵值的列表字典。
    """
    # 创建一个字典来存储每个 DataLoader 的错误分类像素总熵值
    all_incorrect_entropy_values = {}

    with torch.no_grad():
        # 遍历每个 DataLoader
        for dataloader_idx, dataloader in enumerate(dataloaders):
            all_incorrect_entropy_values[dataloader_idx] = []  # 为当前 DataLoader 初始化一个空的列表

            for data in tqdm(dataloader, desc=f"Extracting model outputs (Loader {dataloader_idx})"):
                sample_data = {}
                inputs, target = agent.get_inputs_targets(data)
                sample_data["inputs"] = inputs.detach().cpu().numpy()
                sample_data["target"] = target.detach().cpu().numpy()
                outputs = agent.get_outputs(inputs)
                sample_data["outputs"] = outputs.detach().cpu().numpy()

                for idx in range(len(sample_data["inputs"])):
                    image = sample_data["inputs"][idx][0]
                    target = sample_data["target"][idx][1]
                    new_prob = sample_data["outputs"][idx]
                    predicted = np.argmax(new_prob, axis=0)

                    # 计算每个像素的熵（不确定性）
                    pixel_entropy_map = pixel_entropy(new_prob)  # (H x W)

                    # 识别错误分类的像素
                    incorrect_pixels = (predicted != target).astype(np.float32)  # 错误分类的像素位置

                    # 仅统计错误分类像素的不确定性（熵值）
                    incorrect_entropy = pixel_entropy_map * incorrect_pixels

                    # # 计算该样本的错误分类像素的总熵值
                    # total_incorrect_entropy = np.sum(incorrect_entropy)

                    # # 将该样本的错误分类像素总熵值添加到当前 DataLoader 的列表中
                    # all_incorrect_entropy_values[dataloader_idx].append(total_incorrect_entropy)

                    # 计算错误分类像素的数量
                    num_incorrect_pixels = np.sum(incorrect_pixels)

                    # 计算该样本错误分类像素的平均熵值（防止除以零）
                    if num_incorrect_pixels > 0:
                        avg_incorrect_entropy = np.sum(incorrect_entropy) / num_incorrect_pixels
                    else:
                        avg_incorrect_entropy = 0.0

                    # 将该样本的错误分类像素的平均熵值添加到当前 DataLoader 的列表中
                    all_incorrect_entropy_values[dataloader_idx].append(avg_incorrect_entropy)

    return all_incorrect_entropy_values


def pixel_entropy(probabilities):
    """计算每个像素的熵，不确定性度量"""
    # probabilities 是形状为 (num_classes, height, width) 的数组
    # 对每个像素在所有类别上的概率分布计算熵
    entropy_map = -np.sum(probabilities * np.log(probabilities + 1e-5), axis=0)  # 对每个像素计算熵
    return entropy_map


def visualize_loader_uncertainty(dataloader, uncertainties, avg_uncertainty, std_uncertainty, title, save_path=None):
    """
    可视化整个数据加载器的预测不确定性分布

    Args:
        dataloader: 用于加载数据的dataloader对象。
        uncertainties: 数据加载器中每个样本的不确定性。
        avg_uncertainty: 不确定性的平均值。
        std_uncertainty: 不确定性的标准差。
        title: 图片标题。
        save_path: 如果提供了路径，则将图像保存到该路径。
    """
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties, bins=30, alpha=0.7, label="Uncertainty Distribution")
    plt.axvline(
        avg_uncertainty, color="r", linestyle="dashed", linewidth=2, label=f"Avg Uncertainty: {avg_uncertainty:.4f}"
    )
    plt.axvline(
        avg_uncertainty + std_uncertainty,
        color="g",
        linestyle="dashed",
        linewidth=2,
        label=f"Std Uncertainty: {std_uncertainty:.4f}",
    )
    plt.axvline(avg_uncertainty - std_uncertainty, color="g", linestyle="dashed", linewidth=2)

    plt.title(title)
    plt.xlabel("Uncertainty (Entropy)")
    plt.ylabel("Frequency")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def process_experiment(config, approach, dataset_, out_dir):
    """处理一个实验，包括模型恢复、预测和结果可视化"""
    exp = Experiment(config=config, name=config["experiment_name"], reload_exp=(config["resume_epoch"]))
    train_dataloader, test_dataloader, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")

    with open(best_states_file, "r") as f:
        best_states = [int(line.strip()) for line in f.readlines()]

    model = get_model(config, nr_labels=label_inf["label_nr"])
    agent = get_agent(config, model=model, label_names=label_inf["label_names"])
    if config["approach"] != "seq":
        agent.model.finish()
    agent.restore_state(exp_run.paths["states"], best_states[-1])

    # 提取所有样本的错误分类像素的总熵值，按 DataLoader 分组
    all_incorrect_entropy_values = extract_model_outputs(agent, test_dataloader)

    # 输出每个 DataLoader 的统计结果
    total_entropy_list = []
    mean_entropy_list = []
    for dataloader_idx, incorrect_entropy_values in all_incorrect_entropy_values.items():
        print(f"DataLoader {dataloader_idx}:")
        total_entropy = np.sum(incorrect_entropy_values)
        mean_entropy = np.mean(incorrect_entropy_values)
        print(f"  Total entropy of incorrect pixels: {total_entropy}")
        print(f"  Mean entropy of incorrect pixels: {mean_entropy}")
        total_entropy_list.append(total_entropy)
        mean_entropy_list.append(mean_entropy)

    return total_entropy_list, mean_entropy_list


def main():
    config = parse_args_as_dict(sys.argv[1:])
    seed_all(42)

    datasets_ = ["prostate"]
    approaches = ["seq", "kd", "vma"]
    target = "i"

    from mp.paths import abs_path, storage_path

    out_dir = os.path.join(abs_path, storage_path, "uncertainty_figures")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dataset_ in datasets_:
        plt.figure(figsize=(10, 6))
        for approach in approaches:
            config["experiment_name"] = f"{dataset_}-{approach}"
            config["target_class"] = target
            config["approach"] = approach
            config["dataset"] = dataset_
            config["resume_epoch"] = 40
            config["device-ids"] = "0"
            config["device"] = "cuda:0"
            print(f"Running experiment with config: {config}")
            total_entropy_list, mean_entropy_list = process_experiment(config, approach, dataset_, out_dir)
            plt.plot(range(len(total_entropy_list)), total_entropy_list, label=f"Total Entropy ({approach})")
            plt.plot(range(len(mean_entropy_list)), mean_entropy_list, label=f"Mean Entropy ({approach})")

        plt.title(f"Total and Mean Entropy for Dataset '{dataset_}'")
        plt.xlabel("Data Loader Index")
        plt.ylabel("Entropy")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{dataset_}-{approach}.png"))
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
