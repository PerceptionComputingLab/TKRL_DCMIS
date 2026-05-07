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
import torch.nn.functional as F
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, Normalize


def process_experiment(config, approach, dataset_, out_dir):

    font_path = "analysis/times-new-roman.ttf"
    font_prop = fm.FontProperties(fname=font_path, size=16)
    exp = Experiment(config=config, name=config["experiment_name"], reload_exp=(config["resume_epoch"]))
    train_dataloader, test_dataloader, datasets, exp_run, label_inf = get_dataset(config, exp=exp)
    best_states_file = os.path.join(exp_run.paths["states"], "val_track.txt")

    with open(best_states_file, "r") as f:
        best_states = [int(line.strip()) for line in f.readlines()]

    model = get_model(config, nr_labels=label_inf["label_nr"])
    agent = get_agent(config, model=model, label_names=label_inf["label_names"])
    if config["approach"] != "seq":
        agent.model.finish()

    idx_data = 3  # prostate
    # idx_data = 1  # hippocampus
    agent.restore_state(exp_run.paths["states"], best_states[idx_data])

    for data in tqdm(test_dataloader[idx_data], desc=f"Extracting model outputs (Loader {idx_data})"):
        sample_data = {}
        inputs, target = agent.get_inputs_targets(data)
        sample_data["inputs"] = inputs.detach().cpu().numpy()
        sample_data["target"] = target.detach().cpu().numpy()
        outputs = agent.get_outputs(inputs)
        agent.model.get_features_variance_ranks_visual(inputs)
        sample_data["probes_mask"] = agent.model.probes_mask.detach().cpu().numpy()
        sample_data["multi_mask"] = agent.model.mask_chosed_list
        sample_data["probes_uncertainty_mask"] = agent.model.probes_uncertainty_mask.detach().cpu().numpy()
        sample_data["multi_uncertainty_mask"] = agent.model.uncertain_mask_list
        sample_data["outputs"] = outputs.detach().cpu().numpy()

        out, outputs = agent.model.get_masked_embedding(inputs)
        recom = agent.model.mae_decoder(out)
        recom = F.interpolate(recom, size=inputs.shape[-2:])
        sample_data["recom"] = recom.detach().cpu().numpy()

        # Increase figure size for better visualization
        plt.figure(figsize=(18, 18))

        for idx in range(len(sample_data["inputs"])):
            # Use a 6x4 grid (6 rows, 4 columns)
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 18))
            axes = axes.flatten()  # Flatten axes for easier iteration

            image = sample_data["inputs"][idx][0]
            target = sample_data["target"][idx][1]
            prediction = sample_data["outputs"][idx][1]
            probes_mask = sample_data["probes_mask"][idx][0]
            recom = sample_data["recom"][idx][0]
            probes_uncertainty_mask = sample_data["probes_uncertainty_mask"][idx][0]

            # 第一行展示：原图、目标mask、统一mask、统一mask+原图
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Input", fontproperties=font_prop)
            axes[0].axis("off")

            axes[1].imshow(image, cmap="gray")
            axes[1].imshow(target, cmap="gray", alpha=0.5)
            axes[1].set_title("Target", fontproperties=font_prop)
            axes[1].axis("off")

            axes[2].imshow(probes_mask, cmap="gray")
            axes[2].set_title("Unified Mask", fontproperties=font_prop)
            axes[2].axis("off")

            axes[3].imshow(image, cmap="gray")
            mask_alpha = np.zeros_like(probes_mask)
            mask_alpha[probes_mask > 0] = 0.8  # 非零区域
            rgba_probes_mask = plt.cm.gray(probes_mask)  # 将probes_mask映射到灰度色卡
            rgba_probes_mask[..., -1] = mask_alpha  # 设置alpha通道
            axes[3].imshow(rgba_probes_mask)
            # axes[3].imshow(probes_mask, cmap="gray", alpha=0.5)
            axes[3].set_title("Unified Mask", fontproperties=font_prop)
            axes[3].axis("off")

            # 第二行展示：四个尺度mask
            for i in range(4):
                multimask = sample_data["multi_mask"][i][idx].squeeze().detach().cpu().numpy()
                # 显示原图
                axes[4 + i].imshow(image, cmap="gray", alpha=1.0)

                # 将multimask中值为0的区域设为完全透明
                mask_alpha = np.zeros_like(multimask)
                mask_alpha[multimask > 0] = 1  # 非零区域

                # 创建RGBA数组，使用灰度色卡
                rgba_multimask = plt.cm.gray(multimask)  # 将multimask映射到灰度色卡
                rgba_multimask[..., -1] = mask_alpha  # 设置alpha通道

                # 显示multimask
                axes[4 + i].imshow(rgba_multimask)

                axes[4 + i].set_title(f"Scale Mask {i+1}", fontproperties=font_prop)
                axes[4 + i].axis("off")

            # 第三行展示：四个尺度uncertainty融合原图

            for i in range(4):
                # 提取不确定性mask
                uncertain_mask = sample_data["multi_uncertainty_mask"][i][idx].squeeze().detach().cpu().numpy()

                # 显示image和不确定性mask，确保image在底层，uncertain_mask叠加在上面
                axes[8 + i].imshow(image, cmap="gray", alpha=1.0)  # 原图完全不透明

                # 设置透明背景，值为0的部分完全透明，其他区域半透明
                mask_alpha = np.zeros_like(uncertain_mask)
                mask_alpha[uncertain_mask > 0] = 0.8  # 非零区域设置半透明

                # 创建Normalize对象进行颜色归一化（仅影响颜色映射）
                norm = Normalize(vmin=uncertain_mask.min(), vmax=uncertain_mask.max())

                # 将不确定性mask转换为RGBA格式，影响的是颜色映射
                rgba_uncertainty = plt.cm.viridis(uncertain_mask, bytes=True)  # 使用viridis colormap，不会自动归一化
                rgba_uncertainty[..., -1] = (mask_alpha * 255).astype(int)  # 将alpha通道与透明度结合（0-255范围）

                # 显示不确定性mask，并应用颜色归一化
                im = axes[8 + i].imshow(rgba_uncertainty, cmap="viridis", norm=norm)

                axes[8 + i].set_title(f"Image + Uncertainty Mask {i+1}", fontproperties=font_prop)
                axes[8 + i].axis("off")

                # 为不确定性mask添加横向colorbar
                cbar_ax = fig.add_axes(
                    [
                        axes[8 + i].get_position().x0 - 0.11 + i * 0.045,  # 确保colorbar左侧与子图左侧对齐
                        axes[8 + i].get_position().y0 - 0.08,  # 放在子图下方，距离可以适当加大，避免重叠
                        axes[8 + i].get_position().width + 0.06,  # 宽度与子图宽度一致
                        0.01,  # 高度设为0.02，适合作为横向colorbar
                    ]
                )

                # 创建colorbar（确保colorbar的范围与原始数据的最小最大值一致）
                fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

            # Save the figure
            filename = (
                f"{os.path.basename(out_dir)}_{approach}_{dataset_}_epoch_{best_states[idx_data]}_{idx}_together.png"
            )
            path = os.path.join(out_dir, filename)
            plt.tight_layout()  # Adjust layout to avoid overlap
            plt.savefig(path)
            plt.close(fig)
            # =========================================================================================================
            # Use a 6x4 grid (1 rows, 5 columns)
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
            axes = axes.flatten()  # Flatten axes for easier iteration

            axes[0].imshow(image, cmap="gray")
            axes[0].imshow(target, cmap="gray", alpha=0.5)
            axes[0].axis("off")
            for i in range(4):
                # 提取不确定性mask
                uncertain_mask = sample_data["multi_uncertainty_mask"][i][idx].squeeze().detach().cpu().numpy()

                # 显示image和不确定性mask，确保image在底层，uncertain_mask叠加在上面
                axes[1 + i].imshow(image, cmap="gray", alpha=1.0)  # 原图完全不透明

                # 设置透明背景，值为0的部分完全透明，其他区域半透明
                mask_alpha = np.zeros_like(uncertain_mask)
                mask_alpha[uncertain_mask > 0] = 0.8  # 非零区域设置半透明

                # 创建Normalize对象进行颜色归一化（仅影响颜色映射）
                norm = Normalize(vmin=uncertain_mask.min(), vmax=uncertain_mask.max())

                # 将不确定性mask转换为RGBA格式，影响的是颜色映射
                rgba_uncertainty = plt.cm.viridis(uncertain_mask, bytes=True)  # 使用viridis colormap，不会自动归一化
                rgba_uncertainty[..., -1] = (mask_alpha * 255).astype(int)  # 将alpha通道与透明度结合（0-255范围）

                # 显示不确定性mask，并应用颜色归一化
                im = axes[1 + i].imshow(rgba_uncertainty, cmap="viridis", norm=norm)

                axes[1 + i].axis("off")

                # 为不确定性mask添加横向colorbar
                cbar_ax = fig.add_axes(
                    [
                        axes[i + 1].get_position().x0 - 0.07 + i * 0.038,  # 确保colorbar左侧与子图左侧对齐
                        axes[1 + i].get_position().y0 - 0.16,  # 放在子图下方，距离可以适当加大，避免重叠
                        axes[1 + i].get_position().width + 0.04,  # 宽度与子图宽度一致
                        0.01,  # 高度设为0.02，适合作为横向colorbar
                    ]
                )

                # 创建colorbar（确保colorbar的范围与原始数据的最小最大值一致）
                fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

            # Save the figure
            filename = (
                f"{os.path.basename(out_dir)}_{approach}_{dataset_}_epoch_{best_states[idx_data]}_{idx}_uncertainty.png"
            )
            path = os.path.join(out_dir, filename)
            plt.tight_layout()  # Adjust layout to avoid overlap
            plt.savefig(path)
            plt.close(fig)

        return


def main():
    from mp.paths import abs_path, storage_path

    config = parse_args_as_dict(sys.argv[1:])
    seed_all(42)

    # datasets_ = ["hippocampus"]
    # datasets_ = ["prostate"]
    datasets_ = ["polyp"]
    approaches = ["vma"]
    out_dir = os.path.join(abs_path, storage_path, "image_vma")
    target = "i"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dataset_ in datasets_:
        for approach in approaches:
            config["experiment_name"] = f"{dataset_}-{approach}"
            config["target_class"] = target
            config["approach"] = approach
            config["dataset"] = dataset_
            config["resume_epoch"] = 40
            config["device-ids"] = "0"
            config["device"] = "cuda:0"
            print(f"Running experiment with config: {config}")
            process_experiment(config, approach, dataset_, out_dir)


if __name__ == "__main__":
    main()
