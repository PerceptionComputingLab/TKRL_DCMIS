import os
from typing import OrderedDict
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


def plot_first_task_curve(table_dict, out_dir, dataset):
    """
    Plot the curve of the first dataset's performance over time for each approach.

    Args:
        table_dict (dict): Dictionary containing the curves for each approach.
        out_dir (str): Directory to save the plots.
        dataset (str): Dataset name.
    """
    plt.figure()
    plt.title(f"First Task Performance Curve ({dataset})")
    for key, values in table_dict.items():
        plt.plot(range(len(values["avg_curve"])), values["avg_curve"], label=key)
    plt.xlabel("Tasks")
    plt.ylabel("Performance")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{dataset}_first_task_curve.png"))
    plt.close()


def load_csv_data(file_path):
    """
    Load data from a CSV file and return it as a numpy array.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        numpy.ndarray: Extracted data from the CSV file.
    """
    data = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if "test" in row[0]:
                continue
            data.append(np.array([float(r) for r in row]))
    return np.array(data)


def compute_metrics(result, up_bound):
    """
    Compute avg, ravg, bwt, and rbwt metrics and their curves.

    Args:
        result (numpy.ndarray): Result data for the approach.
        up_bound (numpy.ndarray): Upper bound data for the dataset.

    Returns:
        dict: A dictionary containing the computed metrics and their curves.
    """
    avg = np.average(result[-1, :])
    avg_curve = [np.average(result[i, : i + 1]) for i in range(len(up_bound[0]))]
    # print(np.average(up_bound[0]))

    ravg = np.average(result[-1, :] / up_bound[0])
    ravg_curve = [np.average(result[i, : i + 1] / up_bound[0, : i + 1]) for i in range(len(up_bound[0]))]

    bwt_curve = []
    rbwt_curve = []
    for j in range(1, len(up_bound[0])):
        bwt = np.mean([result[j, i] - result[i, i] for i in range(j)])
        rbwt = np.mean([(result[j, i] - result[i, i]) / result[i, i] for i in range(j)])
        bwt_curve.append(bwt)
        rbwt_curve.append(rbwt)

    bwt = np.mean([result[-1, i] - result[i, i] for i in range(len(up_bound[0]) - 1)])
    rbwt = np.mean([(result[-1, i] - result[i, i]) / result[i, i] for i in range(len(up_bound[0]) - 1)])

    return {
        "avg": avg,
        "ravg": ravg,
        "bwt": bwt,
        "rbwt": rbwt,
        "avg_curve": avg_curve,
        "ravg_curve": ravg_curve,
        "bwt_curve": bwt_curve,
        "rbwt_curve": rbwt_curve,
    }


def plot_curves(metric_name, table_dict, out_dir, dataset):
    """
    Plot metric curves for each approach and save the figures.

    Args:
        metric_name (str): The name of the metric to plot.
        table_dict (dict): Dictionary containing metric data for each approach.
        out_dir (str): Directory to save the plots.
        dataset (str): The dataset name.
    """
    plt.figure()
    plt.title(f"{metric_name} Curve")
    for key, values in table_dict.items():
        plt.plot(range(len(values[metric_name])), values[metric_name], label=key)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{dataset}_{metric_name}.png"))
    plt.close()


def integrate_csv_results(file_dirs, dataset, metric, out_dir):
    """
    Integrate CSV results from multiple approaches into a single CSV file.

    Args:
        file_dirs (str): Base directory where result files are stored.
        dataset (str): The name of the dataset (e.g., 'hippocampus').
        metric (str): The metric to integrate (e.g., 'Mean_ScoreDice').
        out_dir (str): Directory to save the integrated CSV.
    """
    combined_data = None

    # Process each approach
    for name in os.listdir(file_dirs):
        if dataset not in name or "joint" in name:
            continue

        result_path = os.path.join(file_dirs, name, "0", "results", f"result_{metric}.csv")
        if not os.path.exists(result_path):
            continue

        # Load the result from CSV
        result_data = load_csv_data(result_path)
        approach_name = name.split("-")[-1]  # Extract the approach name

        # Convert the numpy array to DataFrame for easy column naming
        result_df = pd.DataFrame(result_data)
        result_df.columns = [f"{approach_name}_task_{i+1}" for i in range(result_df.shape[1])]

        # Combine the result data with previous results
        if combined_data is None:
            combined_data = result_df
        else:
            combined_data = pd.concat([combined_data, result_df], axis=1)

    # Save the combined results to a CSV file
    output_file = os.path.join(out_dir, f"{dataset}_integrated_results.csv")
    combined_data.to_csv(output_file, index=False)
    print(f"Integrated results saved to {output_file}")


def main(dataset="prostate", metric="Mean_ScoreDice"):

    from mp.paths import abs_path, storage_path

    file_dirs = os.path.join(abs_path, storage_path, "exp")
    out_dir = os.path.join(abs_path, storage_path, "figure")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    table_dict = {}

    # Load the upper bound data for comparison
    up_bound = load_csv_data(os.path.join(file_dirs, f"{dataset}-joint", "0", "results", f"result_{metric}.csv"))

    # Process each approach and compute metrics
    for name in os.listdir(file_dirs):
        if dataset not in name or "joint" in name:
            continue

        result_path = os.path.join(file_dirs, name, "0", "results", f"result_{metric}.csv")
        if not os.path.exists(result_path):
            continue

        result = load_csv_data(result_path)

        metrics = compute_metrics(result, up_bound)
        approach_name = name.split("-")[1]
        table_dict[approach_name] = metrics

    # SORT table_dict BY [ seq, mas ,ewc, kd, mib, plop, ted, tkrl]
    # table_dict = OrderedDict(
    #     sorted(
    #         table_dict.items(),
    #         key=lambda x: ["seq", "mas", "ewc", "kd", "mib", "plop", "ted", "pcd", "vma", "tkrl"].index(x[0]),
    #     )
    # )
    # Print results in table format
    print(f"{'Approach'.rjust(16)} {'avg'.rjust(16)} {'ravg'.rjust(16)} {'bwt'.rjust(16)} {'rbwt'.rjust(16)}")

    table_dict = dict(sorted(table_dict.items()))
    for approach, values in table_dict.items():
        print(
            f"{approach.rjust(16)} {str(round(values['avg'], 4)).rjust(16)} "
            f"{str(round(values['ravg'], 4)).rjust(16)} {str(round(values['bwt'], 4)).rjust(16)} "
            f"{str(round(values['rbwt'], 4)).rjust(16)}"
        )

    # Save results to a CSV file
    pd.DataFrame(table_dict).to_csv(os.path.join(out_dir, f"{dataset}_results.csv"), index=False)

    # Plot the curves for avg, ravg, bwt, and rbwt
    for metric_name in ["avg_curve", "ravg_curve", "bwt_curve", "rbwt_curve"]:
        plot_curves(metric_name, table_dict, out_dir, dataset)

    # Integrate all original CSV results into one file
    integrate_csv_results(file_dirs, dataset, metric, out_dir)


if __name__ == "__main__":

    metric = "Mean_ScoreDice"  # Mean_ScoreDice, Mean_ScoreIoU
    datasets = ["prostate", "hippocampus", "polyp", "optici", "optico", "mmi", "mmo", "mmr"]
    # datasets = ["prostate", "hippocampus", "polyp", ]
    # datasets = ["mmi", "mmo", "mmr"]
    for dataset in datasets:
        main(dataset=dataset, metric=metric)
