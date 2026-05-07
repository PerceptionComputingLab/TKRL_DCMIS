from mp.utils.load_restore import pkl_load
import os
from mp.visualization.plot_results import plot_results
import pandas as pd


def extract_metric_data(file_dir, metric):
    """
    Extract metric data from the results.pkl file and save it as a CSV file.

    Args:
        file_dir (str): Directory containing the results file.
        metric (str): Metric to be extracted.
    """
    try:
        result = pkl_load("results.pkl", file_dir)
        df = result.to_pandas()

        # Filter the DataFrame to only include the specified metric for "dataset"
        dataset_ = dataset
        if dataset_ is "mm":
            dataset_ = "cardiac"
        df = df[df["Metric"] == f"{metric}[{dataset_}]"]

        # Keep only rows containing "test" in the third column
        df = df[df.iloc[:, 2].str.contains("test", na=False)]

        data_dict = df.groupby(df.iloc[:, 2], sort=False)[df.columns[3]].apply(list).to_dict()

        # Convert to a DataFrame and save as CSV
        output_df = pd.DataFrame(data_dict)
        csv_filename = os.path.join(file_dir, f"result_{metric}.csv")
        output_df.to_csv(csv_filename, index=False)
        print(f"Saved {metric} data to {csv_filename}")

        # Uncomment to save plots if required
        # plot_results(result=df, save_path=file_dir, save_name=f"result_{metric}.png")

    except Exception as e:
        print(f"Error while extracting data for metric {metric} in {file_dir}: {e}")


def extracted_test(file_dir):
    """
    Extract test data for multiple metrics and save them as CSV files.

    Args:
        file_dir (str): Directory containing the results file.
    """
    metrics = [
        "Mean_ScoreDice",
        "Mean_ScoreIoU",
        "Mean_ScoreHausdorff",
        "Std_ScoreDice",
        "Std_ScoreIoU",
        "Std_ScoreHausdorff",
    ]

    for metric in metrics:
        extract_metric_data(file_dir, metric)


def main(dataset="prostate"):
    from mp.paths import abs_path, storage_path

    file_dirs = os.path.join(abs_path, storage_path, "exp")

    approach_path_list = [
        os.path.join(file_dirs, name, "0", "results") for name in os.listdir(file_dirs) if dataset in name
    ]

    for path in approach_path_list:
        print(f"Processing results in: {path}")
        extracted_test(path)


if __name__ == "__main__":
    datasets = ["prostate", "optic", "mm", "hippocampus", "polyp"]
    # datasets = ["mm"]
    for dataset in datasets:
        main(dataset=dataset)
