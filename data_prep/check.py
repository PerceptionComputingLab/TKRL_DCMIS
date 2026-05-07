import os
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_image(img, gt, dataset, sample, save_path):
    """
    Plot and save the sample image with its ground truth overlay.

    Args:
        img (numpy.ndarray): The image data.
        gt (numpy.ndarray): The ground truth mask.
        dataset (str): The name of the dataset.
        sample (str): The sample file name.
        save_path (str): The directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth Overlay")
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, cmap="jet", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{dataset}_{sample}.png"))
    plt.close()  # Close the plot to free memory


def load_sample_data(sample_file):
    """
    Load image and ground truth data from an .npz file.

    Args:
        sample_file (str): Path to the .npz file.

    Returns:
        tuple: Image and ground truth arrays.
    """
    sample_dict = np.load(sample_file)
    img = sample_dict["image"]
    gt = sample_dict["label"]

    # If the image has a channel dimension, take the first channel
    if len(img.shape) == 3:
        img = img[0, :, :]
        gt = gt[0, :, :]

    return img, gt


def main(data_dir, out_dir):
    """
    Main function to process datasets, extract samples, and save plots.

    Args:
        data_dir (str): The directory containing the datasets.
        out_dir (str): The directory where output plots will be saved.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dataset in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset)
        samples = [filename for filename in os.listdir(dataset_path) if filename.endswith(".npz")]

        if not samples:  # Skip the dataset if there are no .npz files
            print(f"No .npz files found in dataset: {dataset}")
            continue

        sample_id = np.random.randint(0, len(samples))
        sample = samples[sample_id]

        # Load image and ground truth data
        sample_file = os.path.join(dataset_path, sample)
        print(f"Loading sample data from: {sample_file}")
        img, gt = load_sample_data(sample_file)

        print(f"Dataset: {dataset}; Sample: {sample}; Shape: {img.shape}")

        # Plot and save the image and ground truth overlay
        plot_sample_image(img, gt, dataset, sample, out_dir)


if __name__ == "__main__":
    from mp.paths import abs_path, storage_path
    data_directory = os.path.join(abs_path, storage_path, "data")
    output_directory = os.path.join(abs_path, storage_path, "data_png")
    main(data_directory, output_directory)
