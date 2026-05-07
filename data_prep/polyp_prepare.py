import os
import numpy as np
import cv2
from mp.paths import storage_data_path, source_data_path


def process_mask(mask_path, hw=192):
    """
    Process the mask image by converting its pixel values and resizing.

    Args:
        mask_path (str): Path to the mask image.
        hw (int): The target size for resizing the mask.

    Returns:
        numpy.ndarray: Processed mask image.
    """
    old_mask = cv2.imread(mask_path)
    mask = np.ones(old_mask.shape[:2], dtype=np.uint8)
    mask[old_mask[:, :, 0] == 255] = 1
    mask[old_mask[:, :, 0] == 0] = 0
    mask = cv2.resize(mask, (hw, hw), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(mask, axis=0)


def process_image(image_path, hw=192):
    """
    Process the image by converting to grayscale and resizing.

    Args:
        image_path (str): Path to the image file.
        hw (int): The target size for resizing the image.

    Returns:
        numpy.ndarray: Processed image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (hw, hw), interpolation=cv2.INTER_NEAREST)
    image = np.expand_dims(image, axis=0)
    return image / 255.0  # Normalize image to [0, 1]


def process_dataset(datapath, group, new_dir, hw=192):
    """
    Process a dataset (train or test) by converting masks and images.

    Args:
        datapath (str): Path to the dataset directory.
        group (str): Name of the group (sub-directory).
        new_dir (str): Path to the output directory.
        hw (int): The target size for resizing the images.
    """
    image_dir = os.path.join(datapath, group, "images_" + group)
    mask_dir = os.path.join(datapath, group, "masks_" + group)

    for patient in os.listdir(image_dir):
        image_path = os.path.join(image_dir, patient)
        mask_path = os.path.join(mask_dir, patient.replace(".jpg", "_mask.jpg"))

        # Process mask and image
        mask = process_mask(mask_path, hw)
        image = process_image(image_path, hw)

        target = os.path.join(new_dir, patient.replace(".jpg", ".npz"))
        np.savez_compressed(target, label=mask, image=image)


def main():
    """
    Main function to process fundus image datasets.
    """
    datapath = os.path.join(source_data_path, "PolypGen")
    data_out = storage_data_path
    hw = 192  # Target size for resizing

    for group in os.listdir(datapath):
        if not os.path.isdir(os.path.join(datapath, group)):
            continue
        print(f"Processing group: {group}")

        new_dir = os.path.join(data_out, group)
        os.makedirs(new_dir, exist_ok=True)

        # Process both train and test datasets
        process_dataset(datapath, group, new_dir, hw)

        print(f"Group {group} done!")


if __name__ == "__main__":
    main()
