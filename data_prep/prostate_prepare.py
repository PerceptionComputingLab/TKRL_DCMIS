import os
import shutil
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from mp.paths import storage_data_path, source_data_path


def resize_volume(img, gt, target_shape=(None, 192, 192)):
    """
    Resize the volume of image and ground truth to the target shape using interpolation.

    Args:
        img (numpy.ndarray): The image volume data.
        gt (numpy.ndarray): The ground truth segmentation data.
        target_shape (tuple): The desired shape for resizing.

    Returns:
        tuple: Resized image and ground truth volumes.
    """
    zoom_factors = [
        1,  # Keep the depth the same
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    ]

    # Resize image using linear interpolation and ground truth using nearest neighbor interpolation
    resized_img = zoom(img, zoom_factors, order=1)
    resized_gt = zoom(gt, zoom_factors, order=0)

    return resized_img, resized_gt


def rescale_slices(img, gt):
    """
    Rescale each slice of the image data to the range [0.001, 0.99].

    Args:
        img (numpy.ndarray): The image data.
        gt (numpy.ndarray): The ground truth segmentation data.

    Returns:
        tuple: Rescaled image data and filtered ground truth.
    """
    valid_slices = [i for i in range(len(gt)) if np.max(gt[i]) > 0]
    img = img[valid_slices]
    gt = gt[valid_slices]

    image_rescaled = np.zeros_like(img)
    for i in range(len(img)):
        slice_min = np.min(img[i])
        slice_max = np.max(img[i])
        if slice_max > slice_min:  # Avoid division by zero
            image_rescaled[i] = np.clip((img[i] - slice_min) / (slice_max - slice_min), 0.001, 0.99)
        else:
            image_rescaled[i] = img[i]  # If min == max, retain the original slice

    return image_rescaled, gt


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  #
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)  #
    resampler.SetReferenceImage(itkimage)  #
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  #
    return itkimgResampled


def process_image_and_mask(source_path, name):
    """
    Load and process image and mask, including resizing and rescaling.

    Args:
        source_path (str): Path to the source directory.
        name (str): File name of the segmentation or mask.

    Returns:
        tuple: Processed image array, mask array, and updated file name.
    """
    mask = sitk.ReadImage(os.path.join(source_path, name))
    mask_array = sitk.GetArrayFromImage(mask)

    # Update mask values and file name based on naming conventions
    if "segmentation" in name.lower():
        name = name.replace("segmentation", "gt")
        name = name.replace("Segmentation", "gt")
        mask_array[mask_array > 0.0] = 1
    else:
        return None, None, None  # Skip non-segmentation files

    image = sitk.ReadImage(os.path.join(source_path, name.replace("_gt", "")))
    image_array = sitk.GetArrayFromImage(image)

    # Resize and rescale the image and mask arrays
    image_array, mask_array = resize_volume(image_array, mask_array)
    image_array, mask_array = rescale_slices(image_array, mask_array)

    return image_array, mask_array, name


def process_subset(datapath, data_out, subset):
    """
    Process all images and masks within a given subset.

    Args:
        datapath (str): Path to the data directory.
        data_out (str): Path to the output directory.
        subset (str): Subset name to be processed.
    """
    source_path = os.path.join(datapath, subset)
    target_path = os.path.join(data_out, subset)
    os.makedirs(target_path, exist_ok=True)

    for name in os.listdir(source_path):
        image_array, mask_array, new_name = process_image_and_mask(source_path, name)
        if image_array is None or mask_array is None:
            continue

        np.savez_compressed(
            os.path.join(target_path, new_name.replace("_gt.nii.gz", ".npz")),
            label=mask_array,
            image=image_array,
        )

    print(f"{subset} done!")


def main():
    """
    Main function to process multiple subsets of image data.
    """
    print("Start processing...")
    groups = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]
    datapath = os.path.join(source_data_path, "continual prostate")
    data_out = storage_data_path

    for subset in groups:
        process_subset(datapath, data_out, subset)


if __name__ == "__main__":
    main()
