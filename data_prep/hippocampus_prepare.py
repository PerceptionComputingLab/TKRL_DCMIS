import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import mp.data.datasets.dataset_utils as du
from mp.utils.load_restore import join_path
from scipy.ndimage import zoom
import re
import matplotlib.pyplot as plt
from mp.paths import storage_data_path, source_data_path


def resize_volume(img, gt, target_shape=(None, 192, 192)):
    """
    Resize the volume of the image and ground truth to the target shape using interpolation.

    Args:
        img (numpy.ndarray): Image volume data.
        gt (numpy.ndarray): Ground truth segmentation data.
        target_shape (tuple): Target shape for resizing.

    Returns:
        tuple: Resized image and ground truth volumes.
    """
    zoom_factors = [1, target_shape[1] / img.shape[1], target_shape[2] / img.shape[2]]
    resized_img = zoom(img, zoom_factors, order=1)  # Linear interpolation for image
    resized_gt = zoom(gt, zoom_factors, order=0)  # Nearest neighbor for mask to preserve labels
    return resized_img, resized_gt


def rescale_slices(img, gt):
    """
    Rescale each slice of the image data to the range [0.001, 0.99].

    Args:
        img (numpy.ndarray): Image data.
        gt (numpy.ndarray): Ground truth segmentation data.

    Returns:
        tuple: Rescaled image data and filtered ground truth.
    """
    valid_slices = [i for i in range(len(gt)) if np.max(gt[i]) > 0]
    img = img[valid_slices]
    gt = gt[valid_slices]

    image_rescaled = np.zeros_like(img)
    for i in range(len(img)):
        slice_min, slice_max = np.min(img[i]), np.max(img[i])
        if slice_max > slice_min:  # Avoid division by zero
            image_rescaled[i] = np.clip((img[i] - slice_min) / (slice_max - slice_min), 0.001, 0.99)
        else:
            image_rescaled[i] = img[i]  # Preserve slice if min == max

    return image_rescaled, gt


def dryad_extract_images(source_path, target_path, merge_labels, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(os.path.join(target_path))

    # Patient folders s01, s02, ...
    for patient_folder in filter(lambda s: re.match(r"^s[0-9]+.*", s), os.listdir(source_path)):

        # Loading the image
        image_path = os.path.join(
            source_path,
            patient_folder,
            f"{patient_folder}_{subset['Modality'].lower()}_" f"{subset['Resolution'].lower()}_defaced_MNI.nii.gz",
        )
        x = sitk.ReadImage(image_path)
        x = sitk.GetArrayFromImage(x)

        # For each MRI, there are 2 segmentation (left and right hippocampus)
        for side in ["L", "R"]:
            # Loading the label
            label_path = os.path.join(
                source_path,
                patient_folder,
                f"{patient_folder}_hippolabels_"
                f"{'hres' if subset['Resolution'] == 'Hires' else 't1w_standard'}"
                f"_{side}_MNI.nii.gz",
            )

            y = sitk.ReadImage(label_path)
            y = sitk.GetArrayFromImage(y)

            # We need to recover the study name of the image name to construct the name of the segmentation files
            study_name = f"{patient_folder}_{side}"

            # Average label shape (T1w, standard): (37.0, 36.3, 26.7)
            # Average label shape (T1w, hires): (94.1, 92.1, 68.5)
            # Average label shape (T2w, hires): (94.1, 92.1, 68.5)
            assert x.shape == y.shape

            # Disclaimer: next part is ugly and not many checks are made

            # So we first compute the bounding box
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(y)

            # Compute the start idx for each dim
            dr = (rmax - rmin) // 4
            dc = (cmax - cmin) // 4
            dz = (zmax - zmin) // 4

            # Reshaping
            y = y[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            if merge_labels:
                y[y > 1] = 1

            x_cropped = x[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            x_cropped, y = resize_volume(x_cropped, y)
            x_cropped, y = rescale_slices(x_cropped, y)

            # Save new images so they can be loaded directly
            np.savez_compressed(join_path([target_path, study_name + ".npz"]), label=y, image=x_cropped)


def decathlon_extract_images(source_path, target_path, merge_labels):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    images_path = os.path.join(source_path, "imagesTr")
    labels_path = os.path.join(source_path, "labelsTr")

    # Filenames have the form 'hippocampus_XX.nii.gz'
    filenames = [x for x in os.listdir(images_path) if x[:5] == "hippo"]

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for filename in filenames:

        # Extract only T2-weighted
        x = sitk.ReadImage(os.path.join(images_path, filename))
        x = sitk.GetArrayFromImage(x)
        y = sitk.ReadImage(os.path.join(labels_path, filename))
        y = sitk.GetArrayFromImage(y)

        # Shape expected: (35, 51, 35)
        # Average label shape: (24.5, 37.8, 21.0)
        assert x.shape == y.shape

        # No longer distinguish between hippocampus proper and subiculum
        if merge_labels:
            y[y == 2] = 1

        # Save new images so they can be loaded directly
        study_name = filename.replace("_", "").split(".nii")[0]
        # resize to None*192*192

        x, y = resize_volume(x, y)
        x, y = rescale_slices(x, y)

        # Save new images so they can be loaded directly
        np.savez_compressed(join_path([target_path, study_name + ".npz"]), label=y, image=x)


def harp_extract_images(source_path, target_path, subset):
    r"""Extracts images, merges mask labels (if specified) and saves the
    modified images.
    """

    def bbox_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    # Folder 100 is for training (100 subjects), 35 subjects are left over for validation
    affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    images_path = os.path.join(source_path, subset)
    labels_path = os.path.join(source_path, f"Labels_{subset}_NIFTI")

    # Create directories
    if not os.path.exists(target_path):
        os.makedirs(os.path.join(target_path))

    # For each MRI, there are 2 segmentation (left and right hippocampus)
    for filename in os.listdir(images_path):
        # Loading the .mnc file and converting it to a .nii.gz file
        minc = nib.load(os.path.join(images_path, filename))
        x = nib.Nifti1Image(minc.get_fdata(), affine=affine)

        # We need to recover the study name of the image name to construct the name of the segmentation files
        match = re.match(r"ADNI_[0-9]+_S_[0-9]+_[0-9]+", filename)
        if match is None:
            raise Exception(f"A file ({filename}) does not match the expected file naming format")

        # For each side of the brain
        for side in ["_L", "_R"]:
            study_name = match[0] + side

            y = sitk.ReadImage(os.path.join(labels_path, study_name + ".nii"))
            y = sitk.GetArrayFromImage(y)

            # Shape expected: (189, 233, 197)
            # Average label shape (Training): (27.1, 36.7, 22.0)
            # Average label shape (Validation): (27.7, 35.2, 21.8)
            assert x.shape == y.shape
            # Disclaimer: next part is ugly and not many checks are made
            # BUGFIX: Some segmentation have some weird values eg {26896.988, 26897.988} instead of {0, 1}
            y = (y - np.min(y.flat)).astype(np.uint8)

            # So we first compute the bounding box
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(y)

            # Compute the start idx for each dim
            dr = (rmax - rmin) // 4
            dc = (cmax - cmin) // 4
            dz = (zmax - zmin) // 4

            # Reshaping
            y = y[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            x_cropped = x.get_fdata()[rmin - dr : rmax + dr, cmin - dc : cmax + dc, zmin - dz : zmax + dz]

            x_cropped, y = resize_volume(x_cropped, y)
            x_cropped, y = rescale_slices(x_cropped, y)

            # Save new images so they can be loaded directly
            np.savez_compressed(join_path([target_path, study_name + ".npz"]), label=y, image=x_cropped)


def main():

    # Processing HarP dataset
    global_name = "HarP"
    print(global_name)
    original_data_path = os.path.join(source_data_path, "continual hippocampus/HarP")
    harp_subsets = [("100", "Training"), ("35", "Validation")]
    for orig_folder, _ in harp_subsets:
        harp_extract_images(original_data_path, os.path.join(storage_data_path, global_name), orig_folder)

    # Processing DecathlonHippocampus dataset
    global_name = "DecathlonHippocampus"
    print(global_name)
    original_data_path = os.path.join(source_data_path, "continual hippocampus/DecathonHippocampus")
    decathlon_extract_images(original_data_path, os.path.join(storage_data_path, global_name), merge_labels=True)

    # Processing DryadHippocampus dataset
    global_name = "DryadHippocampus"
    print(global_name)
    default_subset = {"Modality": "T1w", "Resolution": "Standard"}
    original_data_path = os.path.join(source_data_path, "continual hippocampus/DryadHippocampus")
    dryad_extract_images(original_data_path, os.path.join(storage_data_path, global_name), True, default_subset)


if __name__ == "__main__":
    main()
