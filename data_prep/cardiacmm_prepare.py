import os
import nibabel as nib
import numpy as np
import pandas as pd
from mp.paths import storage_data_path, source_data_path


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def crop_pad_data(img, target_size=(None, 192, 192)):

    for idx, size in enumerate(target_size):
        if size is None:  # Skip dimensions where no specific size is defined
            continue

        if size < img.shape[idx]:
            # crop current dimension
            before = (img.shape[idx] - size) // 2
            after = img.shape[idx] - (img.shape[idx] - size) // 2 - ((img.shape[idx] - size) % 2)
            slicing = [slice(None)] * img.ndim
            slicing[idx] = slice(before, after)
            img = img[tuple(slicing)]
        elif size > img.shape[idx]:
            # pad current dimension
            before = (size - img.shape[idx]) // 2
            after = (size - img.shape[idx]) // 2 + ((size - img.shape[idx]) % 2)
            pad_width = [(0, 0)] * img.ndim
            pad_width[idx] = (before, after)
            img = np.pad(img, pad_width, mode="constant", constant_values=0)

    return img


def rescale_image(img, gt):
    valid_slices = [i for i in range(len(gt)) if np.max(gt[i]) > 0]
    img = img[valid_slices]
    gt = gt[valid_slices]

    image_rescaled = np.zeros_like(img)
    for i in range(img.shape[0]):
        slice_min = np.min(img[i])
        slice_max = np.max(img[i])
        if slice_max > slice_min:  # 确保除数不为零
            image_rescaled[i] = np.clip((img[i] - slice_min) / (slice_max - slice_min), 0.001, 0.99)
        else:
            image_rescaled[i] = img[i]  # 如果最小值等于最大值，保留原始切片

    return image_rescaled, gt


def process_group_data(groups, datapath, data_out):
    """
    Process each data group by loading, cropping, rescaling, and saving image data.

    Args:
        groups (dict): Dictionary of groups with vendor names as keys and image paths as values.
        datapath (str): Path to the input data directory.
        data_out (str): Path to the output data directory.
    """
    for vendor, paths in groups.items():
        vendor_path = os.path.join(data_out, vendor)
        os.makedirs(vendor_path, exist_ok=True)
        print(f"Processing vendor: {vendor}")
        for in_path in paths:
            in_path_full = os.path.join(datapath, in_path)
            if not os.path.exists(in_path_full):
                continue

            for name in os.listdir(in_path_full):
                if "frame" not in name or "gt" in name:
                    continue

                in_path_frame = os.path.join(in_path_full, name)
                gt_path_frame = in_path_frame.replace(".nii.gz", "_gt.nii.gz")

                # print(f"Processing: {name}")
                data, affine, header = load_nii(in_path_frame)
                data = data.transpose(2, 0, 1)
                data = crop_pad_data(data)

                gt, _, _ = load_nii(gt_path_frame)
                gt = gt.transpose(2, 0, 1)
                gt = crop_pad_data(gt).astype(np.uint8)

                data, gt = rescale_image(data, gt)

                output_file = os.path.join(vendor_path, name.replace(".nii.gz", ".npz"))
                np.savez_compressed(output_file, label=gt, image=data)


def main():
    """
    Main function to handle data processing and group-wise image handling.
    """
    datapath = os.path.join(source_data_path, "mm")
    data_out = storage_data_path
    file_name = os.path.join(datapath, "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")

    df = pd.read_csv(file_name)
    groups = df.groupby("VendorName")["External code"].apply(list).to_dict()
    os.makedirs(data_out, exist_ok=True)

    process_group_data(groups, datapath, data_out)


if __name__ == "__main__":
    main()
