# Medical Image Segmentation Preprocessing Instructions

This repository contains scripts and guidelines for preparing datasets used in segmentation tasks across different anatomical regions and modalities. The datasets have been categorized according to their specific tasks: prostate segmentation, hippocampus segmentation, cardiac segmentation, and fundus segmentation. The preprocessing steps are necessary to standardize the datasets, ensuring compatibility with the segmentation models.

## Prostate Segmentation (6 Domains)

### Dataset Information

- [Prostate Dataset](https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view): Contains multi-domain MRI images used for prostate segmentation.

### Preprocessing Steps

1. **Intensity Normalization:** Rescale the intensity values to the range [0, 1] to standardize the image data.
2. **Dataset Renaming:** Rename the dataset 'BMC' folder from 'Seg' to 'seg' for consistency in naming conventions.
3. **Relabeling:** Convert the segmentation labels to binary format for datasets 'RUNMC' and 'BMC' to simplify the classification task.

### Command to Execute Preprocessing

Run the following command to apply the preprocessing steps:

```
python prostate_prepare.py
```

## Hippocampus Segmentation (3 Domains)

### Dataset Information

- [HarP - Standard Operating Protocol](http://www.hippocampal-protocol.net/SOPs/labels.php#final) and [HarP Index](http://www.hippocampal-protocol.net/SOPs/index.php): These datasets follow specific protocols for hippocampus segmentation.
- [Decathlon Hippocampus](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2): A dataset from the Medical Decathlon Challenge.
- [Dryad Hippocampus](https://datadryad.org/stash/dataset/doi:10.5061/dryad.gc72v): Contains curated hippocampus data available for segmentation tasks.

### Preprocessing Steps

1. **Intensity Normalization:** Rescale the intensity values to the range [0, 1].
2. **VOI Cropping:** Crop the volume of interest (VOI) to focus on the hippocampal regions, reducing computational complexity.
3. **Label Merging:** Merge segmentation labels into a unified format for consistency across datasets.

### Command to Execute Preprocessing

Run the following command to prepare the hippocampus datasets:

```
python hippocampus_prepare.py
```

## Cardiac Segmentation (LV-endo, LV-epi, RV) - 4 Domains

### Dataset Information

- [M&M Dataset](https://www.ub.edu/mnms/): Includes cardiac MRI images used for segmenting the left ventricle (LV-endo, LV-epi) and right ventricle (RV).

### Preprocessing Steps

1. **Intensity Normalization:** Rescale the intensity values to the range [0, 1].
2. **Domain Grouping:** Group the data by 'VendorName' into four domains to accommodate different data sources and imaging conditions.

### Command to Execute Preprocessing

Run the following command to apply the preprocessing steps for cardiac segmentation:

```
python cardiacmm_prepare.py
```

## Fundus Segmentation (Optic Cup and Optic Disc) - 4 Domains

### Dataset Information

- [Fundus Dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view?usp=sharing): Includes fundus images used for segmenting the optic cup and optic disc.

### Preprocessing Steps

1. **Intensity Normalization:** Rescale the intensity values to the range [0, 1].
2. **Center Cropping:** Crop the image to a size of 800x800 pixels to focus on the region of interest.
3. **Resizing:** Downsize the cropped image to 192x192 pixels for efficient model training.
4. **Binary Mask Conversion:** Convert the segmentation mask to binary format to simplify the label space.

### Command to Execute Preprocessing

Run the following command to prepare the fundus dataset:

```
python optic_prepare.py
```

# Notes

- Ensure that all dependencies are installed before running the scripts.
- Modify the paths in the scripts as necessary to point to your local dataset locations.
- The preprocessing steps are critical to ensure that all datasets are in a consistent format, which is essential for model training and evaluation.

This README aims to provide a clear, step-by-step guide for preprocessing each dataset to streamline your segmentation tasks. Feel free to modify or extend these instructions based on your specific requirements.
