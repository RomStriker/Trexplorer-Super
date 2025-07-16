import os
import re
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    dataset = "atm22"  # "atm22", "parse2022", "syntrx"
    data_dir = ""  # Path to the dataset
    dst_dir = ""  # Destination directory for the organized dataset
    paths_dict = {
        'annots_extra': os.path.join(dst_dir, "annots_extra"),
        'annotst_sep_test': os.path.join(dst_dir, "annotst_sep_test"),
        'annotst_test': os.path.join(dst_dir, "annotst_test"),
        'annotst_train': os.path.join(dst_dir, "annotst_train"),
        'annotst_val': os.path.join(dst_dir, "annotst_val"),
        'annotst_val_sub_vol': os.path.join(dst_dir, "annotst_val_sub_vol"),
        'images_extra': os.path.join(dst_dir, "images_extra"),
        'images_sep_test': os.path.join(dst_dir, "images_sep_test"),
        'images_test': os.path.join(dst_dir, "images_test"),
        'images_train': os.path.join(dst_dir, "images_train"),
        'images_val': os.path.join(dst_dir, "images_val"),
        'images_val_sub_vol': os.path.join(dst_dir, "images_val_sub_vol"),
        'masks_extra': os.path.join(dst_dir, "masks_extra"),
        'masks_sep_test': os.path.join(dst_dir, "masks_sep_test"),
        'masks_test': os.path.join(dst_dir, "masks_test"),
        'masks_train': os.path.join(dst_dir, "masks_train"),
        'masks_val': os.path.join(dst_dir, "masks_val"),
        'masks_val_sub_vol': os.path.join(dst_dir, "masks_val_sub_vol")}

    os.makedirs(dst_dir, exist_ok=True)
    for key in paths_dict:
        os.makedirs(paths_dict[key], exist_ok=True)

    img_dir = os.path.join(data_dir, "imagesTrRs")
    mask_dir = os.path.join(data_dir, "labelsTrRs")
    annot_dir = os.path.join(data_dir, "centerlines_kimimaro_em_4.0")
    annot_files = sorted(os.listdir(annot_dir))
    img_files = [os.path.join(img_dir, os.path.splitext(file)[0]+'.nii.gz') for file in annot_files]
    mask_files = [os.path.join(mask_dir, os.path.splitext(file)[0]+'.nii.gz') for file in annot_files]
    annot_files = [os.path.join(annot_dir, file) for file in annot_files]

    if dataset == "atm22":
        num_train_samples = 220
        num_val_samples = 16
        num_test_samples = 60
    elif dataset == "parse2022":
        num_train_samples = 72
        num_val_samples = 8
        num_test_samples = 20
    elif dataset == "syntrx":
        num_train_samples = 368
        num_val_samples = 32
        num_test_samples = 100

    # move the first num_train_samples samples to the training set
    start = 0
    end = start + num_train_samples
    for img_file, mask_file, annot_file in tqdm(zip(img_files[start:end], mask_files[start:end], annot_files[start:end])):
        idx = str(int(re.compile(r'\d+').findall(img_file)[1]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_train'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_train'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annotst_train'], idx + '.pickle'))

    # move the next num_val_samples samples to the validation set
    start = num_train_samples
    end = start + num_val_samples
    for img_file, mask_file, annot_file in tqdm(zip(img_files[start:end], mask_files[start:end], annot_files[start:end])):
        idx = str(int(re.compile(r'\d+').findall(img_file)[1]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_val'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_val'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annotst_val'], idx + '.pickle'))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annotst_val_sub_vol'], idx + '.pickle'))

    # move the next num_test_samples samples to the test set
    start = num_train_samples + num_val_samples
    end = start + num_test_samples
    for img_file, mask_file, annot_file in tqdm(zip(img_files[start:end], mask_files[start:end], annot_files[start:end])):
        idx = str(int(re.compile(r'\d+').findall(img_file)[1]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_test'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_test'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annotst_test'], idx + '.pickle'))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_sep_test'], idx + '-0.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_sep_test'], idx + '-0.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annotst_sep_test'], idx + '-0.pickle'))



