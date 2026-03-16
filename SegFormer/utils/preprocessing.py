import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

import config

# =====================
# CONFIG
# =====================
IMG_SIZE = config.IMG_SIZE
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

SEED = config.SEED
random.seed(SEED)

# =====================
# CREATE FOLDERS
# =====================
def create_dirs(out_dir):
    for split in ["train", "val", "test"]:
        for sub in ["im1", "im2", "labels"]:
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)

# =====================
# LOAD FILENAMES
# =====================
def get_filenames(raw_dir):
    im1_dir = os.path.join(raw_dir, "im1")
    files = sorted(os.listdir(im1_dir))
    return files

# =====================
# SPLIT DATA
# =====================
def split_data(files):
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    return train_files, val_files, test_files

# =====================
# CREATE CHANGE LABEL
# =====================
def create_change_label(label1, label2):
    # semantic change logic
    change = np.where(label1 == label2, 0, label2)
    return change.astype(np.uint8)

# =====================
# PROCESS ONE FILE
# =====================
def process_and_save(file, split, raw_dir, out_dir, img_size):
    path_im1 = os.path.join(raw_dir, "im1", file)
    path_im2 = os.path.join(raw_dir, "im2", file)
    path_l1 = os.path.join(raw_dir, "label1", file)
    path_l2 = os.path.join(raw_dir, "label2", file)

    im1 = cv2.imread(path_im1)
    im2 = cv2.imread(path_im2)
    l1 = cv2.imread(path_l1, 0)
    l2 = cv2.imread(path_l2, 0)

    # Resize
    im1 = cv2.resize(im1, (img_size, img_size))
    im2 = cv2.resize(im2, (img_size, img_size))
    l1 = cv2.resize(l1, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    l2 = cv2.resize(l2, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # Create change label
    change_label = create_change_label(l1, l2)

    # Save
    cv2.imwrite(os.path.join(out_dir, split, "im1", file), im1)
    cv2.imwrite(os.path.join(out_dir, split, "im2", file), im2)
    cv2.imwrite(os.path.join(out_dir, split, "labels", file), change_label)

# =====================
# MAIN PIPELINE
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=str(config.RAW_DIR))
    parser.add_argument("--out_dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--test_split", type=float, default=TEST_SPLIT)
    args = parser.parse_args()

    global TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
    TRAIN_SPLIT = args.train_split
    VAL_SPLIT = args.val_split
    TEST_SPLIT = args.test_split

    create_dirs(args.out_dir)

    files = get_filenames(args.raw_dir)
    train_files, val_files, test_files = split_data(files)

    print("Processing Train...")
    for f in tqdm(train_files):
        process_and_save(f, "train", args.raw_dir, args.out_dir, args.img_size)

    print("Processing Val...")
    for f in tqdm(val_files):
        process_and_save(f, "val", args.raw_dir, args.out_dir, args.img_size)

    print("Processing Test...")
    for f in tqdm(test_files):
        process_and_save(f, "test", args.raw_dir, args.out_dir, args.img_size)

    print("Done.")

if __name__ == "__main__":
    main()