import argparse
import os
import sys

import cv2

from preprocess import grayscale, median_filter, gaussian_filter
from segmentation import show_histogram, seg
from chaincode import get_chaincode
from grouping import group_images, save_groups

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Group images of single objects by shape similarity."
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing the .png / .jpg images to process",
    )
    return parser.parse_args()


def list_images(folder):
    if not os.path.isdir(folder):
        print("Error: '" + folder + "' is not a valid folder.")
        sys.exit(1)
    files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not files:
        print("Error: no .png or .jpg images found in '" + folder + "'.")
        sys.exit(1)
    return files


def main():
    args = parse_args()
    input_folder = args.input_folder
    image_files = list_images(input_folder)

    # segmentation.show_histogram saves its plots here
    os.makedirs("outputs", exist_ok=True)

    names = []
    codes = []

    for file in image_files:
        # 1 read image
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        if image is None:
            print("Skipped (could not read):", file)
            continue
        name, extension = os.path.splitext(file)

        # 2 convert to grey scale
        gray_img = grayscale(image)

        # 3 apply median filter to remove outliers
        median_img = median_filter(gray_img, 11)

        # 4 apply gaussian filter to remove gaussian noise
        gaussian_img = gaussian_filter(median_img, 13, 2)

        # 5 get histogram threshold for segmentation
        t = show_histogram(median_img, file)

        # 6 segmentation
        output = seg(median_img, t)

        # 7 chain code
        code = get_chaincode(output)

        names.append(name)
        codes.append(code)

    # 8 grouping
    THRESHOLD = 8
    groups = group_images(names, codes, THRESHOLD)
    save_groups(groups, "output/groups.txt")


if __name__ == "__main__":
    main()