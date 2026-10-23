import cv2
import os
from preprocess import grayscale, median_filter, gaussian_filter
from segmentation import show_histogram, seg

input_folder = "inputs"
output_folder = "outputs"

image_files = os.listdir(input_folder)

for file in image_files:
    if file.endswith(".png"):
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        print(image.shape)
        name, extension = os.path.splitext(file)
        gray_img = grayscale(image)
        print(gray_img.shape)
        cv2.imwrite(output_folder+"/"+name+"grey.png", gray_img)
        

        median_img=median_filter(gray_img)
        cv2.imwrite(output_folder+"/"+name+"median.png", median_img)

        gaussian_img=gaussian_filter(median_img,5,1)
        cv2.imwrite(output_folder+"/"+name+"gaus.png", gaussian_img)

        t=show_histogram(gaussian_img,file)
        out= seg(gaussian_img,t)
        cv2.imwrite(output_folder+"/"+name+"seg.png", out)

        if image is not None:
            print("Loaded:", file)