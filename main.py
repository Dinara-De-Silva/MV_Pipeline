import cv2
import os
from preprocess import grayscale, median_filter, gaussian_filter
from segmentation import show_histogram, seg, remove_small_components
from chaincode import get_chaincode
from grouping import group_images, save_groups
from chaincode import get_chaincode

input_folder = "inputs"
output_folder = "image_output"

image_files = os.listdir(input_folder)


names = []
codes = []

for file in image_files:
    if file.endswith(".png"):
        # 1 read image
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        print(image.shape)
        name, extension = os.path.splitext(file)
        # 2 convert to grey scale
        gray_img = grayscale(image)
        print(gray_img.shape)
        # cv2.imwrite(output_folder+"/"+name+"grey.png", gray_img)
        
        #3 appliy median filter to remove outliers
        median_img=median_filter(gray_img,11)
        # cv2.imwrite(output_folder+"/"+name+"median.png", median_img)

        #4 apply gaus filter to remove gaussian noise
        gaussian_img=gaussian_filter(median_img,13,2)
        # cv2.imwrite(output_folder+"/"+name+"gaus.png", gaussian_img)

        #5 get hitogram threshold for segmentation
        t=show_histogram(median_img,file)

        #6 segmentation
        output= seg(median_img,t)
        # cv2.imwrite(output_folder+"/"+name+"seg.png", out)

        # 7 chaincode
        code= get_chaincode(output)

        names.append(name)
        codes.append(code)
        

        if image is not None:
            print("Loaded:", file)
#8 grouping
THRESHOLD = 8                              
groups = group_images(names, codes, THRESHOLD)
save_groups(groups, "output/groups.txt")