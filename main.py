import cv2
import os
from preprocess import grayscale, median_filter, gaussian_filter
from segmentation import show_histogram, seg, remove_small_components
from chaincode import get_boundary,highlight_boundary


input_folder = "inputs"
output_folder = "outputs"

image_files = os.listdir(input_folder)


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
        cv2.imwrite(output_folder+"/"+name+"grey.png", gray_img)
        
        # appliy mean filter to remove outliers
        median_img=median_filter(gray_img,15)
        cv2.imwrite(output_folder+"/"+name+"median.png", median_img)

        # apply gaus filter to remove gaussian noise
        gaussian_img=gaussian_filter(median_img,13,2)
        cv2.imwrite(output_folder+"/"+name+"gaus.png", gaussian_img)
        # get hitogram threshold for segmentation
        t=show_histogram(median_img,file)
        out= seg(median_img,t)
        cv2.imwrite(output_folder+"/"+name+"seg.png", out)

        # remove white color small components of small astroids
        mask1 = remove_small_components(out)
        cv2.imwrite(output_folder+"/"+name+"mask1.png", mask1)

        # remove black color small components inside the large astroid
        mask2 = remove_small_components(mask1, False)
        cv2.imwrite(output_folder+"/"+name+"mask2.png", mask2)

        bq = get_boundary(mask2)
        result = highlight_boundary(gray_img, bq, (0, 0, 255), 3)
        # cv2.imwrite("outputs/boundary.png", result)
        cv2.imwrite(output_folder+"/"+name+"bd.png", result)

        if image is not None:
            print("Loaded:", file)