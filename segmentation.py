import cv2
import matplotlib.pyplot as plt
import numpy as np;

def show_histogram(image, filename):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    s_hist=smooth_histogram(histogram,filename)
    t= find_vally(s_hist)
    mean=np.mean(histogram)
    print("mean ", mean)
    print(t)


    plt.plot(histogram)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Image Histogram")
    plt.savefig(f"outputs/{filename}_histogram.png")
    plt.close()
    return t

def smooth_histogram(hist,filename):
    l=256
    smooth_hist=np.zeros(l)
    for i in range(1,l-1):
        smooth_hist[i]=(hist[i-1]+hist[i]+hist[i+1])/3
    plt.plot(smooth_hist)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Image Histogram")
    plt.savefig(f"outputs/{filename}_smooth_hist.png")
    plt.close()
    return smooth_hist

def find_vally(smooth_histogram):
    l=2
    # climb first hill
    t=0
    while(smooth_histogram[t+1]>=smooth_histogram[t]):
        t+=1
        if(t==256):
            break
    print("hill point :",t)
    while(smooth_histogram[t+1]<=smooth_histogram[t]):
        t+=1
        if(t==256):
            break
    print("vally point: ",t)
    return t


def seg(image,t):
    h,w=image.shape

  
    output = np.zeros((h,w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if(image[i,j]>=t):
                output[i,j]=255
            else:
                output[i,j]=0
    return output