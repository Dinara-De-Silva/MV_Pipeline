import cv2;
import numpy as np;
import os;

# convert to grey scale
def grayscale(img):
    b = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    r = img[:, :, 2].astype(np.float64)

    gray_img = 0.299 * r + 0.587 * g + 0.114 * b
    return np.round(gray_img).astype(np.uint8)

# support funtions for filters
def wrap_image(img, N):
    H, W = img.shape
    k = N // 2

    new_height = H + 2 * k
    new_width = W + 2 * k

    output = np.zeros((new_height, new_width), dtype=np.uint8)

    # Original image
    output[k:k+H, k:k+W] = img
    output[:k, k:k+W] = img[-k:, :] 
    output[k+H:, k:k+W] = img[:k, :] 
    output[k:k+H, :k] = img[:, -k:]
    output[k:k+H, k+W:] = img[:, :k] 
    # corners
    output[:k, :k] = img[-k:, -k:]
    output[:k, k+W:] = img[-k:, :k]
    output[k+H:, :k] = img[:k, -k:]
    output[k+H:, k+W:] = img[:k, :k]

    return output

def get_median(mask):
    array = [element for row in mask for element in row]
    array.sort()

    n= len(array)
    mid=n//2
    if n%2==1:
        median= array[mid]
    else:
        median = (int(array[mid-1]) + int(array[mid])) / 2
    return np.uint8(np.round(median))

# median filter
def median_filter(img, N=5):
    H, W = img.shape

    wrapped_img = wrap_image(img, N)

    output = np.zeros((H, W), dtype=np.uint8)


    for i in range(H):
        for j in range(W):
            mask = wrapped_img[i:i+N, j:j+N]
            output[i, j] = get_median(mask)

    return output

# gaussian filter
def get_gaussian_kernel_2d(k, sigma=1):
    if k % 2 == 0: 
        k+=1
    r = k // 2
    
    x = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, x)
    
    kernel = np.exp(-(X**2 + Y**2) / (2 * (sigma**2)))
    
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(img, k, sigma=1):
    H, W= img.shape
    wrapped_img = wrap_image(img, k)
    output = np.zeros((H, W), dtype=np.uint8)

    kernal = get_gaussian_kernel_2d(k,sigma)

    for i in range(H):
        for j in range(W):
            mask = wrapped_img[i:i+k, j:j+k]
            output[i, j] = np.sum(mask*kernal)
    return output