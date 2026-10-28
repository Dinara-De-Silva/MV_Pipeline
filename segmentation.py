import cv2
import matplotlib.pyplot as plt
import numpy as np;

def show_histogram(image, filename):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    smooth = smooth_histogram(histogram, 3)
    t = find_valley(smooth)
    mean=np.mean(histogram)
    print("mean ", mean)
    print(t)


    plt.plot(histogram)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Image Histogram")
    plt.savefig(f"outputs/{filename}_histogram.png")
    plt.close()

    
    plt.plot(smooth)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Image Histogram")
    plt.savefig(f"outputs/{filename}smooth_histogram.png")
    plt.close()

    return t


def smooth_histogram(hist, passes=3):
    h = [float(v) for v in hist]
    n = len(h)
    for _ in range(passes):
        new = h[:]
        for i in range(n):
            left = h[i - 1] if i > 0 else h[i]
            right = h[i + 1] if i < n - 1 else h[i]
            new[i] = (left + h[i] + right) / 3
        h = new
    return h


def find_two_peaks(smooth, min_gap=40):
    n = len(smooth)
    top = 0
    for i in range(n):
        if smooth[i] > smooth[top]:
            top = i
    other = None
    for i in range(n):
        if abs(i - top) >= min_gap and (other is None or smooth[i] > smooth[other]):
            other = i
    return (top, other) if top < other else (other, top)


def find_valley(smooth, min_gap=40, rise=0.05):
    first, second = find_two_peaks(smooth, min_gap)

    # go down from the first peak towards the second one
    valley = first
    t = first + 1
    while t < second:
        if smooth[t] < smooth[valley]:
            valley = t                                  
        elif smooth[t] > smooth[valley] * (1 + rise):   
            break
        t += 1
    return valley

def seg(image,t):
    h,w=image.shape
    output = np.zeros((h,w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if(image[i,j]>=t):
                output[i,j]=255
            else:
                output[i,j]=0
    # remove white color small components of small astroids
    mask1 = remove_small_components(output)
    # remove black color small components inside the large astroid
    mask2 = remove_small_components(mask1, False)
    return mask2



#remove smaller components except the main large component 
def remove_small_components(binary, is_white=True):
    if (is_white):
        intensity=255
        result = np.zeros_like(binary)
    else:
        intensity=0
        result = np.full_like(binary, 255)
        
    height, width = binary.shape

    visited = np.zeros((height, width), dtype=bool)
    components = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0),  ( 1, 1)
    ]

    # Scan the image
    for y in range(height):
        for x in range(width):

            if binary[y, x] == intensity and not visited[y, x]:

                stack = [(y, x)]
                visited[y, x] = True
                component = []

                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))

                    for dy, dx in neighbors:
                        ny = cy + dy
                        nx = cx + dx

                        if (0 <= ny < height and
                            0 <= nx < width and
                            not visited[ny, nx] and
                            binary[ny, nx] == intensity):

                            visited[ny, nx] = True
                            stack.append((ny, nx))

                components.append(component)
    
    if components:
        largest = max(components, key=len)

        for y, x in largest:
            result[y, x] = intensity

    return result

