import cv2
import matplotlib.pyplot as plt
import numpy as np;



# def get_boundary(img):
#     w,h=img.size
#     neighbors = [
#     (-1, -1), (-1, 0), (-1, 1),
#     ( 0, -1),           ( 0, 1),
#     ( 1, -1), ( 1, 0),  ( 1, 1)
#     ]
#     bq=[]
#     for i in range(h):
#         for j in range(w):
#             if (img[i,j]==0):
#                 pass
#             else:
#                 start_pixel= (i,j)
#     bq.append(start_pixel)
#     qi,qj=start_pixel
#     for i,j in neighbors:
#         pi,pj=qi+i, qj+j
#         if

def get_boundary(img):
    h, w = img.shape
    # clockwise from north

# 3  2  1
# 4  X  0
# 5  6  7

#     (-1, -1), (-1, 0), (-1, 1),
#     ( 0, -1),    X     ( 0, 1),
#     ( 1, -1), ( 1, 0),  ( 1, 1)
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]
    

    # start pixel = first object pixel in a row-by-row scan
    start_pixel = None
    for i in range(h):
        for j in range(w):
            if img[i, j] != 0:
                start_pixel = (i, j)
                break
        if start_pixel is not None:
            break
    if start_pixel is None:
        return []
# boundary value queue
    bq = [start_pixel]
    qi, qj = start_pixel
    back = 6                 # backtrack pixel: the background pixel west of the start
    first_back = back

    while True:
        found = False
        for k in range(1, 9):                      # scan clockwise, just after the backtrack pixel
            idx = (back + k) % 8
            pi, pj = qi + neighbors[idx][0], qj + neighbors[idx][1]
            if 0 <= pi < h and 0 <= pj < w and img[pi, pj] != 0:
                # the neighbour checked just before this one becomes the new backtrack pixel
                prev = (idx - 1) % 8
                bi, bj = qi + neighbors[prev][0], qj + neighbors[prev][1]
                back = neighbors.index((bi - pi, bj - pj))
                qi, qj = pi, pj
                found = True
                break
        if not found:                              # single isolated pixel
            break
        if (qi, qj) == start_pixel and back == first_back:
            break                                  # back at the start, same entry side
        bq.append((qi, qj))

    return bq



def highlight_boundary(gray, bq, color=(0, 0, 255), thickness=1):
    """Return a colour copy of the image with the boundary points painted.
    gray: 2-D grey image, bq: list of (row, col) from get_boundary,
    color: (B, G, R), thickness: size of each painted dot in pixels."""
    h, w = gray.shape
    out = np.stack([gray, gray, gray], axis=2).astype(np.uint8)   # grey -> 3 channels
    r = thickness // 2
    for i, j in bq:
        out[max(i - r, 0):min(i + r + 1, h), max(j - r, 0):min(j + r + 1, w)] = color
    return out

def get_centroid(mask):
    row_sum=0
    col_sum=0
    count=0

    h,w=mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i,j]==255:
                row_sum+=i
                col_sum+=j
                count+=1
    if count==0:
        return -1
    centroid_row= row_sum//count
    centroid_col= col_sum//count
    print("centroid: ",centroid_row,centroid_col)
    return centroid_row,centroid_col

# Reorder the boundary list so it starts at the point farthest from the centroid.
def rotate_to_farthest(bq, centroid):
    ci, cj = centroid
    d2 = [(i - ci) ** 2 + (j - cj) ** 2 for i, j in bq]     
    k = int(np.argmax(d2))                                    
    return bq[k:] + bq[:k]



def resample_boundary(bq, n=32):
    N = len(bq)
    cum = [0.0] * (N + 1)
    for i in range(N):
        nxt = (i + 1) % N
        d = ((bq[i][0] - bq[nxt][0]) ** 2 + (bq[i][1] - bq[nxt][1]) ** 2) ** 0.5
        cum[i + 1] = cum[i] + d
    boundary_len = cum[N]
    spacing = boundary_len / n

    points = []
    k = 0                                       
    for m in range(n):
        target = m * spacing                      
        while k < N - 1 and cum[k + 1] < target:  
            k += 1
        next = (k + 1) % N
        seg = cum[k + 1] - cum[k]                 # length of this segment
        t = (target - cum[k]) / seg if seg > 0 else 0.0
        row = bq[k][0] + t * (bq[next][0] - bq[k][0])
        col = bq[k][1] + t * (bq[next][1] - bq[k][1])
        points.append((row, col))
    return points

import math


def chords_to_digits(points):
    n = len(points)
    digits = []
    for i in range(n):
        nxt = (i + 1) % n
        d_row = points[nxt][0] - points[i][0]
        d_col = points[nxt][1] - points[i][1]

        angle = math.degrees(math.atan2(-d_row, d_col))  
        if angle < 0:
            angle += 360                                 

        if angle < 22.5 or angle >= 337.5:
            digit = 0
        elif angle < 67.5:
            digit = 1
        elif angle < 112.5:
            digit = 2
        elif angle < 157.5:
            digit = 3
        elif angle < 202.5:
            digit = 4
        elif angle < 247.5:
            digit = 5
        elif angle < 292.5:
            digit = 6
        else:
            digit = 7
        digits.append(digit)
    print(digits)
    return digits

def first_difference(code):
    return [(code[i] - code[i - 1]) % 8 for i in range(len(code))]


def min_rotation(code):
    n = len(code)
    return min((code[k:] + code[:k] for k in range(n))) 

def make_signature(mask, get_boundary, n=32):
    bq = get_boundary(mask)
    bq = rotate_to_farthest(bq, get_centroid(mask))
    digits = chords_to_digits(resample_boundary(bq, n))
    return min_rotation(first_difference(digits)), digits





    