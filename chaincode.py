import math
import cv2

# track the boundary of the object
def get_boundary(img):
    h, w = img.shape
    # neighbours clockwise from north: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]

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

    bq = [start_pixel]
    qi, qj = start_pixel
    back = 6                                 
    first_back = back

    while True:
        found = False
        for k in range(1, 9):                
            idx = (back + k) % 8
            pi, pj = qi + neighbors[idx][0], qj + neighbors[idx][1]
            if 0 <= pi < h and 0 <= pj < w and img[pi, pj] != 0:
                prev = (idx - 1) % 8
                bi, bj = qi + neighbors[prev][0], qj + neighbors[prev][1]
                back = neighbors.index((bi - pi, bj - pj))
                qi, qj = pi, pj
                found = True
                break
        if not found:                        # single isolated pixel
            break
        if (qi, qj) == start_pixel and back == first_back:
            break                            
        bq.append((qi, qj))
    return bq


# get the centroid of the object to find a start point for boundary devision
def get_centroid(mask):
    row_sum = 0
    col_sum = 0
    count = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 0:
                row_sum += i
                col_sum += j
                count += 1
    if count == 0:
        return None
    return row_sum / count, col_sum / count         

# take the farthest pont from centroid to boundary and then rotate the boundary to that farthest point.
# this make the startpoint of the boundary resampling is similar for all similar objects
def farthest_index(boundary, centroid):
    ci, cj = centroid
    best_k = 0
    best_d = -1.0
    for k in range(len(boundary)):
        d = (boundary[k][0] - ci) ** 2 + (boundary[k][1] - cj) ** 2
        if d > best_d:
            best_d = d
            best_k = k
    return best_k


def rotate_to_farthest(boundary, centroid):
    k = farthest_index(boundary, centroid)
    return boundary[k:] + boundary[:k]


# devide boundary into equal lenth parts
def resample_boundary(boundary, n=32):
    N = len(boundary)
    cum = [0.0] * (N + 1)
    for i in range(N):
        nxt = (i + 1) % N
        d = ((boundary[i][0] - boundary[nxt][0]) ** 2 + (boundary[i][1] - boundary[nxt][1]) ** 2) ** 0.5
        cum[i + 1] = cum[i] + d
    spacing = cum[N] / n

    points = []
    k = 0
    for m in range(n):
        target = m * spacing
        while k < N - 1 and cum[k + 1] < target:
            k += 1
        nxt = (k + 1) % N
        seg = cum[k + 1] - cum[k]
        t = (target - cum[k]) / seg if seg > 0 else 0.0
        points.append((boundary[k][0] + t * (boundary[nxt][0] - boundary[k][0]),
                       boundary[k][1] + t * (boundary[nxt][1] - boundary[k][1])))
    return points

# take the chaincode
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
    return digits

# get fist different to make chain code rotaion invarient
def first_difference(code):
    return [(code[i] - code[i - 1]) % 8 for i in range(len(code))]


# get min rotaion to make chain code start point invarient
def min_rotation(code):
    n = len(code)
    return min(code[k:] + code[:k] for k in range(n))



def get_chaincode(mask, n=32, return_raw=False):
    boundary = get_boundary(mask)
    centroid = get_centroid(mask)
    if len(boundary) < 2 or centroid is None:
        return ([], []) if return_raw else []
    boundary = rotate_to_farthest(boundary, centroid)                  
    raw = chords_to_digits(resample_boundary(boundary, n))       
    code = min_rotation(first_difference(raw))             
    return (code, raw) if return_raw else code

# to visualize chain points
def get_chain_points(mask, n=32):
    boundary = get_boundary(mask)
    centroid = get_centroid(mask)
    if len(boundary) < 2 or centroid is None:
        return []
    boundary = rotate_to_farthest(boundary, centroid)
    return resample_boundary(boundary, n)


def draw_chain_points(gray, points, line_color=(0, 0, 255), dot_color=(0, 255, 255),
                      start_color=(0, 200, 0), thickness=2, numbers=True):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts = [(int(round(col)), int(round(row))) for row, col in points]  
    n = len(pts)
    for i in range(n):
        cv2.line(out, pts[i], pts[(i + 1) % n], line_color, thickness)  
    for i, p in enumerate(pts):
        if i == 0:
            cv2.circle(out, p, 3 * thickness + 2, start_color, -1)
        else:
            cv2.circle(out, p, 2 * thickness, dot_color, -1)
        if numbers:
            cv2.putText(out, str(i), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out