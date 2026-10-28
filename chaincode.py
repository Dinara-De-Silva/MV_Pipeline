import math


# ---------------------------------------------------------------- boundary
def get_boundary(img):
    """Moore-neighbour tracing, clockwise. Returns ordered (row, col) boundary pixels."""
    h, w = img.shape
    # neighbours clockwise from north: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]

    start_pixel = None                       # first object pixel in a row-by-row scan
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
    back = 6                                 # backtrack pixel = background pixel west of the start
    first_back = back

    while True:
        found = False
        for k in range(1, 9):                # scan clockwise, just after the backtrack pixel
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
            break                            # back at the start, same entry side
        bq.append((qi, qj))
    return bq


# ---------------------------------------------------------------- centroid / start point
def get_centroid(mask):
    """Mean (row, col) of all object pixels. Returns None for an empty mask."""
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
    return row_sum / count, col_sum / count         # true division, keep the decimals


def farthest_index(bq, centroid):
    """Index of the boundary point farthest from the centroid (first one wins a tie)."""
    ci, cj = centroid
    best_k = 0
    best_d2 = -1.0
    for k in range(len(bq)):
        d2 = (bq[k][0] - ci) ** 2 + (bq[k][1] - cj) ** 2
        if d2 > best_d2:
            best_d2 = d2
            best_k = k
    return best_k


def rotate_to_farthest(bq, centroid):
    k = farthest_index(bq, centroid)
    return bq[k:] + bq[:k]


# ---------------------------------------------------------------- 32 equal segments
def resample_boundary(bq, n=32):
    N = len(bq)
    cum = [0.0] * (N + 1)
    for i in range(N):
        nxt = (i + 1) % N
        d = ((bq[i][0] - bq[nxt][0]) ** 2 + (bq[i][1] - bq[nxt][1]) ** 2) ** 0.5
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
        points.append((bq[k][0] + t * (bq[nxt][0] - bq[k][0]),
                       bq[k][1] + t * (bq[nxt][1] - bq[k][1])))
    return points


# ---------------------------------------------------------------- digits and normalization
def chords_to_digits(points):
    n = len(points)
    digits = []
    for i in range(n):
        nxt = (i + 1) % n
        d_row = points[nxt][0] - points[i][0]
        d_col = points[nxt][1] - points[i][1]
        angle = math.degrees(math.atan2(-d_row, d_col))   # 0 = right, positive = up
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


def first_difference(code):
    return [(code[i] - code[i - 1]) % 8 for i in range(len(code))]


def min_rotation(code):
    n = len(code)
    return min(code[k:] + code[:k] for k in range(n))


# ---------------------------------------------------------------- the one function main.py calls
def get_chaincode(mask, n=32, return_raw=False):
    """mask: 2-D array, object non-zero, background 0 (one clean object).
    Returns the normalised n-digit chain code (list of ints), invariant to
    translation, scale, rotation and starting point.
    With return_raw=True returns (normalised_code, raw_code)."""
    bq = get_boundary(mask)
    centroid = get_centroid(mask)
    if len(bq) < 2 or centroid is None:
        return ([], []) if return_raw else []

    bq = rotate_to_farthest(bq, centroid)                  # repeatable start point
    raw = chords_to_digits(resample_boundary(bq, n))       # n equal segments -> n digits
    code = min_rotation(first_difference(raw))             # rotation + start-point invariance
    return (code, raw) if return_raw else code

# ========================
def get_chain_points(mask, n=32):
    """The n equally spaced points, starting at the point farthest from the centroid."""
    bq = get_boundary(mask)
    centroid = get_centroid(mask)
    if len(bq) < 2 or centroid is None:
        return []
    bq = rotate_to_farthest(bq, centroid)
    return resample_boundary(bq, n)

import cv2

def draw_chain_points(gray, points, line_color=(0, 0, 255), dot_color=(0, 255, 255),
                      start_color=(0, 200, 0), thickness=2, numbers=True):
    """Colour copy of the grey image with the points joined by lines (closed loop).
    Colours are (B, G, R). The first point (start) is drawn bigger in start_color."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts = [(int(round(col)), int(round(row))) for row, col in points]   # OpenCV wants (x, y) = (col, row)
    n = len(pts)
    for i in range(n):
        cv2.line(out, pts[i], pts[(i + 1) % n], line_color, thickness)   # last point joins back to the first
    for i, p in enumerate(pts):
        if i == 0:
            cv2.circle(out, p, 3 * thickness + 2, start_color, -1)
        else:
            cv2.circle(out, p, 2 * thickness, dot_color, -1)
        if numbers:
            cv2.putText(out, str(i), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return out