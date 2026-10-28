import os


def code_distance(a, b):
    """Total circular digit difference between two codes, using the best of all circular shifts.
    Digits 7 and 0 are neighbours (difference 1), so the difference of two digits is min(d, 8 - d)."""
    n = len(a)
    best = None
    for shift in range(n):
        total = 0
        for i in range(n):
            d = abs(a[i] - b[(i + shift) % n])
            if d > 4:
                d = 8 - d
            total += d
            if best is not None and total >= best:      # already worse than the best shift, stop early
                break
        if best is None or total < best:
            best = total
    print(best)
    return best


def group_images(names, codes, threshold):
    """names: list of file names, codes: list of chain codes (same order).
    An image joins the first group whose first member is within `threshold`;
    otherwise it starts a new group. Returns a list of groups (lists of names)."""
    groups = []            # each group = list of indices
    for idx in range(len(names)):
        placed = False
        for g in groups:
            if code_distance(codes[idx], codes[g[0]]) <= threshold:
                g.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
    return [[names[i] for i in g] for g in groups]


def save_groups(groups, path="output/groups.txt"):
    """Write one line per group."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)               # create the output folder if it is missing
    with open(path, "w") as f:
        for number, g in enumerate(groups, start=1):
            f.write("Group " + str(number) + ": " + ", ".join(g) + "\n")