import os


def code_distance(a, b):
    n = len(a)
    total = 0
    for i in range(n):
        d = abs(a[i] - b[i])
        if d > 4:
            d = 8 - d
        total += d
    return total


def group_images(names, codes, threshold):
    groups = []           
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
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:           
        for number, g in enumerate(groups, start=1):
            f.write("Group " + str(number) + ": " + ", ".join(g) + "\n")