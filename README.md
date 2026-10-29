# MV_Pipeline

Groups images of single objects using 32 bit chaincode.

## How to run

Install the dependencies:

```
pip install opencv-python numpy matplotlib
```

Run the pipeline, passing the folder that contains your `.png` / `.jpg` images:

```
python main.py ./inputs
```

Results:

- `output/groups.txt` – the image groups

## Pipeline

1. **Load images** – reads `.png` / `.jpg` files from the given folder.
2. **Grayscale** – converts to a single channel using weighted RGB.
3. **Median filter** – 11×11 window to remove noise and outliers.
4. **Threshold** – picks the valley between the two peaks of the smoothed histogram.
5. **Segmentation** – pixels at or above the threshold become the object (binary mask).
6. **Clean-up** – keeps only the largest object region and fills holes inside it.
7. **Boundary trace** – follows the object outline using Moore tracing.
8. **Boundary subdivision** – splits the boundary into 32 equally spaced points.
9. **Chain code** – converts the points into direction codes, made invariant to rotation and start point.
10. **Code distance** – compares two chain codes using circular differences.
11. **Grouping** – places images with a distance of 8 or less into the same group.
12. **Output** – writes the groups to `output/groups.txt`.

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs the whole pipeline |
| `preprocess.py` | Grayscale, median and Gaussian filters |
| `segmentation.py` | Thresholding and clean-up |
| `chaincode.py` | Boundary trace, subdivision and chain code |
| `grouping.py` | Code distance and grouping |
