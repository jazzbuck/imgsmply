# imgsmply

Pronounced "image sarm-plee", `imgsmply` is a program that looks through a directory and samples every `.tiff` in that directory.
It samples from a grid overlay, and then draws its sampled cells.
Finally, it produces a csv with the grid locations.

## installation

To install requirements:

```shell
pip install -r requirements.txt
```

## running

To run:

```python
python main.py
```

## configuration options

`input_path`: location of directory of .tiff files

`output_dir`: location where outputs will be placed

`pixels_per_mm`: scale, not used

`grid_number`: [number of cells in y direction, number of cells in x direction] - this divides the image equally. If choosing the size of an individual cell use `grid_size` instead and comment this out.

`grid_size`: [size of cell in pixels y direction, size of cell in pixels x direction] - this creates a grid based on cells of the chosen size. If dividing the image into equal divisions use `grid_number` instead and comment this out.

`number_of_samples`: number of cells to sample (without replacement)

`downsample_ratio`: relative size of outlined version of input file

`seed`: random seed for reproducibility

`how`: either `"simple"` or `"rowwise"` at the moment. Simple creates a sample of size `number_of_samples` from the entire grid, whilst rowwise takes a sample of size `number_of_samples` from each row of the image.
