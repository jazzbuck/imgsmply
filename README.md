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

`grid`: [number of cells in y direction, number of cells in x direction]

`number_of_samples`: number of cells to sample (without replacement)

`downsample_ratio`: relative size of outlined version of input file

`seed`: random seed for reproducibility
