from imgsmply import *
from pathlib import Path
from PIL import Image

img = SamplePhoto(Path("./sample.tiff"), scale=10)
img.create_grid(5,5)
img.sample(5, seed=10)
image_arrays = img.split_samples()

for array, number in zip(image_arrays,range(len(image_arrays))):
    Image.fromarray(array).save(f"sample{number}.tiff")

