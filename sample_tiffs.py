from imgsmply import *
from pathlib import Path
from PIL import Image

import yaml
Image.MAX_IMAGE_PIXELS = int(4.295e+9)

with open("conf.yml","r") as f:
    conf = yaml.safe_load(f)

input_path = Path(conf["input_path"])
output_path = Path(conf["output_dir"])
seed = conf["seed"]
for path in list_all_tiffs(input_path):
    (output_path / path.stem).mkdir(parents=True, exist_ok=True)

    img = SamplePhoto(path, scale=conf["pixels_per_mm"])
    img.create_grid(*conf["grid"])
    img.sample(conf["number_of_samples"], seed=seed)
    image_arrays = img.split_samples()

    for array, number in zip(image_arrays,range(len(image_arrays))):
        Image.fromarray(array).save(output_path / path.stem / f"sample{number}.tiff")
    
    print("got_to_here_1")

    Image.fromarray(img.outline_samples(downsample_ratio=conf["downsample_ratio"])).save(output_path / path.stem / "outlined_samples.tiff")

    with open(output_path / path.stem / "sample_coordinates.csv","w") as f:
        f.write("sample_number, x_start, y_start, x_end, y_end \n")
        sample_number = 0
        for sample in img.samples:
            f.write(f"{sample_number}, {sample[0][1]}, {sample[0][0]}, {sample[1][1]}, {sample[1][0]}\n ")
            sample_number += 1

    seed += 1

