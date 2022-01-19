from pathlib import Path
from math import floor
import numpy as np
from PIL import Image

def list_all_tiffs(path: Path = Path("./")) -> list:
    p = path.glob('**/*')
    files = [x for x in p if x.is_file() & (x.suffix in [
        ".tiff",
        ".tif"]
                                            )
             ]
    return files



def sample_photo(n: int, 
                 image_path: Path,
                 scale: int):
    pass


class SamplePhoto():
    """
    Ties together image data, path, and sampling grid.
    """
    def __init__(self,
                 image_path: Path,
                 scale: int):
        """
        Create SamplePhoto.
        """
        if not isinstance(image_path, Path):
            raise TypeError("image_path not of type Path")
        if not isinstance(scale, int):
            raise TypeError("scale not of type int")

        self.path = image_path
        self.scale = scale
        
        img = Image.open(self.path)

        self.image_data = np.array(img)
        self.grid = np.zeros([self.image_data.shape[1], 
                              self.image_data.shape[0]])

    def create_grid(self, nx, ny):
        """
        Using number of cells creates a grid for the photo and trims the image data.
        """
        x_pixels = self.image_data.shape[1]
        y_pixels = self.image_data.shape[0]
        x = floor(x_pixels/nx)
        y = floor(y_pixels/ny)

        self.grid = np.zeros([x,y])
        # discard data from the bottom right - ask Yas
        self.image_data = self.image_data[0:y * ny, 0:x * nx, :]


    def sample(self, 
               n: int, 
               seed: int=8008135,
               **kwargs: bool):
        """
        Choose divisions for sample.
        """
        #kwarg separate images, with replacement
        
        # generate sample on grid

        # from grid calculate stop/start and return tuples of top left and bottom right corners



    def split_samples(self):

        pass


    def outline_samples(self,
                        **kwargs):

        if with_replacement:
            raise 
        pass




