from pathlib import Path
from math import floor
import numpy as np
from PIL import Image
from copy import deepcopy

from typing import List
from skimage.transform import rescale
from skimage.draw import line

def list_all_tiffs(path: Path = Path("./")) -> list:
    """
    Takes a path and lists all the .tiff files within that directory structure.
    """
    p = path.glob('**/*')
    files = [x for x in p if x.is_file() & (x.suffix in [
        ".tiff",
        ".tif"]
                                            )
             ]
    return files



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

    def create_grid(self, 
                    ny: int = 0, 
                    nx: int = 0,
                    sizex: int = 0,
                    sizey: int = 0,
                    ):
        """
        Using number of cells creates a grid for the photo and trims the image data.
        """
        x_pixels = self.image_data.shape[1]
        y_pixels = self.image_data.shape[0]
        if (ny != 0) and (nx != 0):
            x = floor(x_pixels/nx)
            y = floor(y_pixels/ny)
        elif (sizex != 0) and (sizey !=0):
            x = sizex
            y = sizey
            nx = floor(x_pixels/x)
            ny = floor(y_pixels/y)
        else:
            raise KeyError("Expected either [x,y] arguments or [sizex, sizey] arguments")

        self.grid = np.zeros([ny,nx])
        # discard data from the bottom and right - ask Yas
        self.image_data = self.image_data[0:y * ny, 0:x * nx, :]

    def sample(self, 
               n: int, 
               seed: int=8008135,
               how: str="simple",
               **kwargs: bool):
        """
        Choose divisions for sample.
        """
        #reset grid
        try:
            if np.sum(self.grid) != 0:
                self.grid = np.zeros(self.grid.shape)
        except AttributeError:
            raise AttributeError("grid not found, try using create_grid() first")



        #kwarg separate images, with replacement
        
        # generate sample on grid
        
        np.random.seed(seed)

        indices = []

        if how == "simple":
            indices = np.random.choice(np.arange(self.grid.size), replace=False, size=n,)

        if how == "rowwise":
            indices = np.zeros(n * self.grid.shape[0], dtype=int)
            for index in range(self.grid.shape[0]):
                indices[index * n:(index + 1) * n] = np.random.choice(
                    np.arange(self.grid.shape[1]) + self.grid.shape[1] * index,
                    replace=False,
                    size=n,
                )

        
        
        self.grid[np.unravel_index(indices,self.grid.shape)] = list(range(1, len(indices)+1)) 
        # from grid calculate stop/start and return tuples of top left and bottom right corners
        x_pixels = self.image_data.shape[1]
        y_pixels = self.image_data.shape[0]
        x_pixels_per_cell = floor(x_pixels / self.grid.shape[1])
        y_pixels_per_cell = floor(y_pixels / self.grid.shape[0])
        
        coordinates = []
        for i in range(1,len(indices)+1):
            gridy, gridx = np.where(self.grid == i)
            top_left = (y_pixels_per_cell * gridy[0],
                        x_pixels_per_cell * gridx[0])
            bottom_right = (y_pixels_per_cell * (gridy[0] + 1) - 1,
                            x_pixels_per_cell * (gridx[0] + 1) - 1)
            coordinates.append((top_left,bottom_right))

        self.samples = coordinates


    def split_samples(self) -> List[np.ndarray]:
        """
        Splits samples into multiple arrays and returns them.
        """
        
        image_data = self.image_data
        try:
            coordinates = self.samples
        except AttributeError:
            raise AttributeError("self.samples do not exist, try using self.sample() first")

        x_pixels_per_cell = floor(image_data.shape[1] / self.grid.shape[1])
        y_pixels_per_cell = floor(image_data.shape[0] / self.grid.shape[0])
        list_of_images = [np.zeros((y_pixels_per_cell, x_pixels_per_cell, image_data.shape[2]))] * len(coordinates)

        for i in range(len(coordinates)):
            list_of_images[i] = image_data[
                coordinates[i][0][0]:coordinates[i][1][0] + 1,
                coordinates[i][0][1]:coordinates[i][1][1] + 1,
                :
            ]

        return list_of_images

    def outline_samples(self,
                        downsample_ratio: float = 0.1,
                        grid_colour: list[int] = [255,255,255,255],
                        **kwargs: bool) -> np.ndarray:
        if downsample_ratio != 1.0:
            image = rescale(self.image_data,
                            downsample_ratio,
                            anti_aliasing=False,
                            channel_axis = 2)
            coordinates = deepcopy(self.samples)
            for i in range(len(coordinates)):
                coordinates[i] = tuple(
                    (floor(coordinate[0] * downsample_ratio),
                     floor(coordinate[1] * downsample_ratio)
                    )
                    for coordinate in coordinates[i])
                
                # check if top-left of box is same coordinates as bottom-right and add 1 if they are
                if coordinates[i][0][0] == coordinates[i][1][0]:
                    coordinates[i] = (coordinates[i][0],
                                      (coordinates[i][0][0]+1,coordinates[i][1][1]))
                if coordinates[i][0][1] == coordinates[i][1][1]:
                    coordinates[i] = (coordinates[i][0],
                                      (coordinates[i][1][0], coordinates[i][0][1]+1))

                # check if bottom-right of box is outside bounds and take away 1 if it is

                if coordinates[i][1][0] == image.shape[0]:
                    coordinates[i] = (
                        (coordinates[i][0][0] - 1, coordinates[i][0][1]),
                        (coordinates[i][1][0] - 1, coordinates[i][1][1]))
                if coordinates[i][1][1] == image.shape[1]:
                    coordinates[i] = (
                        (coordinates[i][0][0], coordinates[i][0][1] - 1),
                        (coordinates[i][1][0], coordinates[i][1][1] - 1))


        else:
            image = self.image_data
            coordinates = self.samples
        if image.shape[2] != len(grid_colour):
            grid_colour = grid_colour[0:image.shape[2]]

        grid_colour = [i/255 for i in grid_colour]

        for pair in coordinates:

            points = [pair[0],(pair[0][0],pair[1][1]),pair[1],(pair[1][0],pair[0][1])]
            for i in range(len(points)):
                ypoints, xpoints = line(*points[i-1],*points[i])
                colour_array = np.array(grid_colour * len(xpoints))
                colour_array.shape = (len(ypoints), len(grid_colour))
                image[ypoints,xpoints,:] = colour_array

        image = np.floor(image * 255).astype(np.uint8)

        return image 





