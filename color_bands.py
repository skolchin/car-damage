#-------------------------------------------------------------------------------
# Name:        Color bands demo script
# Purpose:     Some fun with colors
#
# Author:      kol
#
# Created:     25.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------
import cv2
import numpy as np
import itertools
from img_utils import *

IMG_WIDTH = 340
IMG_HEIGHT = 340
BAND_WIDTH = 12
DELAY_STEP = 5

def shift_vector(vec, dir, step, new_val):
    if dir == "in":
        new_vec = vec[step:]
        new_vec.extend([new_val] * step)
    else:
        new_vec = [new_val] * step
        new_vec.extend(vec[0:(-1 * step)])
    return new_vec

class RandomColorIterator:
    def __init__(self, dir, num_bands, band_width=BAND_WIDTH):
        self.num_bands = num_bands
        self.band_width = band_width
        self.colors = random_colors(num_bands)
        self.colors = list(np.repeat(self.colors, BAND_WIDTH, axis = 0))
        self.dir = dir

    def __iter__(self):
        return self

    def __next__(self):
        new_color = random_colors(1)[0]
        self.colors = shift_vector(self.colors, self.dir, self.band_width, new_color)
        return self.colors

class GradientColorIterator:
    def __init__(self, dir, num_bands, band_width=BAND_WIDTH):
        self.band_width = band_width

        n = int(np.ceil(band_width / 2))
        self.spectre =  list(gradient_colors(random_colors(2), n))
        self.iter = itertools.cycle(self.spectre)

        self.colors = self.spectre
        self.colors.extend(reversed(self.colors))
        self.colors = list(np.repeat(self.colors, band_width, axis = 0))
        self.dir = dir

    def __iter__(self):
        return self

    def __next__(self):
        new_color = next(self.iter)
        self.colors = shift_vector(self.colors, self.dir, self.band_width, new_color)
        return self.colors


class ContGradientColorIterator:
    def __init__(self, dir, num_bands, band_width=BAND_WIDTH):
        #GradientColorIterator.__init__(self, dir, 1, num_bands * band_width)

        self.num_bands = 1
        self.band_width = num_bands * band_width

        n = int(np.ceil(self.band_width/ 2))
        self.spectre =  list(gradient_colors(random_colors(2), n))
        self.iter = itertools.cycle(self.spectre)

        self.colors = self.spectre
        self.colors.extend(reversed(self.colors))
        self.dir = dir

    def __next__(self):
        new_color = next(self.iter)
        self.colors = shift_vector(self.colors, self.dir, 1, new_color)
        return self.colors

DIRECTIONS = ["in", "out"]
COLOR_MODES = { "random" : RandomColorIterator,
                "gradient": ContGradientColorIterator }

DELAYS = np.linspace(1, 101, num=20, endpoint=False).astype(np.int)

# Allocate a window
cv2.namedWindow('Colors', cv2.WINDOW_AUTOSIZE)

# Prepare controur image and contour
contour_img = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], "uint8")
h, w = contour_img.shape[:2]
contour = np.array([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])

# Determine number of bands within the image
num_colors = get_max_contour_colors(contour, contour_img.shape)
num_bands = int(np.ceil(num_colors / BAND_WIDTH))

# Define colors and contour drawing params
mode_iter = itertools.cycle(COLOR_MODES)
dir_iter = itertools.cycle(DIRECTIONS)
delay_iter = itertools.cycle(DELAYS)

mode = next(mode_iter)
dir = next(dir_iter)
delay = next(delay_iter)

colors_iter = COLOR_MODES[mode](dir, num_bands)

print("Press q to stop, d to change direction, c to change color mode, +/- to change delay")
params = None

while(True):
    # Get color vector
    colors = next(colors_iter)

    # Draw a contour or updates colors of previously drawn one
    if params is None:
        params = draw_contour_(contour_img, contour, colors, filled=True)
    else:
        update_contour(contour_img, colors, params)

    # Show image
    cv2.imshow('Colors', contour_img)

    # Handle user input
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord(' ') or key == ord('d'):
        dir = next(dir_iter)
        print("Direction changed to", dir)
        colors_iter.dir = dir
    elif key == ord('+'):
        delay = next(delay_iter)
        print("Delay changed to", delay)
    elif key == ord('c'):
        mode = next(mode_iter)
        print("Color mode changed to", mode)
        colors_iter = COLOR_MODES[mode](dir, num_bands)


cv2.destroyAllWindows()
print("Done")

