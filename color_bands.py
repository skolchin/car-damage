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
from img_utils import *

IMG_WIDTH = 340
IMG_HEIGHT = 340
BAND_WIDTH = 12
DELAY_STEP = 5
MIN_DELAY = 1
MAX_DELAY = MIN_DELAY + 20 * DELAY_STEP
DIR_IN = "in"
DIR_OUT = "out"
MODE_RANDOM = "random"
MODE_GRAD = "grad"

dir = DIR_IN
mode = MODE_RANDOM
delay = MIN_DELAY * DELAY_STEP * 4

# Allocate a window
cv2.namedWindow('Colors', cv2.WINDOW_AUTOSIZE)

# Prepare controur image and contour
contour_img = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], "uint8")
h, w = contour_img.shape[:2]
contour = np.array([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])

# Determine number of bands within the image
num_bands = get_max_contour_colors(contour, contour_img.shape)
num_bands = int(np.ceil(num_bands / BAND_WIDTH))

# Function to define color vector according to current color mode
grad_colors =  None
n_grad_color = None

def get_colors():
    global grad_colors, n_grad_color

    if mode == MODE_RANDOM:
        colors = random_colors(num_bands)
        colors = list(np.repeat(colors, BAND_WIDTH, axis = 0))
    else:
        n = num_bands // 2
        grad_colors =  list(gradient_colors(random_colors(2), n))
        grad_colors.extend(reversed(grad_colors))
        n_grad_color = len(grad_colors)-1
        colors = list(np.repeat(grad_colors, BAND_WIDTH, axis = 0))

    return colors

# Function to get next color from color vector
def next_color():
    global grad_colors, n_grad_color

    if mode == MODE_RANDOM:
        new_color = random_colors(1)[0]
    else:
        new_color = grad_colors[n_grad_color]
        n_grad_color += 1
        if n_grad_color >= len(grad_colors):
            n_grad_color = 0

    return new_color

# Function to shift color vector according to current direction
def shift_colors():
    global colors

    new_color = next_color()
    if dir == DIR_IN:
        colors = colors[BAND_WIDTH:]
        colors.extend([new_color] * BAND_WIDTH)
    else:
        new_color = [new_color] * BAND_WIDTH
        new_color.extend(colors[0:(-1 * BAND_WIDTH)])
        colors = new_color

# Define colors and contour drawing params
colors = get_colors()
params = None

print("Press q to stop, d to change direction, c to change color mode, +/- to change delay")

while(True):
    # Draw a contour or updates colors of previously drawn one
    if params is None:
        params = draw_contour_(contour_img, contour, colors, filled=True)
    else:
        update_contour(contour_img, colors, params)

    # Show image
    cv2.imshow('Colors', contour_img)

    # Shift color vector
    shift_colors()

    # Handle user input
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord(' ') or key == ord('d'):
        dir = DIR_IN if dir != DIR_IN else DIR_OUT
        print("Direction changed to", dir)
    elif key == ord('+'):
        delay = delay + DELAY_STEP if delay < MAX_DELAY else MAX_DELAY
        print("Delay changed to", delay)
    elif key == ord('-'):
        delay = delay - DELAY_STEP if delay > MIN_DELAY else MIN_DELAY
        print("Delay changed to", delay)
    elif key == ord('c'):
        mode = MODE_RANDOM if mode != MODE_RANDOM else MODE_GRAD
        print("Color mode changed to", mode)
        colors = get_colors()


cv2.destroyAllWindows()
print("Done")

