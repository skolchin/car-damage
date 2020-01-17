#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     17.01.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import sys
import cv2
import numpy as np
import json

from pathlib import Path
from imutils.perspective import four_point_transform
from imutils import grab_contours
from skimage.measure import compare_ssim, find_contours

META_FILE = "img\\meta.json"
FILE_NAME = "photo_003.jpg"

def get_image_area(img, r):
    """Get part of an image defined by rectangular area.

    Parameters:
        img      An OpenCv image
        r        Area to extract (list or tuple [x1,y1,x2,y2])

    Returns:
        Extracted area
    """
    if r[0] < 0 or r[1] < 0:
       raise ValueError('Invalid area origin: {}'.format(r))
    dx = r[2] - r[0]
    dy = r[3] - r[1]
    if dx <= 0 or dy <= 0:
       raise ValueError('Invalid area length: {}'.format(r))

    im = None
    if len(img.shape) > 2:
       im = np.empty((dy, dx, img.shape[2]), dtype=img.dtype)
    else:
       im = np.empty((dy, dx), dtype=img.dtype)

    im[:] = img[r[1]:r[3], r[0]:r[2]]
    return im


with open(str(META_FILE), "r") as f:
    meta = json.load(f)
    f.close()

file_name = meta[FILE_NAME]['file_name']
img = cv2.imread(file_name)
if img is None:
    raise Exception("File not found", file_name)
cv2.imshow('Source', img)

split = meta[FILE_NAME]['split']
parts = {}
parts['left'] = get_image_area(img, [0, 0, split, img.shape[0]])
parts['right'] = get_image_area(img, [split, 0, img.shape[1], img.shape[0]])

for k in parts:
    cv2.imshow("Image: " + k, parts[k])

tr = {}
patches = {}
grays = {}
for k in ['left', 'right']:
    tr[k] = np.array(meta[FILE_NAME][k])
    patches[k] = four_point_transform(img, tr[k])

new_size = (
    max([x.shape[1] for x in patches.values()]),
    max([x.shape[0] for x in patches.values()])
)

for k in patches:
    patches[k] = cv2.resize(patches[k], dsize = new_size, interpolation = cv2.INTER_CUBIC)
    grays[k] = cv2.cvtColor(patches[k], cv2.COLOR_BGR2GRAY)

for k in grays:
    cv2.imshow("Patch: " + k, patches[k])

(score, diff) = compare_ssim(patches['left'], patches['right'], multichannel = True, gaussian_weights = False, full=True)

diff_gray = cv2.cvtColor((diff * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)
contours = find_contours(diff_gray, level = 0.2, fully_connected = "low", positive_orientation = "low")

diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
cv2.imshow("Diff", diff_gray)

diff_img = np.zeros((diff.shape[0], diff.shape[1], 3), "uint8")

for n, contour in enumerate(contours):
    for x, y in contour:
        diff_img[int(x), int(y)] = [0, 0, 255]

cv2.imshow("Diff contours", diff_img)

##diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
##thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
##cv2.imshow("Thresh", thresh)
##
##cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##cnts = grab_contours(cnts)
##
##for c in cnts:
##    (x, y, w, h) = cv2.boundingRect(c)
##    if x != 0 or y != 0 or w != diff.shape[1] or h != diff.shape[0]:
##        print(x, y, w, h)
##        cv2.rectangle(patches['left'], (x, y), (x + w, y + h), (0, 0, 255), 2)
##        cv2.rectangle(patches['right'], (x, y), (x + w, y + h), (0, 0, 255), 2)
##
##for k in patches:
##    cv2.imshow("Result: " + k, patches[k])


cv2.waitKey()
cv2.destroyAllWindows()

