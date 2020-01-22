#-------------------------------------------------------------------------------
# Name:        Testing script
# Purpose:     Car damage analysis project
#
# Author:      kol
#
# Created:     17.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------

import cv2
import numpy as np
import json

from img_utils import *

IMG_DIR = "img\\"
META_FILE = "meta.json"
FILE_NAME = "photo_004.jpg"

with open(IMG_DIR + META_FILE) as f:
    meta = json.load(f)
    f.close()

img = cv2.imread(IMG_DIR + FILE_NAME)
#cv2.imshow('Source', img)

#result_img = align_images_meta(img, meta[FILE_NAME], warp_mode=cv2.MOTION_HOMOGRAPHY, debug=True)
#cv2.imshow('Aligned image', result_img)

score, patch_img, result_img = get_diff_meta(img, meta[FILE_NAME],
                                            color="gradient",
                                            fill_contours=True,
                                            align=False,
                                            apply_clahe=False,
                                            apply_filter=False,
                                            gradient_colors=("blue", "green"),
                                            debug = True)
print(score)

cv2.imshow('Result patch', patch_img)
cv2.imshow('Result image', result_img)


cv2.waitKey()
cv2.destroyAllWindows()

