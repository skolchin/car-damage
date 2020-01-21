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

import cv2
import numpy as np
import tkinter as tk
import json

from pathlib import Path
from imutils.perspective import order_points
from skimage.measure import compare_ssim, find_contours
from skimage.draw import polygon
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

from gr.utils import get_image_area

# Modified version of imutils.four_point_transform()
# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com
def four_point_transform(image, pts, inverse=False):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    d = np.min(rect)
    if d >= 0:
        (tl, tr, br, bl) = rect
    else:
        # Correct all rectangle points to be greater than or equal to 0
        corrected_rect = deepcopy(rect)
        d = abs(d)
        for r in corrected_rect:
            r[0] += d
            r[1] += d
        (tl, tr, br, bl) = corrected_rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    if inverse:
        M = np.linalg.pinv(M)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def clahe(img):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to l_channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge back and convert to RGB color space
    merged = cv2.merge((cl,a,b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

def gauss_filter(img):
    s = 2
    w = 5
    t = (((w - 1)/2)-0.5)/s
    return gaussian_filter(img, sigma=s, truncate=t)

def align_images(im1, im2, warp_mode=cv2.MOTION_TRANSLATION, debug=False):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow("Image 1 gray", im1_gray)
        cv2.imshow("Image 2 gray", im2_gray)

    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    print(cc)

    try:
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

        if debug:
            cv2.imshow('Aligned image', im2_aligned)
        return im2_aligned

    except:
        # Alignment unsuccessfull
        return im2

def get_parts(img, meta, apply_clahe=False, apply_filter=False, debug=False):
    # Get image areas
    split = meta['split']
    parts = {}
    parts['left'] = get_image_area(img, [0, 0, split, img.shape[0]])
    parts['right'] = get_image_area(img, [split, 0, img.shape[1], img.shape[0]])
    if debug:
        for k in parts:
            cv2.imshow('Parts ' + k, parts[k])

    # Transform areas to patches according to specified transform rect
    tr = {}
    patches = {}
    for k in parts:
        tr[k] = np.array(meta[k])
        patches[k] = four_point_transform(img, tr[k])

    # Find out maximum size of a patch
    new_size = (
        max([x.shape[1] for x in patches.values()]),
        max([x.shape[0] for x in patches.values()])
    )

    # Resize patches to common size and make grays
    patches_resized = {}
    grays = {}
    for k in patches:
        patches_resized[k] = cv2.resize(patches[k], dsize = new_size, interpolation = cv2.INTER_CUBIC)
        if apply_clahe:
            patches_resized[k] = clahe(patches_resized[k])
        if apply_filter:
            patches_resized[k] = gauss_filter(patches_resized[k])

        if debug:
            cv2.imshow('Patch ' + k, patches[k])
            cv2.imshow('Patch resized ' + k, patches_resized[k])

    return parts, tr, patches, patches_resized

def align_images_meta(img, meta, warp_mode=cv2.MOTION_TRANSLATION, debug=False):
    parts, tr, patches, patches_resized = get_parts(img, meta, debug=debug)
    return align_images(patches_resized['left'], patches_resized['right'], warp_mode, debug)


def get_diff(img, meta, align=False, fill_contours=False, apply_clahe=False, apply_filter=False, debug=False):
    # Get image parts
    parts, tr, patches, patches_resized = get_parts(img, meta,
                                                    apply_clahe=apply_clahe,
                                                    apply_filter=apply_filter,
                                                    debug=debug)

    # Algin patches
    if align:
        patch_aligned = align_images(patches_resized['left'], patches_resized['right'],
                                     cv2.MOTION_TRANSLATION, debug)
        patches_resized['right'] = patch_aligned

    # Make up grays
    grays = {}
    for k in patches_resized:
        grays[k] = cv2.cvtColor(patches_resized[k], cv2.COLOR_BGR2GRAY)

    # Calculate scrore and difference
    (score, diff) = compare_ssim(grays['left'], grays['right'], full=True, gaussian_weights=True)

    diff_img = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imshow('Diff image', thresh)

    # Find contours and draw them on the left patch
    # In addition create a binary difference mask
    diff_img = patches_resized['left'].copy()
    diff_mask = np.zeros((diff_img.shape[:2]), "uint8")

##    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    cv2.drawContours(diff_img, contours, -1, (0, 0, 255), 1)
##    cv2.drawContours(diff_mask, contours, -1, 255, 1)

    contours = find_contours(diff, level=0.5, fully_connected="low", positive_orientation="low")
    if fill_contours:
        # Draw filled contour
        for n, c in enumerate(contours):
            rr, cc = polygon(c[:, 0], c[:, 1], shape=diff_img.shape)
            diff_img[rr, cc] = [0, 0, 255]
            diff_mask[rr, cc] = 255
    else:
        # Draw contour of contour only
        for n, c in enumerate(contours):
            for y, x in c:
                # Mark contour points as red on the patch and white on the mask
                diff_img[int(y), int(x)] = [0, 0, 255]
                diff_mask[int(y), int(x)] = 255

    if debug:
        cv2.imshow('Diff img', diff_img)
        cv2.imshow('Diff mask', diff_mask)

    # Resize difference mask to original patch size
    diff_mask_resized = cv2.resize(diff_mask,
                                   dsize=(patches['left'].shape[1], patches['left'].shape[0]),
                                   interpolation=cv2.INTER_CUBIC)

    if debug:
        cv2.imshow('Diff mask resized', diff_mask_resized)

    # Undo 4 point transformation
    # Top-left corner of original transform rect is a key point on target image
    rect = deepcopy(tr['left'])
    tl = order_points(tr['left'])[0]
    for r in rect:
        r[0] -= tl[0]
        r[1] -= tl[1]

    diff_mask_tr = four_point_transform(diff_mask_resized, rect, inverse=True)
    if debug:
        cv2.imshow('Diff mask transformed', diff_mask_tr)

    # Transform diff mask to patch of the same size as left image area
    result_img = parts['left']

    result_patch = np.zeros(result_img.shape[:2], dtype = result_img.dtype)
    dy, dx = diff_mask_tr.shape
    dx += tl[0]
    dy += tl[1]

    result_patch[int(tl[1]):int(dy), int(tl[0]):int(dx)] = diff_mask_tr

    # Apply patch to left image area
    idx = (result_patch != 0)
    result_img[idx] = (0, 0, 255)

    return score, diff_img, result_img


