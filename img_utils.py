#-------------------------------------------------------------------------------
# Name:        Image functions
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

from pathlib import Path
from imutils.perspective import order_points
from skimage import measure
from skimage import draw
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from matplotlib import colors as plt_colors
from scipy.spatial import distance
from random import randint

from gr.utils import get_image_area

def random_colors(n):
    """Returns n random colors"""
    rr = []
    for i in range(n):
        r = randint(0,255)
        g = randint(0,255)
        b = randint(0,255)
        rr.extend([(r,g,b)])
    return rr

def gradient_colors(colors, n):
    """Returns color gradient of length n from colors[0] to colors[1]"""
    if len(colors) < 2:
        raise ValueError("Two colors required to compute gradient")
    if n < 2:
        raise ValueError("Gradient length must be greater than 1")

    c = np.linspace(0, 1, n)[:, None, None]
    x = np.array([colors[0]])
    y = np.array([colors[1]])
    g = y + (x - y) * c

    return g.astype(x.dtype)

def color_to_cv_color(name):
    """Convert color with given name to OpenCV color"""
    mp_rgb = plt_colors.to_rgb(name)
    cv_bgr = [c * 255 for c in reversed(mp_rgb)]
    return cv_bgr

def rgba_to_rgb(rgba):
    """Convert RGBA color to RGB color and mask"""
    assert rgba.shape[-1] == 4

    rgb = np.empty((rgba.shape[0], rgba.shape[1], 3), dtype=rgba.dtype)
    r, g, b, m = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return [rgb, m]

def rgb_to_rgba(rgb, fill=1):
    """Convert RGB color to RGBA color"""
    assert rgb.shape[-1] == 3

    rgba = np.full((rgb.shape[0], rgb.shape[1], 4), fill, dtype=rgb.dtype)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    rgba[:,:,0] = r
    rgba[:,:,1] = g
    rgba[:,:,2] = b

    return rgba

def ensure_numeric_color(color, gradients=None, max_colors=None):
    """Ensures color is numeric and, if a vector requested, have specified length.
    The function performs the following conversions:
        if 'random' color name is specified, generates requred number of random colors,
        if 'gradient' color name is speciefied, computes a gradient vector,
        translates color from textual name ('red') to RGB value,
        if a list is provided, performs checks on its elements and
            guarantees it has required number of elements.
    Parameters:
        color       Either a color name, RGB value or special 'random' and 'gradient' strings
                    Could also be a list of names or RGB values
        gradients   Array-like of 2 colors to generate a gradient from.
                    Requred if `color` == `gradient`
        max_color   Requested number of colors (see return value).
                    Required if `color` == `gradient` or `random`.
    Returns:
        If `color` is a name or RGB value, result will be a vector of
        this color with `max_color` length or single value if `max_color` is None
        If `color` is `random`, result will be a list of colors with `max_color` length
        If `color` is `gradient`, result will be a list of colors for computed gradient
        If `color` is list, result will be a list with `max_colors` or original length
            with all elements converted as they are single values
    """
    ret_color = None
    if color is None:
        raise ValueError("Color not specified")

    if type(color) == list:
        # List of colors
        # Assume all items are of the same type
        # Make up a list of required length
        if len(color) == 0:
            raise ValueError("Empty color list")

        if max_colors is None:
            ret_color = color
        else:
            ret_color = color * int(np.ceil(max_colors / len(color)))
            ret_color = ret_color[0:max_colors]

    elif color == 'random':
        # Random color
        ret_color = random_colors(max_colors if max_colors is not None else 1)
        if max_colors is None:
            ret_color = ret_color[0]
    elif color == "gradient":
        # Gradient color
        if gradients is None or max_colors is None:
            raise ValueError("Cannot determine gradient of a single color")
        else:
            if len(gradients) < 2:
                raise ValueError("Two colors required to compute gradient")
            gc = (ensure_numeric_color(gradients[0]), ensure_numeric_color(gradients[1]))
            ret_color = gradient_colors(gc, max_colors)
    elif type(color) == str:
        # Named color
        ret_color = color_to_cv_color(color)
        if max_colors is not None:
            ret_color = [ret_color]
    else:
        # Should be a color
        ret_color = color
        if max_colors is not None:
            ret_color = [ret_color]

    return ret_color


# Modified version of imutils.four_point_transform() function
# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com
def four_point_transform(image, pts, inverse=False):
    """Perform 4-point transformation or reverses it"""

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
    """Apply CLAHE filter for luminocity equalization"""

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
    """Apply Gaussian filter"""
    s = 2
    w = 5
    t = (((w - 1)/2)-0.5)/s
    return gaussian_filter(img, sigma=s, truncate=t)

# Align two images
# Taken from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
# Author Satya Mallick
def align_images(im1, im2, warp_mode=cv2.MOTION_TRANSLATION, debug=False):
    """Algin two images.
    warp_mode is either cv2.MOTION_TRANSLATION for affine transformation or
    cv2.MOTION_HOMOGRAPHY for perspective one.
    Note that under OpenCV < 3.4.0 if images cannot be aligned, the function fails
    crashing the calling program (unless it uses global exception hook)"""

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
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix,
                                             motionType=warp_mode,
                                             criteria=criteria)

    try:
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        if debug:
            cv2.imshow('Aligned image', im2_aligned)
        return im2_aligned

    except:
        # Alignment unsuccessfull
        return im2

def get_diff(im1, im2, align=False, debug=False):
    """Get difference of two images"""

    # Algin images
    if align:
        im2 = align_images(im1, im2, cv2.MOTION_HOMOGRAPHY, debug)

    # Make up grays
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Calculate scrore and difference
    (score, diff) = measure.compare_ssim(gray1, gray2, full=True, gaussian_weights=True)
    return score, diff


def draw_contour(contour_img,
                 contour,
                 color,
                 gradient_colors=("red", "blue"),
                 min_size=10,
                 filled=True,
                 compute_mask=False):
    """ Draw a contour  as optionally fill it with color.

    Parameters:
        contour_img     An image to draw on
        contour         A list of contour points as returned by cv2.findContour() or
                        measure.find_contours()
        color           A color to draw contour or fill in if `filled` == True
                        Color could be RGB-numeric, color name ('red'), special
                        'random` and 'gradient` names or a list of such values
        gradient_colors A tuple or list to generate gradient if `color` == 'gradient'
        min_size        Minimum number of points in a contour to draw it
        filled          If True, contour fill be filled with specified color
        compute_mask    if True, a boolean mask of contour area will be computed
    Returns:
        contour_img     An image
        mask            Area mask or None if compute_mask is False
    """
    r = draw_contour_(contour_img, contour, color, gradient_colors, min_size, filled, compute_mask)
    return r[:2]

def draw_contour_(contour_img,
                 contour,
                 color,
                 gradient_colors=("red", "blue"),
                 min_size=10,
                 filled=True,
                 compute_mask=False):
    """ Draw a contour  as optionally fill it with color.
    Differs from draw_contour() in that in addition to contour_img and mask also
    returns perimeter, area, contour center and distance of each pixel of area in a center"""

    # Determine contour perimeter
    mask, perimeter, area, centroid, D = None, None, None, None, None
    try:
        rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1], shape=contour_img.shape)
        if max(rr.shape[0], cc.shape[0]) < min_size:
            return
        perimeter = np.column_stack((rr, cc))
    except IndexError:
        return None, None

    if not filled:
        # Contour not filled, draw it
        color = ensure_numeric_color(color)
        contour_img[rr, cc] = color
    else:
        # Get contour area
        rr, cc = draw.polygon(contour[:, 0], contour[:, 1], shape=contour_img.shape)
        if max(rr.shape[0], cc.shape[0]) < min_size:
            return
        area = np.column_stack((rr, cc))

        if compute_mask:
            mask = np.zeros((contour_img.shape[0], contour_img.shape[1]), dtype="uint8")
            mask[rr, cc] = True

        if color != "random" and color != "gradient" and type(color) != list:
            # If a solid color provided, simply fill the polygon
            color = ensure_numeric_color(color)
            contour_img[rr, cc] = color
        else:
            # Gradient, random colors or list of colors specified
            # Define contour center
            M = measure.moments_coords(perimeter)
            centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
            area_center = np.array([centroid])

            # Determine distance of each area point from the center
            D = distance.cdist(area, area_center, metric='euclidean')

            # Define colors
            num_colors = int(np.max(D)) + 1
            colours = ensure_numeric_color(color, gradient_colors, num_colors)

            # Fill polygon by colors specified by distance of each pixel from the center
            for n, p in enumerate(area):
                d = int(D[n])
                c = colours[d] if len(colours) > 1 else colours[0]
                contour_img[p[0], p[1]] = c

    if compute_mask:
        return contour_img, mask, perimeter, area, centroid, D
    else:
        return contour_img, None, perimeter, area, centroid, D

def update_contour(contour_img, color, params, gradient_colors=("red", "blue")):
    """Updates colors of previosly drawn contour.
    `params` should be result of draw_contour_() call
    """

    contour_img, mask, perimeter, area, centroid, D = params
    num_colors = int(np.max(D)) + 1
    colours = ensure_numeric_color(color, gradient_colors, num_colors)

    for n, p in enumerate(area):
        d = int(D[n])
        c = colours[d] if len(colours) > 1 else colours[0]
        contour_img[p[0], p[1]] = c


def get_max_contour_colors(contour, image_shape):
    """Determine number of colors to fill a contour with gradient or random colors"""

    # Determine contour perimeter
    try:
        rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1], shape=image_shape)
        perimeter = np.column_stack((rr, cc))

        M = measure.moments_coords(perimeter)
        centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
        area_center = np.array([centroid])

        # Determine distance of each point from the center
        D = distance.cdist(perimeter, area_center, metric='euclidean')

        # Get nunber of colors
        num_colors = int(np.max(D)) + 1
        return num_colors

    except IndexError:
        return None

def apply_patch(src, x, y, patch, mask=None, clip=False, alpha=None):
    """Applies a patch at given x, y (optionally masked)"""
    if y >= src.shape[0] or x  >= src.shape[1] or y < 0 or x < 0:
        raise Exception("Invalid coordinates")

    h, w = patch.shape[:2]
    if (y + h > src.shape[0] or x + w > src.shape[1]) and not clip:
        raise Exception("Patch is outside image area and clipping not specified")
    if y + h >= src.shape[0] and clip:
        h = src.shape[0] - y
    if x + w >= src.shape[1] and clip:
        w = src.shape[1] - x
        patch = patch[0:h, 0:w]
        mask = mask[0:h, 0:w]

    if patch.shape[-1] == 3 and src.shape[-1] == 4:
        patch = rgb_to_rgba(patch)

    dst = src.copy()
    area = dst[y:(y+h), x:(x+w)]
    if mask is None:
        if alpha is None:
            area = patch
        else:
            cv2.addWeighted(area, 1 - alpha, patch, alpha, 0, area)
    else:
        if alpha is None:
            area[mask > 0] = patch[mask > 0]
        else:
            dtyp = area.dtype
            a = area[mask > 0] * (1-alpha)
            p = patch[mask > 0] * alpha
            area[mask > 0] = a.astype(dtyp) + p.astype(dtyp)

    dst[y:(y+h), x:(x+w)] = area
    return dst


def get_parts(img, meta, apply_clahe=False, apply_filter=False, debug=False):
    """Internal function specific to cars-damage project"""

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
    """Align two images using parameters stored in meta dictionary"""
    parts, tr, patches, patches_resized = get_parts(img, meta, debug=debug)
    return align_images(patches_resized['left'], patches_resized['right'], warp_mode, debug)

def get_diff_meta(img, meta,
                  align=False,
                  color="red",
                  fill_contours=False,
                  apply_clahe=False,
                  apply_filter=False,
                  gradient_colors=("red", "blue"),
                  debug=False):
    """Get difference of two images using parameters stored in meta dictionary"""

    # Get image parts
    parts, tr, patches, patches_resized = get_parts(img, meta,
                                                    apply_clahe=apply_clahe,
                                                    apply_filter=apply_filter,
                                                    debug=debug)

    # Calculate scrore and difference
    (score, diff) = get_diff(patches_resized['left'], patches_resized['right'],
                             align=align, debug=debug)

    # Convert diff to image and threshold
    diff_img = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imshow('Thresh', thresh)

    # Find contours and draw them on the left patch
    # In addition create a binary difference mask
##    diff_img = patches_resized['left'].copy()
##    diff_mask = np.zeros((diff_img.shape[:2]), "uint8")

    diff_img = np.zeros(patches_resized['left'].shape, "uint8")

    contours = measure.find_contours(thresh, level=0.5, fully_connected="low", positive_orientation="low")
    for n, contour in enumerate(contours):
        mask = draw_contour(diff_img, contour, color,
                            gradient_colors=gradient_colors,
                            filled=fill_contours,
                            compute_mask=False)
##        if mask is not None:
##            diff_mask += mask

    if debug:
        cv2.imshow('Diff img', diff_img)

    # Apply diff image to patch
##    idx = (diff_mask != 0)
    patch_diff_img = patches_resized['left'].copy()
    idx = diff_img.any(axis=2)
    patch_diff_img[idx] = diff_img[idx]

    # Resize difference image back to original patch size
    diff_resized = cv2.resize(diff_img,
                              dsize=(patches['left'].shape[1], patches['left'].shape[0]),
                              interpolation=cv2.INTER_CUBIC)

    if debug:
        cv2.imshow('Diff resized', diff_resized)

    # Undo 4 point transformation
    # Top-left corner of original transform rect is a key point on target image
    rect = deepcopy(tr['left'])
    tl = order_points(tr['left'])[0]
    for r in rect:
        r[0] -= tl[0]
        r[1] -= tl[1]

    diff_tr = four_point_transform(diff_resized, rect, inverse=True)
    if debug:
        cv2.imshow('Diff transformed', diff_tr)

    # Transform diff image to patch of the same size as left image area
    result_img = parts['left']
    result_patch = np.zeros(result_img.shape, dtype=result_img.dtype)

    y, x = int(tl[1]), int(tl[0])
    dy, dx = diff_tr.shape[:2]
    rx = min(dx + x, result_img.shape[1])
    ry = min(dy + y, result_img.shape[0])
    dx = min(result_img.shape[1] - x, dx)
    dy = min(result_img.shape[0] - y, dy)

    result_patch[y:ry, x:rx] = diff_tr[0:dy, 0:dx]

    # Apply patch to left image area
##    idx = (result_patch != 0)
    idx = result_patch.any(axis=2)
    result_img[idx] = result_patch[idx]

    return score, patch_diff_img, result_img


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
