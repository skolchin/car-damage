import cv2
import numpy as np
from img_utils import *

##from skimage import measure
##from skimage import draw
##from scipy.spatial.distance import cdist
##from random import randint
##from matplotlib import colors

FILE_NAME = "1.jpg"

img = cv2.imread(FILE_NAME)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
contours = measure.find_contours(thresh, 0.8)

contour_img = np.full(img.shape, (255, 255, 255), "uint8")
for c in contours:
    draw_contour(contour_img, c, "gradient", filled=True)

cv2.imshow('Contour', contour_img)
cv2.waitKey()

cv2.destroyAllWindows()


##def random_colors(n):
##    """Returns n random colors"""
##    rr = []
##    for i in range(n):
##        r = randint(0,255)
##        g = randint(0,255)
##        b = randint(0,255)
##        rr.extend([(r,g,b)])
##    return rr
##
##def color_gradient(colors, n):
##    c = np.linspace(0, 1, n)[:, None, None]
##    x = np.array([colors[0]])
##    y = np.array([colors[1]])
##    g = x + (y - x) * c
##    g = g.astype(x.dtype)
##    return g
##
##def color_to_cv_color(name):
##    mp_rgb = colors.to_rgb(name)
##    cv_bgr = [c * 255 for c in reversed(mp_rgb)]
##    return cv_bgr
##
##img = cv2.imread(IMG_DIR + FILE_NAME)
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
##
##contour_img = np.full(img.shape, (255, 255, 255), "uint8")
##
##contours = measure.find_contours(thresh, 0.8)
##
##for contour in contours:
##    rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1], shape=contour_img.shape)
##    if max(rr.shape[0], cc.shape[0]) < 10:
##        continue
##
##    contour_img[rr, cc] = (0, 0, 0)
##
##    perimeter = np.column_stack((rr, cc))
##    M = measure.moments_coords(perimeter)
##    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
##
##    cv2.circle(contour_img, (int(centroid[1]), int(centroid[0])), 2, (0, 0, 255), -1)
##
##    rr, cc = draw.polygon(contour[:, 0], contour[:, 1], shape=contour_img.shape)
##    if max(rr.shape[0], cc.shape[0]) < 10:
##        continue
##
##    area = np.column_stack((rr, cc))
##    area_center = np.array([centroid])
##
##    Dp = cdist(perimeter, area_center, metric='euclidean')
##    D = cdist(area, area_center, metric='euclidean')
##
##    print(perimeter.shape, area.shape, area_center.shape, Dp.shape, D.shape)
##
##    #colours = random_colors(int(np.max(D))+1)
##    colours = color_gradient( [color_to_cv_color("green"), color_to_cv_color("blue")], int(np.max(D))+1)
##
##    for n, p in enumerate(area):
##        d = int(D[n])
##        c = colours[d]
##        contour_img[p[0], p[1]] = c


##
####im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
####
####cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 1)
####
####for c in contours:
####    m = cv2.moments(c)
####
####    cx = int(np.ceil(m['m10'] / m['m00']))
####    cy = int(np.ceil(m['m01'] / m['m00']))
####
####    cv2.circle(contour_img, (cy, cx), 2, (0, 0, 255), -1)
##
####x, y = np.ogrid[-np.pi:np.pi:200j, -np.pi:np.pi:200j]
####r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
####
####contours = measure.find_contours(r, 0.8)
####
####contour_img = np.zeros((r.shape[0], r.shape[1], 3), "uint8")
####
####for n, c in enumerate(contours):
####    rr, cc = polygon(c[:, 0], c[:, 1], shape=contour_img.shape)
####    #contour_img[rr, cc] = [255, 255, 255]
####
####    pp = np.column_stack((rr, cc))
####    for p in pp:
####        contour_img[p[0], p[1]] = (255, 255, 255)
####
####    m = cv2.moments(pp)
####
####    cx = int(np.ceil(m['m10'] / m['m00']))
####    cy = int(np.ceil(m['m01'] / m['m00']))
####
####    cv2.circle(contour_img, (cy, cx), 2, (0, 0, 255), -1)

