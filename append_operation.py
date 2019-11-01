"""Image stitching is implemented in two parts separately, namely the left and
the right half of the view. And cylindrical projection is implemented during
stitching, so two intermediate results are not horizontal.
Once the two stitched parts have been generated, apply 4-point perspective
transformation to both of them to get two horizontal images. Then join them
together to obtain the complete panoramic image.
"""
import cv2
import numpy as np


def append_image(img1, img2):
    img1 = cv2.resize(img1, (1200, 360))
    img2 = cv2.resize(img2, (1200, 360))

    pts1 = np.float32([[60, 120], [1200, 30], [50, 280], [1200, 300]])
    pts2 = np.float32([[1, 1], [1200, 0], [1, 360], [1200, 360]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    img1 = cv2.warpPerspective(img1, M, (1200, 360))

    pts1 = np.float32([[20, 150], [1200, 30], [1, 270], [1200, 340]])
    pts2 = np.float32([[0, 0], [1200, 0], [0, 360], [1200, 360]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    img2 = cv2.warpPerspective(img2, M, (1200, 360))

    vis = np.hstack((img1, img2))

    return vis
