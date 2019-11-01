import cv2
import numpy as np


def stitch_by_H(img1, img2, H):
    """Use the key points to stitch the images.
    img1: the image containing frames that have been joint before.
    img2: the newly selected key frame.
    H: Homography matrix, usually from compute_homography(img1, img2).
    """
    # Get heights and widths of input images
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # Store the 4 ends of each original canvas
    img1_canvas_orig = np.float32([[0, 0], [0, h1],
                                   [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_canvas = np.float32([[0, 0], [0, h2],
                              [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # The 4 ends of (perspective) transformed img1
    img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

    # Get the coordinate range of output (0.5 is fixed for image completeness)
    output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
    [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

    # The output matrix after affine transformation
    transform_array = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

    # Warp the perspective of img1
    img_output = cv2.warpPerspective(img1, transform_array.dot(H),
                                     (x_max - x_min, y_max - y_min))

    for i in range(-y_min, h2 - y_min):
        for j in range(-x_min, w2 - x_min):
            if np.any(img2[i + y_min][j + x_min]):
                img_output[i][j] = img2[i + y_min][j + x_min]
    return img_output


# Inspired from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
def compute_homography(img1, img2):
    """Find SIFT matches and return the estimated Homography matrix."""
    # Call the SIFT method
    sift = cv2.xfeatures2d.SIFT_create()

    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    img1_crop = img1[:, w1 - w2:]  # Crop the right part of img1 for detecting SIFT
    diff = np.size(img1, axis=1) - np.size(img1_crop, axis=1)

    # Detect and compute key points and descriptors
    kp1, des1 = sift.detectAndCompute(img1_crop, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use the Brute-Force matcher to obtain matches
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)  # Using L2 (Euclidean) distance
    matches = bf.knnMatch(des1, des2, k=2)

    # Define a Valid Match: whose distance is less than match_ratio times the
    # distance of the second best nearest neighbor.
    match_ratio = 0.6

    # Pick up valid matches
    valid_matches = []
    for m1, m2 in matches:
        if m1.distance < match_ratio * m2.distance:
            valid_matches.append(m1)

    min_match_num = 4  # Minimum number of matches (to ensure a good stitch)
    if len(valid_matches) > min_match_num:
        # Extract the coordinates of matching points
        img1_pts = []
        img2_pts = []
        for match in valid_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)

        # Formalize as matrices (for the sake of computing Homography)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img1_pts[:, :, 0] += diff  # Recover its original coordinates
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute the Homography matrix
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        # Recompute the Homography in order to improve robustness
        for i in range(mask.shape[0] - 1, -1, -1):
            if mask[i] == 0:
                np.delete(img1_pts, [i], axis=0)
                np.delete(img2_pts, [i], axis=0)

        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        return H
    else:
        # Print a warning message, but compute Homography anyway as long as
        # len(valid_matches) > 4, which is the least requirement of computing
        # Homography.
        pass


def stitch_image(img1_path, img2_path):
    # Input images
    if isinstance(img1_path, str):
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (640, 360))
        img1 = cylindrical_project(img1)
    else:
        img1 = img1_path

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (640, 360))

    # Apply cylindrical projection to the new frame
    img2 = cylindrical_project(img2)
    H = compute_homography(img1, img2)

    stitched_image = stitch_by_H(img1, img2, H)

    stitched_image = crop_black(stitched_image)

    return stitched_image


def crop_black(img):
    """Crop off the black edges."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)

    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
                   best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        img_crop = img

    return img_crop


# Equations are from https://www.cnblogs.com/cheermyang/p/5431170.html
def cylindrical_project(img):
    """Inverse interpolation is applied to find the cylindrical projection of
    the original image.
    """
    cylindrical_img = np.zeros_like(img)
    height, width, depth = cylindrical_img.shape

    f = 550.2562584220408  # focal length

    centerX = width / 2
    centerY = height / 2

    # pointX, pointY are coordinates in planar axis;
    # i, j, k are coordinates in cylindrical axis.
    for i in range(width):
        for j in range(height):
            theta = (i - centerX) / f
            pointX = int(f * np.tan((i - centerX) / f) + centerX)
            pointY = int((j - centerY) / np.cos(theta) + centerY)

            for k in range(depth):
                if 0 <= pointX < width and 0 <= pointY < height:
                    cylindrical_img[j, i, k] = img[pointY, pointX, k]
                else:
                    cylindrical_img[j, i, k] = 0

    return cylindrical_img


# Below are our attempts for linear blending and Laplacian blending.
"""
def stitch_by_H_blend(img1, img2, H):
    # Get heights and widths of input images
    w1, h1, depth = img1.shape
    w2, h2 = img2.shape[0:2]

    # Store the 4 ends of each original canvas
    img1_canvas = np.float32([[0, 0], [0, w1],
                              [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_canvas_orig = np.float32([[0, 0], [0, w2],
                                   [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # The 4 ends of (perspective) transformed img2
    if H is None:
        H = compute_homography(img1, img1)
    img2_canvas = cv2.perspectiveTransform(img2_canvas_orig, H)

    # Get the coordinate range of output (0.5 is fixed for image completeness)
    output_canvas = np.concatenate((img1_canvas, img2_canvas), axis=0)
    [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

    # The output matrix after affine transformation
    transform_array = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

    # Warp the perspective of img2
    img_output = cv2.warpPerspective(img2, transform_array.dot(H),
                                     (x_max - x_min, y_max - y_min))

    # # Average blending: If a pixel in img1 contains information,
    # #                   copy it to img_output.
    # for i in range(-y_min, w1 - y_min):
    #     for j in range(-x_min, h1 - x_min):
    #         if np.any(img1[i + y_min][j + x_min]):
    #             if np.any(img_output[i][j]):
    #                 for k in range(depth):
    #                     img_output[i][j][k] = np.uint8(
    #                         (int(img1[i + y_min][j + x_min][k])
    #                          + int(img_output[i][j][k])) / 2)
    #             else:
    #                 img_output[i][j] = img1[i + y_min][j + x_min]

    # # Linear blending: Weighted average for transition area.
    # alpha = 0.8
    # for i in range(-y_min, w1 - y_min):
    #     for j in range(-x_min, h1 - x_min):
    #         if np.any(img1[i + y_min][j + x_min]):
    #             if np.any(img_output[i][j]):
    #                 img_output[i][j] = (alpha * img1[i + y_min][j + x_min]
    #                                     + (1 - alpha) * img_output[i][j])
    #             else:
    #                 img_output[i][j] = img1[i + y_min][j + x_min]

    # Laplacian blending: Build Gaussian pyramids and Laplacian pyramids
    # for both images, and join them together on each resolution level, then sum
    # them up to get the final result.
    img2_blending = img_output.copy()
    for i in range(-y_min, w1 - y_min):
        for j in range(-x_min, h1 - x_min):
            if np.any(img1[i + y_min][j + x_min]):
                if np.any(img2_blending[i][j]):
                    continue
                else:
                    img2_blending[i][j] = img1[i + y_min][j + x_min]
    mask = np.ones_like(img2, dtype='float32')
    mask = cv2.warpPerspective(mask, transform_array.dot(H),
                               (x_max - x_min, y_max - y_min))
    img2_blending = img2_blending[-y_min:w1 - y_min, -x_min:h1 - x_min]
    mask_blending = mask[-y_min:w1 - y_min, -x_min:h1 - x_min]
    output_blending = Laplacian_blending(img2_blending, img1, mask_blending, 6)
    img_output[-y_min:w1 - y_min, -x_min:h1 - x_min] = output_blending
    return img_output
"""


# From https://gist.github.com/royshil/0b20a7dd08961c83e210024a4a7d841a for test
"""
def Laplacian_blending(img1, img2, mask, levels=4):
    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gp1.append(np.float32(G1))
        gp2.append(np.float32(G2))
        gpM.append(np.float32(GM))

    # Generate Laplacian Pyramids for A, B and masks
    lp1 = [gp1[levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2 = [gp2[levels - 1]]
    gpMr = [gpM[levels - 1]]
    for i in range(levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        L1 = np.subtract(gp1[i - 1], cv2.pyrUp(gp1[i]))
        L2 = np.subtract(gp2[i - 1], cv2.pyrUp(gp2[i]))
        lp1.append(L1)
        lp2.append(L2)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpMr):
        ls = l1 * gm + l2 * (1.0 - gm)
        LS.append(ls)

    # Now reconstruct
    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_
"""
