# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


FLANN_INDEX_KDTREE = 0
IM1_PATH = os.path.join(os.getcwd(), r'input\images\same_side\1_05.jpg')
IM2_PATH = os.path.join(os.getcwd(), r'input\images\same_side\2_05.jpg')

# points we choose in the images with corresponding real points
# same view im1:
points1 = np.array([[753, 664, 1],
                    [65, 415, 1],
                    [1097, 484, 1],
                    [1127, 378, 1],
                    [1194, 281, 1],
                    [930, 339, 1]])

real_points = np.array([[0, 0, 0, 1],
                        [0, 9, 0, 1],
                        [8, 0, 0, 1],
                        [8, 0, 1.42, 1],
                        [8, -0.91, 2.44, 1],
                        [9, 2.2, 1.7, 1]])  # the corresponding real points

# same view im2:
points2 = np.array([[575, 513, 1],
                    [172, 410, 1],
                    [103, 390, 1],
                    [1053, 291, 1],
                    [1107, 199, 1],
                    [884, 274, 1]])


def findMatchingPoints(kp1, des1, kp2, des2, d):
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < d * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts2 = np.int32(pts2)
    pts1 = np.int32(pts1)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    if F is not None:
        print(F)

    # We select only first 15 points
    if mask is not None:
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

    return pts1, pts2, F

def get_fundamental(img1, img2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    key, des = kp2[0], des2[0]

    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    desc1 = []
    desc2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Draw the keypoint matches between both pictures
    # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask[300:500],
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
    plt.axis("off")
    plt.imshow(keypoint_matches)
    plt.show()
    cv2.imshow("Keypoint matches", keypoint_matches)
    # cv2.waitKey(0)

    pts1, pts2, F = findMatchingPoints(kp1, des1, kp2, des2, 0.7)
    return F


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.) """

    # return null space of F (Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0])
        from a fundamental matrix. """

    e = compute_epipole(F.T)  # left epipole
    Te = skew(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T


def get_cameras_matrix(F):
    P1 = np.append(np.eye(3), np.array([0, 0, 0]).reshape(-1, 1), axis=1)
    P2 = compute_P_from_fundamental(F)
    return P1, P2


def get_constraints(p1, p2):
    return np.array([[p1[0], p1[1], p1[2], 1, 0,0,0,0, 0,0,0,0,-p2[0]*p1[0],-p2[0]*p1[1],-p2[0]*p1[2], -p2[0]],
              [0,0,0,0, p1[0], p1[1], p1[2], 1,0,0,0,0, -p2[1]*p1[0],-p2[1]*p1[1],-p2[1]*p1[2], -p2[1]],
              [0,0,0,0,0,0,0,0, p1[0], p1[1], p1[2], 1, -p2[2]*p1[0],-p2[2]*p1[1],-p2[2]*p1[2], -p2[2]]])
#matrix 3 * 16


def findHomography(points, real_points):
    H = np.ones((18, 16))
    for i, (p, real_p) in enumerate(zip(points, real_points)):
        a = get_constraints(p, real_p)
        H[i * 3: i * 3 + 3] = a
    _, _, vt = np.linalg.svd(H)
    return vt[-1].reshape(4, 4)


def get_eq(p_img, p_real):
    X, Y, Z, _ = p_real
    u, v, _ = p_img
    return np.array([
        [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u],
        [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    ])


def find_projection_matrix(points, real_points):
    H = np.ones((12, 12))
    for i, (p, real_p) in enumerate(zip(points, real_points)):
        a = get_eq(p, real_p)
        H[i * 2: i * 2 + 2] = a
    _, _, vt = np.linalg.svd(H)
    return vt[-1].reshape(3, 4)


def calculate_P1_P2():
    P1 = find_projection_matrix(points1, real_points)
    P2 = find_projection_matrix(points2, real_points)
    return P1, P2

def get_world_points(P, points):
    world_points = []
    inv_P = np.linalg.pinv(P)
    for p in points:
        X = inv_P @ p
        X[-1] = 1 if X[-1] == 0 else X[-1] # TODO weird
        X /= X[-1]
        world_points.append(X)
    return world_points


def calibrate_cameras(P1,P2):
    # points we choose in the images with corresponding real points
    #same view im1:
    points1 = np.array([[753,664, 1],
                        [65, 415, 1],
                        [1097, 484, 1],
                        [1127, 378, 1],
                        [1194, 281, 1],
                        [930, 339, 1]])

    world_points1 = get_world_points(P1, points1)
    real_points = np.array([[0, 0, 0, 1],
                             [0, 9, 0, 1],
                             [8, 0, 0, 1],
                             [8, 0, 1.42, 1],
                             [8, -0.91, 2.44, 1],
                             [9, 2.2, 1.7, 1]])  # the corresponding real points

    #same view im2:
    points2 = np.array([[575, 513, 1],
                        [172, 410, 1],
                        [103, 390, 1],
                        [1053, 291, 1],
                        [1107, 199, 1],
                        [884, 274, 1]])  # points we choose in the images with corresponding real points
    world_points2 = get_world_points(P2, points2)


    # P1 = find_projection_matrix(points1, real_points)
    #todo: see if this method will work for 4*4 points....
    H1 = findHomography(world_points1, real_points)  # works with MSE to find optimal H
    H2 = findHomography(world_points2, real_points)

    return P1 @ np.linalg.inv(H1), P2 @ np.linalg.inv(H2)
    # return P1 @ H1, P2 @ H2


def compute_3d(x1, x2, P1, P2):
    """
    computes the coordinate of the point X corresponding with x, x'
    :param x:
    :param x_: x'
    :param P1:
    :param P2:
    :return:
    """

    A = np.array([x1[0]*P1[2]-P1[0],
              x1[1]*P1[2]-P1[1],
              x2[0]*P2[2]-P2[0],
              x2[1]*P2[2]-P2[1]])
    u, s, vt = np.linalg.svd(A)
    return vt[-1]



def calibrate_camera_matrix():
    img = cv2.imread(f"chess.jpg", 1)

    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:7, 0:6].T.reshape(-1, 2)) * 30  # the board squares are 30 cm
    print(objp)
    # Color-segmentation to get binary mask
    lwr = np.array([0, 0, 180])  # need to cut the image to include only the board
    upr = np.array([179, 61, 252])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)

    # Extract chess-board
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)

    # Displaying chess-board features
    res = np.uint8(res)
    # call imshow() using plt object
    plt.imshow(res, cmap='gray')

    # display that image
    plt.show()
    ret, corners = cv2.findChessboardCorners(res, (7, 6),
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        print(corners)
        # fnl = cv2.drawChessboardCorners(img, (7, 6), corners, ret)
        # cv2.imshow("fnl", fnl)
        # cv2.waitKey(0)
    else:
        print("No Checkerboard Found")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], res.shape[::-1], None, None)
    print(f"mtx: {mtx}")
    print(f"dist: {dist}")
    print(f"rvecs: {rvecs}")
    print(f"tvecs: {tvecs}")

if __name__ == '__main__':
    img1 = cv2.imread(IM1_PATH, 0)
    img2 = cv2.imread(IM2_PATH, 0)

    # F = get_fundamental(img1, img2)
    P1, P2 = calculate_P1_P2()
    # P1, P2 = get_cameras_matrix(F)
    # P1M, P2M = calibrate_cameras(P1, P2)

    # todo - manualy set image coordinates of the ball.
    point1 = np.array([592, 22, 1])
    point2 = np.array([445, 77, 1])

    # X = compute_3d(point1, point2, P1M, P2M)
    X = compute_3d(point1, point2, P1, P2)
    print(X[:3] / X[3])
    # the height will be the z coordinate of X.



