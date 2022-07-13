import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


FLANN_INDEX_KDTREE = 0
IM1_PATH = os.path.join(os.getcwd(), r'input\images\same_side\1_05.jpg')
IM2_PATH = os.path.join(os.getcwd(), r'input\images\same_side\2_05.jpg')
CHESS_PATH = os.path.join(os.getcwd(), r'input\chess\chess.jpeg')
CHESS_NO_CROP_PATH = os.path.join(os.getcwd(), r'input\chess\chess_no_crop.jpeg')
CHESS_3 = os.path.join(os.getcwd(), r'input\chess\chess_3.jpeg')
RIGHT_PATH = os.path.join(os.getcwd(), r'input\chess\right.jpeg')
LEFT_PATH = os.path.join(os.getcwd(), r'input\chess\left.jpeg')

DIFF_SIDE_IM1_PATH = r'input\images\diff_side\left_sharon.jpg'
DIFF_SIDE_IM2_PATH = r'input\images\diff_side\right_sharon.jpg'

# points we choose in the images with corresponding real points
# same view im1:
# points1 = np.array([[753, 664, 1],
#                     [67, 413, 1],
#                     [1097, 484, 1],
#                     [1127, 378, 1],
#                     [1197, 282, 1],
#                     # [930, 339, 1],
#                     # [886, 377, 1]
#                     ])
#                     # [571, 274, 1]])
#
# real_points = np.array([[0, 0, 0, 1],
#                         [0, 8, 0, 1],
#                         [9, 0, 0, 1],
#                         [9, -0.3, 1.42, 1],  # lower net
#                         [9, -0.91, 2.66, 1],  # pole - amood
#                         # [9, 2.2, 1.7, 1],  # itamar
#                         # [4.9, 2, 1.65, 1]
#                         ])
#                         # [8, 10, 2.66, 1]])  # the corresponding real points
# same view im2:
# points2 = np.array([[575, 513, 1],
#                     [169, 410, 1],
#                     [103, 390, 1],
#                     [1053, 291, 1],
#                     [1106, 202, 1],
#                     # [884, 274, 1],
#                     # [791, 281, 1]
#                     ])
#                     # [592, 259, 1]])


# diff view points
# left
points1 = np.array([
    [1165, 468, 1],
    [676, 612, 1],
    [700, 253, 1],
    [663, 307, 1],
    [190, 177, 1],
    [775, 394, 1]


])
# right
points2 = np.array([
    [552, 516, 1],
    [148, 417, 1],
    [1095, 186, 1],
    [1019, 278, 1],
    [609, 257, 1],
    [949, 396, 1]
])

real_points = np.array([
    [0, 0, 0, 1],
    [0, 9, 0, 1],
    [8, -0.91, 2.66, 1],  # pole
    [8, 0, 1.44, 1],  # lower net
    [8, 9, 2.44, 1],  # upper net
    [5, 0, 0, 1]  # 3 line

])

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

    H1 = findHomography(world_points1, real_points)  # works with MSE to find optimal H
    H2 = findHomography(world_points2, real_points)

    return P1 @ np.linalg.inv(H1), P2 @ np.linalg.inv(H2)



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



if __name__ == '__main__':
    # nice_try()
    img1 = cv2.imread(DIFF_SIDE_IM1_PATH, 0)
    img2 = cv2.imread(DIFF_SIDE_IM2_PATH, 0)

    # cv2.imshow("1", img1)
    # cv2.waitKey(0)
    # F = get_fundamental(img1, img2)
    P1, P2 = calculate_P1_P2()

    sanity_point = np.array([0, 4.5, 0, 1])
    middle_court = P1 @ sanity_point
    print(middle_court[:2] / middle_court[2])
    # P1, P2 = get_cameras_matrix(F)
    # P1M, P2M = calibrate_cameras(P1, P2)

    # todo - manualy set image coordinates of the ball.
    point1 = np.array([592, 22, 1])
    point2 = np.array([445, 77, 1])

    point_11 = np.array([296, 495, 1])
    point_22 = np.array([303, 444, 1])

    left_ball = np.array([1151, 221, 1])
    right_ball = np.array([444, 40, 1])

    a = compute_3d(left_ball, right_ball, P1, P2)
    print(a[:3] / a[3])

    # X = compute_3d(point1, point2, P1M, P2M)
    X = compute_3d(point1, point2, P1, P2)
    print(X[:3] / X[3])