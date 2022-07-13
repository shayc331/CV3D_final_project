import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

FLANN_INDEX_KDTREE = 0

DIFF_SIDE_IM1_PATH = r'input\images\diff_side\left_sharon.jpg'
DIFF_SIDE_IM2_PATH = r'input\images\diff_side\right_sharon.jpg'

VIDEO_PATH_LEFT = os.path.join(os.getcwd(), r'input\diffView\3_view2.mp4')
VIDEO_PATH_RIGHT = os.path.join(os.getcwd(), r'input\diffView\3_view1.mp4')

VIDEO_OUT_PATH_LEFT = os.path.join(os.getcwd(), r'output\3_view2.mp4')
VIDEO_OUT_PATH_RIGHT = os.path.join(os.getcwd(), r'output\3_view1.mp4')

VIDEO_PATH_RIGHT_LAKER = os.path.join(os.getcwd(), r'input\diffView\1_view1.mp4')
VIDEO_PATH_LEFT_LAKER = os.path.join(os.getcwd(), r'input\diffView\1_view2.mp4')

# diff view points
# left
points_left = np.array([
    [1165, 468, 1],
    [676, 612, 1],
    [700, 253, 1],
    [663, 307, 1],
    [190, 177, 1],
    [775, 394, 1]

])
# right
points_right = np.array([
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
    return np.array(
        [[p1[0], p1[1], p1[2], 1, 0, 0, 0, 0, 0, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2], -p2[0]],
         [0, 0, 0, 0, p1[0], p1[1], p1[2], 1, 0, 0, 0, 0, -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2], -p2[1]],
         [0, 0, 0, 0, 0, 0, 0, 0, p1[0], p1[1], p1[2], 1, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], -p2[2]]])


# matrix 3 * 16


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
        [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u],
        [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]
    ])


def find_projection_matrix(points, real_points):
    H = np.ones((12, 12))
    for i, (p, real_p) in enumerate(zip(points, real_points)):
        a = get_eq(p, real_p)
        H[i * 2: i * 2 + 2] = a
    _, _, vt = np.linalg.svd(H)
    return vt[-1].reshape(3, 4)


def calculate_P1_P2():
    P_LEFT = find_projection_matrix(points_left, real_points)
    P_RIGHT = find_projection_matrix(points_right, real_points)
    return P_LEFT, P_RIGHT


def get_world_points(P, points):
    world_points = []
    inv_P = np.linalg.pinv(P)
    for p in points:
        X = inv_P @ p
        X[-1] = 1 if X[-1] == 0 else X[-1]  # TODO weird
        X /= X[-1]
        world_points.append(X)
    return world_points


def calibrate_cameras(P1, P2):
    # points we choose in the images with corresponding real points
    # same view im1:
    points1 = np.array([[753, 664, 1],
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

    # same view im2:
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

    A = np.array([x1[0] * P1[2] - P1[0],
                  x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0],
                  x2[1] * P2[2] - P2[1]])
    u, s, vt = np.linalg.svd(A)
    point = vt[-1]
    return point[:3] / point[3]


def get_obj_center(bbox):
    return bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)


def update_frame(frame, centroid, ok, bbox, timer):
    frame = cv2.circle(frame, (centroid[0], centroid[1]), radius=5, color=(0, 255, 0), thickness=-1)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                    2)

    # Display tracker type on frame
    cv2.putText(frame, " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow("Tracking", frame)


def track2(path, bbox, t):
    # Set up tracker.
    # Instead of MIL, you can also use
    centered_points = list()

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(path)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")


    for i in range(t):
        video.read()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')

    # Define an initial bounding box

    # Uncomment the line below to select a different bounding box


    # bbox = cv2.selectROI(frame, False)



    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        centered_points.append(get_obj_center(bbox))

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        # cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    video.release()
    cv2.destroyAllWindows()
    return centered_points


constraints = {
    "FAIL": [0, 5, 6, 9],
    "SUC": [0, 4, 3, 6]
}

DOTS = {
    "FAIL": {"left": [(350, 457), (502, 527), (749, 485), (660, 410)],
             "right": [(401, 382), (258, 399), (352, 409), (476, 387)]},
    "SUC": {"left": [(602, 448), (830, 428), (1103, 487), (907, 542)],
              "right": [(441, 398), (635, 418), (360, 467), (205, 427)]}
}


def check_point_constraints(point3d, frame, user, key):
    position_points = DOTS[user][key]
    const = constraints[user]
    if const[0] < point3d[0] < const[1] and const[2] < point3d[1] < const[3]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.line(frame, position_points[0], position_points[1], color=color, thickness=2)
    cv2.line(frame, position_points[1], position_points[2], color=color, thickness=2)
    cv2.line(frame, position_points[2], position_points[3], color=color, thickness=2)
    cv2.line(frame, position_points[3], position_points[0], color=color, thickness=2)


def show_results(path, points3d, bbox, user, key, out_path, t):
    tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(path)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")

    for i in range(t):
        video.read()

    # Read first frame.
    ok, frame = video.read()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    size = (frame.shape[0], frame.shape[1])
    out = cv2.VideoWriter(out_path, fourcc, 24, size)
    if not ok:
        print('Cannot read video file')

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    i = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        # Display tracker type on frame
        cv2.putText(frame, "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        check_point_constraints(points3d[i], frame, user, key)

        # Display result
        out.write(frame)
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
        i += 1
    out.release()
    cv2.destroyAllWindows()


def run(fp_l, fp_r, left_bbox, right_bbox, user, t=0):
    P_LEFT, P_RIGHT = calculate_P1_P2()
    # track(P_LEFT, P_RIGHT)
    left_centered = track2(fp_l, left_bbox, t)
    right_centered = track2(fp_r, right_bbox, t)
    points3d = [compute_3d(l, r, P_LEFT, P_RIGHT) for l, r in zip(left_centered, right_centered)]
    show_results(fp_l, points3d, left_bbox, user, "left", VIDEO_OUT_PATH_LEFT, t)
    show_results(fp_r, points3d, right_bbox, user, "right", VIDEO_OUT_PATH_RIGHT, t)



if __name__ == '__main__':
    run(VIDEO_PATH_LEFT, VIDEO_PATH_RIGHT, left_bbox=(743, 271, 61, 187), right_bbox=(438, 261, 74, 155),
        user="FAIL")
    run(VIDEO_PATH_LEFT_LAKER, VIDEO_PATH_RIGHT_LAKER, left_bbox=(770, 270, 104, 197), right_bbox=(454, 257, 61, 176),
        user="SUC", t=145)
