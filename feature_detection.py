import cv2
import numpy as np

def detect_keypoints(detector, frame_gray):
    keypoints = detector.detect(frame_gray, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

def track_keypoints(old_gray, frame_gray, p0, lk_params):
    p1, st = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_new, good_old
    return None, None

def update_keypoints_if_needed(p0, frame_gray, detector, threshold=300):
    if len(p0) < threshold:
        new_points = detect_keypoints(detector, frame_gray)
        if len(new_points) > 0:
            p0 = np.vstack((p0, new_points))
    return p0

def detect_features(orb, frame_gray):
    keypoints = orb.detect(frame_gray, None)
    keypoints, descriptors = orb.compute(frame_gray, keypoints)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

def get_matched_points(matches, kp1, kp2):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
