import cv2
import numpy as np
from feature_detection import detect_features, match_features, get_matched_points

class SLAM:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.map_points = []
        self.camera_position = np.zeros((3, 1))
        self.global_position = np.zeros((3, 1))
        self.global_trajectory = []
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.trajectory = []
        self.init_kalman()

    def init_kalman(self):
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kalman.transitionMatrix[0:3, 3:6] = np.eye(3)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kalman.statePost = np.zeros((6, 1), dtype=np.float32)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = detect_features(self.orb, gray)
            return

        keypoints, descriptors = detect_features(self.orb, gray)
        if descriptors is None or len(keypoints) < 50:
            print("Недостатня кількість точок для обробки (потрібно мінімум 50).")
            return

        matches = match_features(self.prev_descriptors, descriptors)
        if len(matches) < 50:
            print("Недостатня кількість відповідностей (мінімум 50).")
            return

        matched_pts_prev, matched_pts_curr = get_matched_points(matches, self.prev_keypoints, keypoints)
        self.localize_camera(matched_pts_prev, matched_pts_curr)

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def localize_camera(self, pts_prev, pts_curr):
        # Локалізація камери з використанням фундаментальної та основної матриць
        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.RANSAC, 0.1, 0.99)
        pts_prev, pts_curr = pts_prev[mask.ravel() == 1], pts_curr[mask.ravel() == 1]

        E, mask_e = cv2.findEssentialMat(pts_prev, pts_curr, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask_e) < 10:
            print("Не вдалося знайти основну матрицю або недостатньо точок.")
            return

        _, R, t, mask_r = cv2.recoverPose(E, pts_prev, pts_curr)
        if np.linalg.norm(t) > 0.01:
            self.update_positions(R, t)

    def update_positions(self, R, t):
        self.camera_position += t
        self.trajectory.append(self.camera_position.copy())

        measurement = self.camera_position[:3].reshape(-1, 1).astype(np.float32)
        self.kalman.correct(measurement)
        self.kalman.predict()

        filtered_position = self.kalman.statePost[:3]
        self.camera_position[:3] = filtered_position

        self.update_global_position(R, t)

    def update_global_position(self, R, t):
        self.global_position += R @ t
        self.global_trajectory.append(self.global_position.copy())

    def get_trajectory(self):
        return np.array(self.trajectory)

    def get_global_trajectory(self):
        return np.array(self.global_trajectory)

    def get_global_position(self):
        print(f"Глобальні координати: {self.global_position.T}")
        return self.global_position.copy()

    def get_camera_position(self):
        print(f"Оновлена позиція камери: {self.camera_position.T}")
        return self.camera_position.copy()