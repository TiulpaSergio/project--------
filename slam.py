import cv2
import numpy as np

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

        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kalman.transitionMatrix[0:3, 0:3] = np.eye(3)
        self.kalman.transitionMatrix[0:3, 3:6] = np.eye(3)
        self.kalman.transitionMatrix[3:6, 3:6] = np.eye(3)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kalman.statePost = np.zeros((6, 1), dtype=np.float32)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = self.detect_features(gray)
            return

        keypoints, descriptors = self.detect_features(gray)

        if descriptors is None:
            print("Дескриптори для поточного кадру не знайдені.")
            return

        matches = self.match_features(self.prev_descriptors, descriptors)

        if len(matches) == 0:
            print("Відповідності не знайдені.")
            return

        matched_pts_prev, matched_pts_curr = self.get_matched_points(matches, self.prev_keypoints, keypoints)

        if len(matched_pts_prev) >= 8:
            self.localize_camera(matched_pts_prev, matched_pts_curr)
        else:
            print("Недостатня кількість відповідних точок для локалізації.")

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def detect_features(self, gray):
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:
            print("Дескриптори не були обчислені.")
        else:
            print(f"Знайдено ключових точок: {len(keypoints)}")
        return np.array([kp.pt for kp in keypoints]), descriptors


    def match_features(self, descriptors1, descriptors2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    def get_matched_points(self, matches, keypoints1, keypoints2):
        matched_pts_prev = np.float32([keypoints1[m.queryIdx] for m in matches])
        matched_pts_curr = np.float32([keypoints2[m.trainIdx] for m in matches])
        return matched_pts_prev, matched_pts_curr

    def localize_camera(self, pts_prev, pts_curr):
        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.RANSAC, 0.1, 0.99)
        if F is None:
            print("Не вдалося знайти фундаментальну матрицю.")
            return

        pts_prev = pts_prev[mask.ravel() == 1]
        pts_curr = pts_curr[mask.ravel() == 1]

        if len(pts_prev) < 8:
            print("Недостатня кількість точок для локалізації камери.")
            return
        
        E, mask_e = cv2.findEssentialMat(pts_prev, pts_curr, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("Не вдалося знайти основну матрицю.")
            return

        print(f"Матриця E до очищення: {E}")

        pts_prev = pts_prev[mask_e.ravel() == 1]
        pts_curr = pts_curr[mask_e.ravel() == 1]

        if len(pts_prev) < 8:
            print("Недостатня кількість точок після очищення для локалізації камери.")
            return

        _, R, t, mask_r = cv2.recoverPose(E, pts_prev, pts_curr)
        if R is None or t is None:
            print("Не вдалося відновити матрицю обертання та вектор переміщення.")
            return

        if R.shape != (3, 3) or t.shape != (3, 1):
            print("Невірні розміри матриці R або вектора t.")
            return

        self.camera_position += t

        self.trajectory.append(self.camera_position.copy())
        self.update_global_position(R, t)

        measurement = self.camera_position[:3].reshape(-1, 1).astype(np.float32)
        self.kalman.correct(measurement)
        self.kalman.predict()

        filtered_position = self.kalman.statePost[:3]

        print(f"Матриця обертання R: {R}")
        print(f"Вектор переміщення t: {t}")
        print(f"Матриця E: {E}")
        print(f"Залишилось точок: {len(pts_prev)}")
        print(f"Оновлена позиція камери: {self.camera_position.T}")
        print(f"Відфільтрована позиція: {filtered_position.T}")
        print(f"Глобальні координати: {self.global_position.T}")

    def update_global_position(self, R, t):
        self.global_position += R @ t
        self.global_trajectory.append(self.global_position.copy())

        if self.global_position.shape != (3, 1):
            print("Невірні розміри глобальних координат.")

    def get_global_position(self):
        return self.global_position.copy()
    
    def get_camera_position(self):
        return self.camera_position.copy()

    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def get_global_trajectory(self):
        return np.array(self.global_trajectory)
