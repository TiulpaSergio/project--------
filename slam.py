import cv2
import numpy as np
from feature_detection import detect_features, match_features, get_matched_points

class SLAM:
    def __init__(self, imu_yaml, camera_yaml, leica_yaml, imu_csv, camera_csv, leica_csv):
        self.orb = cv2.ORB_create()
        self.camera_position = np.zeros((3, 1))
        self.global_position = np.zeros((3, 1))
        self.global_trajectory = []
        self.trajectory = []
        self.velocity = np.zeros((3))
        self.position = np.zeros(3)
        self.gravity = np.array([[0], [0], [9.81]])
        self.imu_data = imu_yaml
        self.camera_data = camera_yaml
        self.leica_data = leica_yaml
        self.imu_csv_data = imu_csv
        self.camera_csv_data = camera_csv
        self.leica_csv_data = leica_csv
        self.imu_index = 0
        self.leica_index = 0
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.imu_rate_hz = self.imu_data['rate_hz']
        self.camera_rate_hz = self.camera_data['rate_hz']
        self.leica_T_BS = np.array(self.leica_data['T_BS']['data']).reshape(4, 4)
        initial_position = np.array([
            self.leica_T_BS[0, 3],  # x з 4-ї позиції
            self.leica_T_BS[1, 3],  # y з 8-ї позиції
            self.leica_T_BS[2, 3]  # z з 12-ї позиції
        ])
        self.p_RS_R = initial_position        
        self.init_kalman()
        self.init_bundle_adjustment()
        self.process_data()

    def process_data(self): 
        self.imu_csv_data.columns = ['timestamp', 'w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z', 'a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']
        self.leica_csv_data.columns = ['timestamp', 'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z']
        self.camera_csv_data.columns = ['timestamp', 'filename']
        # Доступ до даних IMU
        self.imu_timestamps = self.imu_csv_data['timestamp'].values  # Час
        self.imu_w_x = self.imu_csv_data['w_RS_S_x'].values  # Кутова швидкість по осі X
        self.imu_w_y = self.imu_csv_data['w_RS_S_y'].values  # Кутова швидкість по осі Y
        self.imu_w_z = self.imu_csv_data['w_RS_S_z'].values  # Кутова швидкість по осі Z
        self.imu_a_x = self.imu_csv_data['a_RS_S_x'].values  # Прискорення по осі X
        self.imu_a_y = self.imu_csv_data['a_RS_S_y'].values  # Прискорення по осі Y
        self.imu_a_z = self.imu_csv_data['a_RS_S_z'].values  # Прискорення по осі Z
        # Доступ до даних Leica
        self.leica_timestamps = self.leica_csv_data['timestamp'].values  # Час
        self.leica_position_x = self.leica_csv_data['p_RS_R_x'].values  # Координата X
        self.leica_position_y = self.leica_csv_data['p_RS_R_y'].values  # Координата Y
        self.leica_position_z = self.leica_csv_data['p_RS_R_z'].values  # Координата Z
        # Доступ до даних камери
        self.camera_timestamps = self.camera_csv_data.iloc[:, 0].values
        self.camera_images = self.camera_csv_data.iloc[:, 1].values

    def init_kalman(self):
        self.kalman = cv2.KalmanFilter(9, 6)
        self.kalman.measurementMatrix = np.eye(6, 9, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(9, dtype=np.float32)
        self.kalman.transitionMatrix[0:3, 3:6] = np.eye(3)
        self.kalman.transitionMatrix[3:6, 6:9] = np.eye(3)
        self.kalman.processNoiseCov = np.eye(9, dtype=np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-1
        self.kalman.statePost = np.zeros((9, 1), dtype=np.float32)

    def init_bundle_adjustment(self):
        pass

    def process_leica(self, leica_row):
        # Використовуємо нові дані Leica
        self.p_RS_R = np.array([
            leica_row['p_RS_R_x'],
            leica_row['p_RS_R_y'],
            leica_row['p_RS_R_z']
        ])  # Отримуємо координати (форма (3,))

        # Оновлюємо глобальну позицію даними з Leica
        self.absolute_position = self.position + self.p_RS_R

    def integrate_imu(self, imu_row):
        a_RS_S = np.array([
            imu_row['a_RS_S_x'],
            imu_row['a_RS_S_y'],
            imu_row['a_RS_S_z']
        ])

        # Корекція прискорення з урахуванням гравітації
        a_RS_S_corrected = a_RS_S - self.gravity.flatten()

        # Визначення dt
        if self.imu_index > 0:
            dt = (self.imu_timestamps[self.imu_index] - self.imu_timestamps[self.imu_index - 1]) / 1e9  # Наносекунди в секунди
        else:
            dt = 1 / self.imu_rate_hz

        # Перевірка на адекватне значення dt (часовий інтервал між кадрами)
        if dt <= 0:
            raise ValueError(f"Invalid dt: {dt}")

        # Інтеграція прискорення для отримання швидкості
        self.velocity += a_RS_S_corrected * dt

        # Інтеграція швидкості для отримання позиції
        self.position += self.velocity * dt

        # Додавання початкової позиції з Leica
        self.absolute_position = self.position + self.p_RS_R

        # Вивід абсолютної позиції
        print(f"Абсолютна позиція: {self.absolute_position.flatten()}")

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.imu_index < len(self.imu_csv_data):
            imu_row = self.imu_csv_data.iloc[self.imu_index]
            self.integrate_imu(imu_row)  # Обробка даних IMU
            self.imu_index += 1

        if self.leica_index < len(self.leica_csv_data):
            leica_row = self.leica_csv_data.iloc[self.leica_index]
            self.process_leica(leica_row)  # Обробка даних Leica
            self.leica_index += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = detect_features(self.orb, gray)
            return

        keypoints, descriptors = detect_features(self.orb, gray)
        if descriptors is None or len(keypoints) < 50:
            return

        matches = match_features(self.prev_descriptors, descriptors)
        if len(matches) < 50:
            return

        matched_pts_prev, matched_pts_curr = get_matched_points(matches, self.prev_keypoints, keypoints)
        self.localize_camera(matched_pts_prev, matched_pts_curr)

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def localize_camera(self, pts_prev, pts_curr):
        # Обчислення матриці гомографії з використанням RANSAC
        H, mask_h = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 5.0)
        
        if H is not None:
            print("Матриця гомографії:")
            print(H)

            # Перевірка, чи підходить гомографія для опису руху
            if self.is_planar_motion(mask_h):
                print("Рух камери в межах площини. Використання гомографії.")
                self.apply_homography(H)
                return  # Якщо гомографія коректна, завершуємо функцію

        # Якщо гомографія не підходить, переходимо до обчислення Essential Matrix
        E, mask_e = cv2.findEssentialMat(pts_prev, pts_curr, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask_e) < 10:
            return

        # Відновлення обертання та трансляції
        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr)

        if np.linalg.norm(t) > 0.01:
            self.update_positions(R, t)

    def is_planar_motion(self, mask_h):
        """Перевірка, чи більшість точок узгоджуються з гомографією."""
        inliers = np.sum(mask_h)
        return inliers > 0.8 * len(mask_h)

    def apply_homography(self, H):
        """Застосування гомографії для оновлення позиції камери."""
        translation = H[:2, 2]  # Отримання трансляції з гомографії
        self.camera_position[:2] += translation.reshape(2, 1)
        self.trajectory.append(self.camera_position.copy())

    def update_positions(self, R, t):
        self.camera_position += t
        self.trajectory.append(self.camera_position.copy())

        # Використовуємо нові дані для корекції калмана
        measurement = np.hstack((self.camera_position[:3].flatten(), self.velocity)).astype(np.float32)
        self.kalman.correct(measurement)  # Корекція
        self.kalman.predict()  # Прогноз
        filtered_position = self.kalman.statePost[:3]
        self.camera_position[:3] = filtered_position
        self.update_global_position(R, t)

        # Розрахунок абсолютного масштабу
        scale = np.linalg.norm(t)  # Довжина вектора переміщення
        print(f"Абсолютний масштаб (довжина вектора переміщення): {scale}")

    def update_global_position(self, R, t):
        self.global_position += R @ t
        self.global_trajectory.append(self.global_position.copy())

    def filter_data(self, data):
        return data

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
