import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class VO:
    def __init__(self, imu_data, camera_data, leica_data, imu_csv_data, camera_csv_data, leica_csv_data):
        self.imu_data = imu_data  # Дані IMU
        self.camera_data = camera_data  # Дані камери
        self.leica_data = leica_data  # Дані Leica
        self.imu_csv_data = imu_csv_data  # CSV дані IMU
        self.camera_csv_data = camera_csv_data  # CSV дані камери
        self.leica_csv_data = leica_csv_data  # CSV дані Leica

        self.orb = cv2.ORB_create()

        T_BS = self.leica_data['T_BS']['data']

        # Ініціалізація початкових координат
        self.initial_position = np.array([T_BS[3], T_BS[7], T_BS[11]])
        self.current_position = self.initial_position.copy()

        # Ініціалізація змінних для відстаней
        self.distance_leica = 0.0
        self.distance_imu = 0.0
        self.scale = 1.0

        self.gyroscope_noise_density = imu_data['gyroscope_noise_density']
        self.accelerometer_noise_density = imu_data['accelerometer_noise_density']
        self.rate_hz_imu = imu_data['rate_hz']

        # Ініціалізація параметрів камери
        self.intrinsics = camera_data['intrinsics']
        self.resolution = camera_data['resolution']
        self.rate_hz_camera = camera_data['rate_hz']

        # Ініціалізація початкових координат з даних Leica
        self.p_RS_R_x = leica_data['T_BS']['data'][3]  # Витягуємо x-координату
        self.p_RS_R_y = leica_data['T_BS']['data'][7]  # Витягуємо y-координату
        self.p_RS_R_z = leica_data['T_BS']['data'][11] # Витягуємо z-координату

        self.display_initialization_info()

        self.previous_acceleration = np.array([0.0, 0.0, 0.0])  # Зберігання попередніх значень прискорення
        self.time_step = 1.0 / 200

        # Змінні для зберігання попередніх координат
        self.previous_frame = None
        self.current_frame = None
        self.previous_coords = np.array([self.p_RS_R_x, self.p_RS_R_y, self.p_RS_R_z])

    def display_initialization_info(self):
        print(f"Ініціалізовані параметри IMU: gyroscope_noise_density={self.gyroscope_noise_density}, "
              f"accelerometer_noise_density={self.accelerometer_noise_density}, rate_hz_imu={self.rate_hz_imu}")
        print(f"Ініціалізовані параметри камери: intrinsics={self.intrinsics}, "
              f"resolution={self.resolution}, rate_hz_camera={self.rate_hz_camera}")
        print(f"Початкові координати: p_RS_R_x={self.p_RS_R_x}, p_RS_R_y={self.p_RS_R_y}, p_RS_R_z={self.p_RS_R_z}")

    def distanceLeica(self):
        """Обчислює відстань між першими двома позиціями з Leica."""
        # Перевірка наявності даних Leica
        if len(self.leica_data) < 2:
            raise ValueError("Недостатньо даних Leica для обчислення відстані.")

        # Витягуємо перші дві позиції з даних Leica
        position_1 = self.leica_data[0]
        position_2 = self.leica_data[1]

        # Обчислення відстані між позиціями
        distance = np.linalg.norm(position_2 - position_1)
        return distance

    def integrate_imu_data(self):
        # Обчислення загальної відстані на основі IMU
        total_distance = 0.0

        for i in range(1, len(self.imu_csv_data)):
            # Отримати прискорення з CSV
            acceleration = np.array([
                self.imu_csv_data[i]['acceleration_x'],
                self.imu_csv_data[i]['acceleration_y'],
                self.imu_csv_data[i]['acceleration_z']
            ])
            # Виклик методу updateIMU
            distance = self.updateIMU(acceleration)
            total_distance += distance

        return total_distance

    def updateIMU(self, acceleration):
        """Оновлює зміщення за даними IMU та обчислює відстань."""
        # Інтеграція прискорення для отримання зміщення
        self.previous_acceleration += acceleration * self.time_step
        dx, dy, dz = self.previous_acceleration

        # Додавання шуму
        noise_x = np.random.normal(0, self.accelerometer_noise_density)
        noise_y = np.random.normal(0, self.accelerometer_noise_density)
        noise_z = np.random.normal(0, self.accelerometer_noise_density)

        # Обчислення загальної відстані
        total_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        self.previous_coords += (self.previous_acceleration + np.array([noise_x, noise_y, noise_z])) * self.time_step

        return total_distance

    def compute_trajectory(self):
        # Обчислення 3D координат об'єкта з монокулярної візуальної одометрії
        trajectory = []  # Список для збереження траєкторії

        for frame_idx in range(len(self.camera_data)):
            # Отримати дані зображення
            current_frame = self.camera_data[frame_idx]

            if frame_idx > 0:
                # Виконати обробку зображень та обчислення зміни
                dx_camera, dy_camera, dz_camera = self.compute_motion(current_frame)  # Ваш алгоритм, який працює з поточним фреймом
                
                # Оновити координати з IMU
                self.updateIMU(self.previous_acceleration)  # Оновлюємо координати за допомогою IMU
                dx_imu, dy_imu, dz_imu = self.previous_acceleration  # Отримати зміщення з IMU

                # Додавання інформації з камери та IMU
                self.current_position += np.array([dx_camera, dy_camera, dz_camera]) + np.array([dx_imu, dy_imu, dz_imu])

            # Зберегти обчислені координати для подальшого аналізу
            trajectory.append(self.current_position.copy())

            # Оновлення попередньої рамки
            self.previous_frame = current_frame

        return np.array(trajectory)

    def compute_scale(self):
        """
        Обчислює масштаб, використовуючи реальні відстані між позиціями з Leica 
        та відстань, інтегровану з IMU.
        """
        # Перевірка наявності даних Leica
        if len(self.leica_data) < 2:
            raise ValueError("Недостатньо даних Leica для обчислення відстані.")

        # Обчислення реальної відстані між позиціями з Leica
        self.distance_leica = np.linalg.norm(self.leica_data[1] - self.leica_data[0])  # Відстань між першими двома позиціями

        # Інтеграція даних прискорення з IMU
        self.distance_imu = self.integrate_imu_data()

        # Обчислення масштабу
        if self.distance_imu > 0:  # Запобігання діленню на нуль
            self.scale = self.distance_leica / self.distance_imu
        else:
            raise ValueError("Відстань IMU дорівнює нулю, не можна обчислити масштаб.")

    def compute_motion(self, current_frame):
        """Обчислює нові координати на основі поточної та попередньої рамки."""
        if self.previous_frame is None:
            # Зберегти першу рамку та повернути початкові координати
            self.previous_frame = current_frame
            return self.previous_coords

        # Перетворити зображення в відтінки сірого
        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Використання ORB для виявлення ключових точок та описувачів
        keypoints_prev, descriptors_prev = self.orb.detectAndCompute(gray_prev, None)
        keypoints_curr, descriptors_curr = self.orb.detectAndCompute(gray_curr, None)

        # Знайти відповідності між ключовими точками
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_prev, descriptors_curr)

        # Витягнути координати відповідних точок
        src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Виконати естімування руху
        essential_matrix, _ = cv2.findEssentialMat(src_pts, dst_pts, focal=self.intrinsics[0], pp=(self.intrinsics[2], self.intrinsics[3]))
        _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts, dst_pts)

        # Повертаємо зміщення (dx, dy, dz)
        dx, dy, dz = t.flatten()  # Використовуємо лише t, оскільки Z тут ігнорується

        # Оновлюємо попередню рамку
        self.previous_frame = current_frame

        return dx, dy, dz

    def output_results(self):
        """Виводить результати обчислень."""
        print("Результати обчислень:")
        print(f"Поточні координати: {self.current_position}")
        print(f"Відстань з Leica: {self.distance_leica:.2f} м")
        print(f"Відстань з IMU: {self.distance_imu:.2f} м")
        print(f"Обчислений масштаб: {self.scale:.2f}")

    def detect_features(self, frame):
        """Виявлення ключових точок і обчислення дескрипторів (SIFT/ORB)."""
        # Приклад коду для виявлення ознак за допомогою ORB (можна замінити на SIFT)
        orb = self.orb
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        return keypoints, descriptors

    def match_features(self, descriptors_prev, descriptors_curr):
        """Сопоставлення дескрипторів між попередньою та поточною рамкою."""
        # Використовуємо Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_prev, descriptors_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
