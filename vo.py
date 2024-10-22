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

        self.synchronized_imu = self.synchronize_data()
        
        self.orb = cv2.ORB_create()

        T_BS = self.leica_data['T_BS']['data']

        # Ініціалізація початкових координат
        self.initial_position = np.array([T_BS[3], T_BS[7], T_BS[11]])
        self.current_position = self.initial_position.copy()

        # Ініціалізація змінних для відстаней
        self.distance_leica = 0.0
        self.scale = 1.0

        self.gyroscope_noise_density = imu_data['gyroscope_noise_density']
        self.accelerometer_noise_density = imu_data['accelerometer_noise_density']
        self.rate_hz_imu = imu_data['rate_hz']

        # Ініціалізація параметрів камери
        self.intrinsics = camera_data['intrinsics']
        self.resolution = camera_data['resolution']
        self.rate_hz_camera = camera_data['rate_hz']

        self.previous_acceleration = np.array([0.0, 0.0, 0.0])
        self.previous_angular_velocity = np.zeros(3)
        self.previous_orientation = R.identity()
        self.current_orientation = None
        self.time_step = 1.0 / self.rate_hz_imu 
        self.time_step = 1.0 / 200
        self.previous_frame = None
        self.current_frame = None

        self.display_initialization_info()

    def display_initialization_info(self):
        print(f"Ініціалізовані параметри IMU: gyroscope_noise_density={self.gyroscope_noise_density}, "
              f"accelerometer_noise_density={self.accelerometer_noise_density}, rate_hz_imu={self.rate_hz_imu}")
        print(f"Ініціалізовані параметри камери: intrinsics={self.intrinsics}, "
              f"resolution={self.resolution}, rate_hz_camera={self.rate_hz_camera}")

    def distanceLeica(self):
        """Обчислює загальну відстань між усіма послідовними позиціями з Leica."""
        total_distance = 0.0

        for i in range(1, len(self.leica_csv_data)):
            pos1 = np.array([
                self.leica_csv_data[i - 1]['x'],
                self.leica_csv_data[i - 1]['y'],
                self.leica_csv_data[i - 1]['z']
            ])
            pos2 = np.array([
                self.leica_csv_data[i]['x'],
                self.leica_csv_data[i]['y'],
                self.leica_csv_data[i]['z']
            ])
            total_distance += np.linalg.norm(pos2 - pos1)

        return total_distance

    def integrate_imu_data(self):
        """Інтегрує дані IMU для обчислення загальної відстані та оновлення орієнтації."""
        total_distance = 0.0

        for i in range(1, len(self.imu_csv_data)):
            acceleration = np.array([
                self.imu_csv_data[i]['acceleration_x'],
                self.imu_csv_data[i]['acceleration_y'],
                self.imu_csv_data[i]['acceleration_z']
            ])

            angular_velocity = np.array([
                self.imu_csv_data[i]['angular_velocity_x'],
                self.imu_csv_data[i]['angular_velocity_y'],
                self.imu_csv_data[i]['angular_velocity_z']
            ])

            if i == 1:
                self.previous_acceleration = acceleration

            distance = self.updateIMU(acceleration, angular_velocity)
            total_distance += distance
            
            self.previous_acceleration = acceleration
            self.previous_angular_velocity = angular_velocity

        return total_distance

    def updateIMU(self, acceleration, angular_velocity):
        """Оновлює швидкість, позицію та орієнтацію за даними IMU."""
        # Масштабування шуму за частотою IMU
        noise_scale = np.sqrt(1 / self.rate_hz_imu)

        # Додавання шуму до прискорення та кутової швидкості
        noisy_acceleration = (
            acceleration + 
            np.random.normal(0, self.accelerometer_noise_density * noise_scale, size=3)
        )
        noisy_angular_velocity = (
            angular_velocity + 
            np.random.normal(0, self.gyroscope_noise_density * noise_scale, size=3)
        )

        # Оновлення орієнтації на основі гіроскопічних даних
        rotation = R.from_rotvec(noisy_angular_velocity * self.time_step)
        self.previous_orientation = self.current_orientation  # Зберігаємо поточну орієнтацію перед оновленням
        self.current_orientation = self.previous_orientation * rotation

        # Обчислення нової швидкості
        new_velocity = self.previous_acceleration * self.time_step + noisy_acceleration * self.time_step

        # Інтеграція швидкості для обчислення нових координат
        displacement = new_velocity * self.time_step

        # Обчислення загальної відстані
        total_distance = np.linalg.norm(displacement)

        return total_distance

    def compute_trajectory(self):
        # Обчислення 3D координат об'єкта з монокулярної візуальної одометрії
        trajectory = []  # Список для збереження траєкторії
        
        # Перевірка, чи є self.camera_data ітерабельним об'єктом
        if not hasattr(self.camera_data, '__iter__'):
            raise TypeError("self.camera_data must be an iterable (e.g., list or array).")
        
        # Зберегти попередній кадр
        self.previous_frame = None

        for frame_idx in range(len(self.camera_data)):
            # Отримати дані зображення
            current_frame = self.camera_data[frame_idx]

            if frame_idx > 0:
                # Виконати обробку зображень та обчислення зміни
                dx_camera, dy_camera, dz_camera = self.compute_motion(current_frame)  # Ваш алгоритм, який працює з поточним фреймом
                
                # Оновити координати з IMU
                self.updateIMU(self.previous_acceleration, self.previous_angular_velocity)  # Оновлюємо координати за допомогою IMU
                dx_imu, dy_imu, dz_imu = self.previous_acceleration  # Отримати зміщення з IMU

                # Додавання інформації з камери та IMU
                self.current_position += np.array([dx_camera, dy_camera, dz_camera]) + np.array([dx_imu, dy_imu, dz_imu])

            # Зберегти обчислені координати для подальшого аналізу
            trajectory.append(self.current_position.copy())

            print(f"Frame {frame_idx}: Position = {self.current_position}")

            # Оновлення попередньої рамки
            self.previous_frame = current_frame  # Оновити попередній кадр для наступної ітерації

        return np.array(trajectory)


    def compute_scale(self):
        """
        Обчислює масштаб, використовуючи реальні відстані між позиціями з Leica 
        та відстань, інтегровану з IMU.
        """
        # Перевірка наявності даних Leica
        if len(self.leica_csv_data) < 2:
            raise ValueError("Недостатньо даних Leica для обчислення відстані.")

        # Обчислення реальної відстані між позиціями з Leica
        self.distance_leica = self.distanceLeica()  # Відстань між всіма позиціями

        # Інтеграція даних прискорення з IMU
        self.distance_imu = self.integrate_imu_data()

        if self.distance_imu == 0:
            raise ValueError("Відстань IMU дорівнює нулю, не можна обчислити масштаб.")

        # Перевірка на нульову відстань Leica
        if self.distance_leica == 0:
            raise ValueError("Відстань Leica дорівнює нулю, не можна обчислити масштаб.")

        # Обчислення масштабу
        self.scale = self.distance_leica / self.distance_imu


    def compute_motion(self, current_frame):
        """Обчислює нові координати на основі поточної та попередньої рамки."""
        if self.previous_frame is None:
            # Зберегти першу рамку та повернути початкові координати
            self.previous_frame = current_frame
            return 0, 0, 0

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
        if len(matches) < 5:  # Перевірка на кількість матчів
            return 0, 0, 0

        src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Центрування точок
        src_pts_mean = np.mean(src_pts, axis=0)
        dst_pts_mean = np.mean(dst_pts, axis=0)
        
        src_pts_centered = src_pts - src_pts_mean
        dst_pts_centered = dst_pts - dst_pts_mean

        # Масштабування (опційно)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(src_pts_centered, axis=1))
        src_pts_scaled = src_pts_centered * scale
        dst_pts_scaled = dst_pts_centered * scale

        # Виконати естімування руху
        essential_matrix, _ = cv2.findEssentialMat(src_pts_scaled, dst_pts_scaled, focal=self.intrinsics[0], pp=(self.intrinsics[2], self.intrinsics[3]))
        _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts_scaled, dst_pts_scaled)

        self.compute_scale()

        # Повертаємо зміщення (dx, dy, dz)
        dx, dy, dz = (t.flatten() * self.scale)  # Використовуємо лише t, оскільки Z тут ігнорується

        # Оновлюємо попередню рамку
        self.previous_frame = current_frame

        return dx, dy, dz

    def synchronize_data(self):
        """Синхронізує часові ряди даних IMU та камери за часовими мітками."""
        # Отримання часових міток з CSV-файлів
        imu_timestamps = np.array([row['timestamp'] for row in self.imu_csv_data])
        camera_timestamps = np.array([row['timestamp'] for row in self.camera_csv_data])

        # Інтерполяція даних IMU до частоти камери
        interpolated_acceleration_x = np.interp(camera_timestamps, imu_timestamps, 
                                                np.array([row['acceleration_x'] for row in self.imu_csv_data]))
        interpolated_acceleration_y = np.interp(camera_timestamps, imu_timestamps, 
                                                np.array([row['acceleration_y'] for row in self.imu_csv_data]))
        interpolated_acceleration_z = np.interp(camera_timestamps, imu_timestamps, 
                                                np.array([row['acceleration_z'] for row in self.imu_csv_data]))
        
        interpolated_angular_velocity_x = np.interp(camera_timestamps, imu_timestamps, 
                                                    np.array([row['angular_velocity_x'] for row in self.imu_csv_data]))
        interpolated_angular_velocity_y = np.interp(camera_timestamps, imu_timestamps, 
                                                    np.array([row['angular_velocity_y'] for row in self.imu_csv_data]))
        interpolated_angular_velocity_z = np.interp(camera_timestamps, imu_timestamps, 
                                                    np.array([row['angular_velocity_z'] for row in self.imu_csv_data]))

        # Збирання всіх інтерпольованих даних в масиви
        interpolated_acceleration = np.vstack((interpolated_acceleration_x, 
                                                interpolated_acceleration_y, 
                                                interpolated_acceleration_z)).T

        interpolated_angular_velocity = np.vstack((interpolated_angular_velocity_x, 
                                                    interpolated_angular_velocity_y, 
                                                    interpolated_angular_velocity_z)).T

        return interpolated_acceleration, interpolated_angular_velocity

