import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class VO:
    def __init__(self, imu_data, camera_data, leica_data, imu_csv_data, camera_csv_data, leica_csv_data):
        self.imu_data = imu_data
        self.camera_data = camera_data
        self.leica_data = leica_data
        self.imu_csv_data = imu_csv_data
        self.camera_csv_data = camera_csv_data
        self.leica_csv_data = leica_csv_data

        self.synchronized_imu_acceleration, self.synchronized_imu_angular_velocity = self.synchronize_data()
        
        self.orb = cv2.ORB_create()

        first_leica_record = leica_csv_data.iloc[0]
        self.initial_position = np.array([
            first_leica_record[1],
            first_leica_record[2],
            first_leica_record[3]
        ])
        self.current_position = self.initial_position.copy()
        self.previous_position = self.initial_position.copy()
        self.trajectory = [self.initial_position.copy()]

        self.distance_leica = 0.0
        self.scale = 1.0

        self.gyroscope_noise_density = imu_data['gyroscope_noise_density']
        self.accelerometer_noise_density = imu_data['accelerometer_noise_density']
        self.rate_hz_imu = imu_data['rate_hz']

        self.intrinsics = camera_data['intrinsics']
        self.resolution = camera_data['resolution']
        self.rate_hz_camera = camera_data['rate_hz']

        self.previous_acceleration = np.zeros(3)
        self.previous_velocity = np.zeros(3)
        self.current_orientation = R.identity()
        self.time_step = 1.0 / self.rate_hz_imu
        self.previous_frame = None
        self.current_frame = None
        self.current_frame_index = 0
        self.total_distance = 0.0

        self.display_initialization_info()

    def display_initialization_info(self):
        print(f"Ініціалізовані параметри IMU: gyroscope_noise_density={self.gyroscope_noise_density}, "
              f"accelerometer_noise_density={self.accelerometer_noise_density}, rate_hz_imu={self.rate_hz_imu}")
        print(f"Ініціалізовані параметри камери: intrinsics={self.intrinsics}, "
              f"resolution={self.resolution}, rate_hz_camera={self.rate_hz_camera}")

    def distanceLeica(self):
        """Обчислює відстань між позиціями на поточному та наступному кадрах з Leica."""
        # Перевірка, чи є наступний кадр доступним
        if self.current_frame_index < len(self.leica_csv_data) - 1:
            position1 = self.leica_csv_data.iloc[self.current_frame_index, 1:4].values
            position2 = self.leica_csv_data.iloc[self.current_frame_index + 1, 1:4].values

            # Обчислюємо відстань між цими позиціями
            distance = np.linalg.norm(position2 - position1)

            # Переходимо до наступного кадру
            self.current_frame_index += 1

            return distance
        else:
            raise ValueError("Немає наступного кадру для обчислення відстані.")

    def integrate_imu_data(self):
        """Інтегрує дані IMU з контролем масштабування та вдосконаленим розрахунком зміщення."""
        
        # Використання синхронізованих даних IMU
        acceleration_data = np.mean(self.synchronized_imu_acceleration, axis=0)
        angular_velocity_data = self.synchronized_imu_angular_velocity

        # Додавання шуму до прискорення та кутової швидкості
        noise_scale = np.sqrt(1 / self.rate_hz_imu)
        noisy_acceleration = acceleration_data + np.random.normal(
            0, self.accelerometer_noise_density * noise_scale, acceleration_data.shape
        )
        noisy_angular_velocity = angular_velocity_data + np.random.normal(
            0, self.gyroscope_noise_density * noise_scale, angular_velocity_data.shape
        )

        # Обчислення нових орієнтацій
        rotations = R.from_rotvec(noisy_angular_velocity * self.time_step)
        self.current_orientation = rotations * self.current_orientation  
        global_acceleration = self.current_orientation.apply(noisy_acceleration)

        # Масштабування прискорення, якщо воно занадто мале
        acceleration_magnitude = np.linalg.norm(global_acceleration)
        min_acceleration_threshold = 1e-5

        if acceleration_magnitude < min_acceleration_threshold:
            global_acceleration *= (min_acceleration_threshold / acceleration_magnitude)

        # Можливе збільшення time_step для покращення зміщення
        scaled_time_step = self.time_step * 10  # Пробуйте різні значення, наприклад, 5 або 10

        # Розрахунок нових швидкостей з використанням прискорення
        new_velocities = self.previous_velocity + global_acceleration * scaled_time_step

        # Обчислення зміщення
        displacements = new_velocities * scaled_time_step

        # Накопичення загальної відстані
        self.total_distance += np.linalg.norm(displacements)

        # Оновлення попередніх значень
        self.previous_velocity = new_velocities.copy()
        self.previous_acceleration = noisy_acceleration.copy()

        # Виведення важливих даних для дебагу
        # print(f"Total Distance: {self.total_distance}")
        # print(f"Noisy Acceleration: {noisy_acceleration}")
        # print(f"Noisy Angular Velocity: {noisy_angular_velocity}")
        # print(f"New Velocities: {new_velocities}")
        # print(f"Displacements: {displacements}")

        return self.total_distance

    def update_position(self, dx_camera, dy_camera, dz_camera, dx_imu, dy_imu, dz_imu):
        """Оновлює поточну позицію з урахуванням початкової."""
        # Обчислення зміщення відносно IMU та камери
        displacement = (
            np.array([dx_camera, dy_camera, dz_camera]) +
            np.array([dx_imu, dy_imu, dz_imu])
        )

        # Оновлення поточної позиції відносно початкової
        self.current_position = self.previous_position + displacement

        # Розрахунок пройденої відстані
        distance = np.linalg.norm(self.current_position - self.previous_position)
        self.total_distance += distance

        # Оновлення попередньої позиції
        self.previous_position = self.current_position.copy()

        print(f"Updated Position: {self.current_position}, Distance Traveled: {distance}")

    def compute_trajectory(self, current_frame):
        """Обчислює траєкторію на основі поточного кадру та IMU."""
        if self.previous_frame is None:
            self.previous_frame = current_frame  # Ініціалізація першого кадру
            return np.array(self.trajectory)

        # Переконатися, що масштаб обчислено перед використанням
        self.compute_scale()

        # Обчислення зміщень на основі поточного та попереднього кадрів
        dx_camera, dy_camera, dz_camera = self.compute_motion(current_frame)

        # Адаптивне масштабування прискорення
        acceleration_mean = np.mean(np.abs(self.previous_acceleration))
        scaling_factor = max(acceleration_mean * 10, 1e-4)  # Масштаб для корекції
        scaled_acceleration = self.previous_acceleration * scaling_factor
        dx_imu, dy_imu, dz_imu = scaled_acceleration * self.time_step

        print(f"Camera Motion: dx={dx_camera}, dy={dy_camera}, dz={dz_camera}")
        print(f"Scaled IMU Motion: dx={dx_imu}, dy={dy_imu}, dz={dz_imu}")

        # Оновлення позиції з урахуванням камери та IMU
        self.update_position(dx_camera, dy_camera, dz_camera, dx_imu, dy_imu, dz_imu)

        # Додавання поточної позиції до траєкторії
        self.trajectory.append(self.current_position.copy())

        print(f"Trajectory: {self.trajectory}")

        # Оновлення попереднього кадру
        self.previous_frame = current_frame
        return np.array(self.trajectory)

    def compute_scale(self):
        """Обчислює масштаб, використовуючи реальні відстані між позиціями з Leica та відстань, інтегровану з IMU."""
        if len(self.leica_csv_data) < 2:
            raise ValueError("Недостатньо даних Leica для обчислення відстані.")
        
        self.distance_leica = self.distanceLeica()
        self.distance_imu = self.integrate_imu_data()

        print(f"Distance Leica: {self.distance_leica}")
        print(f"Distance IMU: {self.distance_imu}")

        if self.distance_imu == 0:
            raise ValueError("Відстань IMU дорівнює нулю, не можна обчислити масштаб.")
        if self.distance_leica == 0:
            raise ValueError("Відстань Leica дорівнює нулю, не можна обчислити масштаб.")

        self.scale = self.distance_leica / self.distance_imu

    def compute_motion(self, current_frame):
        """Обчислює нові координати на основі поточної та попередньої рамки."""
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return 0, 0, 0

        keypoints_prev, descriptors_prev = self.orb.detectAndCompute(self.previous_frame, None)
        keypoints_curr, descriptors_curr = self.orb.detectAndCompute(current_frame, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_prev, descriptors_curr)

        if len(matches) < 5:
            return 0, 0, 0

        src_pts = np.float32([keypoints_prev[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_curr[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Центрування точок
        src_pts_mean = np.mean(src_pts, axis=0)
        dst_pts_mean = np.mean(dst_pts, axis=0)

        src_pts_centered = src_pts - src_pts_mean
        dst_pts_centered = dst_pts - dst_pts_mean

        scale = np.sqrt(2) / np.mean(np.linalg.norm(src_pts_centered, axis=1))
        src_pts_scaled = src_pts_centered * scale
        dst_pts_scaled = dst_pts_centered * scale

        essential_matrix, _ = cv2.findEssentialMat(src_pts_scaled, dst_pts_scaled, focal=self.intrinsics[0], pp=(self.intrinsics[2], self.intrinsics[3]))
        _, R, t, _ = cv2.recoverPose(essential_matrix, src_pts_scaled, dst_pts_scaled)

        self.compute_scale()
        dx, dy, dz = (t.flatten() * self.scale)

        print(f"Translation (t.flatten()): {t.flatten()}")
        print(f"Scale: {self.scale}")

        self.previous_frame = current_frame

        return dx, dy, dz

    def synchronize_data(self):
        """Синхронізує часові ряди даних IMU та камери за часовими мітками."""
        
        # Отримання часових міток з CSV-файлів за допомогою числових індексів
        imu_timestamps = self.imu_csv_data.iloc[:, 0].values  # Перший стовпець (timestamp)
        camera_timestamps = self.camera_csv_data.iloc[:, 0].values  # Перший стовпець (timestamp) для camera_csv_data

        # Логування отриманих часових міток
        # print(f"IMU Timestamps: {imu_timestamps[:5]}...")  # Виведення перших 5 значень
        # print(f"Camera Timestamps: {camera_timestamps[:5]}...")  # Виведення перших 5 значень

        # Інтерполяція даних IMU до частоти камери
        interpolated_acceleration_x = np.interp(camera_timestamps, imu_timestamps, 
                                                self.imu_csv_data.iloc[:, 1].values)  # Другий стовпець (acceleration_x)
        interpolated_acceleration_y = np.interp(camera_timestamps, imu_timestamps, 
                                                self.imu_csv_data.iloc[:, 2].values)  # Третій стовпець (acceleration_y)
        interpolated_acceleration_z = np.interp(camera_timestamps, imu_timestamps, 
                                                self.imu_csv_data.iloc[:, 3].values)  # Четвертий стовпець (acceleration_z)

        interpolated_angular_velocity_x = np.interp(camera_timestamps, imu_timestamps, 
                                                    self.imu_csv_data.iloc[:, 4].values)  # П'ятий стовпець (angular_velocity_x)
        interpolated_angular_velocity_y = np.interp(camera_timestamps, imu_timestamps, 
                                                    self.imu_csv_data.iloc[:, 5].values)  # Шостий стовпець (angular_velocity_y)
        interpolated_angular_velocity_z = np.interp(camera_timestamps, imu_timestamps, 
                                                    self.imu_csv_data.iloc[:, 6].values)  # Сьомий стовпець (angular_velocity_z)

        # Логування інтерпольованих даних
        # print(f"Interpolated Acceleration X: {interpolated_acceleration_x[:5]}...")  # Виведення перших 5 значень
        # print(f"Interpolated Acceleration Y: {interpolated_acceleration_y[:5]}...")
        # print(f"Interpolated Acceleration Z: {interpolated_acceleration_z[:5]}...")
        # print(f"Interpolated Angular Velocity X: {interpolated_angular_velocity_x[:5]}...")
        # print(f"Interpolated Angular Velocity Y: {interpolated_angular_velocity_y[:5]}...")
        # print(f"Interpolated Angular Velocity Z: {interpolated_angular_velocity_z[:5]}...")

        # Збирання всіх інтерпольованих даних в масиви
        interpolated_acceleration = np.vstack((interpolated_acceleration_x, 
                                                interpolated_acceleration_y, 
                                                interpolated_acceleration_z)).T

        interpolated_angular_velocity = np.vstack((interpolated_angular_velocity_x, 
                                                    interpolated_angular_velocity_y, 
                                                    interpolated_angular_velocity_z)).T

        return interpolated_acceleration, interpolated_angular_velocity
