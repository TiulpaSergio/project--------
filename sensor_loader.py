import yaml
import pandas as pd

class SensorLoader:
    """Клас для завантаження та зберігання сенсорних параметрів та даних."""

    def __init__(self, imu_path, camera_path, leica_path, imu_data_path, camera_data_path, leica_data_path):
        self.imu_data = self.load_yaml(imu_path)
        self.camera_data = self.load_yaml(camera_path)
        self.leica_data = self.load_yaml(leica_path)

        self.imu_csv_data = self.load_csv(imu_data_path)
        self.camera_csv_data = self.load_csv(camera_data_path)
        self.leica_csv_data = self.load_csv(leica_data_path)

    def load_yaml(self, file_path):
        """Завантаження YAML-файлу."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def load_csv(self, file_path):
        """Завантаження CSV-файлу."""
        data = pd.read_csv(file_path, comment='#')
        return data

    def get_imu_data(self):
        """Повертає параметри IMU та повні дані."""
        return self.imu_data

    def get_camera_data(self):
        """Повертає параметри та дані камери."""
        return self.camera_data  # Додаємо дані камери
        
    def get_leica_data(self):
        """Повертає параметри та дані Leica-сенсора."""
        return self.leica_data  

    def get_imu_csv(self):
        """Повертає лише дані IMU з CSV."""
        return self.imu_csv_data

    def get_camera_csv(self):
        """Повертає лише дані камери з CSV."""
        return self.camera_csv_data

    def get_leica_csv(self):
        """Повертає лише дані Leica з CSV."""
        return self.leica_csv_data