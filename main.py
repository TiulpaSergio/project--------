import cv2
import numpy as np
import os
from collections import deque
from slam import SLAM
from draw_drawing import draw_trajectory
from feature_detection import detect_keypoints, track_keypoints, update_keypoints_if_needed
from sensor_loader import SensorLoader

def apply_moving_average(trajectory, window_size=5):
    smoothed_trajectory = []
    window = deque(maxlen=window_size)
    for point in trajectory:
        window.append(point)
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        smoothed_trajectory.append((avg_x, avg_y))
    return smoothed_trajectory

def draw_line_if_in_bounds(img, pt1, pt2, color, thickness):
    if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
        0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
        cv2.line(img, pt1, pt2, color, thickness)

def load_frames_from_folder(folder_path):
    # Завантаження та сортування файлів PNG за іменами
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    return [os.path.join(folder_path, f) for f in frame_files]

def main():
    # Завантаження даних сенсорів
    sensor_loader = SensorLoader(
        'mav0/imu0/sensor.yaml', 
        'mav0/cam0/sensor.yaml', 
        'mav0/leica0/sensor.yaml',
        'mav0/imu0/data.csv', 
        'mav0/cam0/data.csv', 
        'mav0/leica0/data.csv'
    )
    
    imu_data = sensor_loader.get_imu_data()
    camera_data = sensor_loader.get_camera_data()
    leica_data = sensor_loader.get_leica_data()
    imu_csv_data = sensor_loader.get_imu_csv()
    camera_csv_data = sensor_loader.get_camera_csv()
    leica_csv_data = sensor_loader.get_leica_csv()

    print("IMU Data:", imu_data)
    print("Camera Data:", camera_data)
    print("Leica Data:", leica_data)
    print("IMU_csv Data:", imu_csv_data)
    print("Camera_csv Data:", camera_csv_data)
    print("Leica_csv Data:", leica_csv_data)

    # Ініціалізація SLAM з даними сенсорів
    slam = SLAM(imu_data, camera_data, leica_data, imu_csv_data, camera_csv_data, leica_csv_data)
    
    # Завантаження PNG фреймів
    frames = load_frames_from_folder('mav0/cam0/data')

    if not frames:
        print("Не знайдено жодного фрейму в каталозі.")
        return

    # Ініціалізація параметрів
    lk_params = {'winSize': (21, 21), 'maxLevel': 3, 
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)}
    detector = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE, edgeThreshold=15)

    # Завантаження першого кадру
    old_frame = cv2.imread(frames[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detect_keypoints(detector, old_gray)

    h, w, _ = old_frame.shape
    trajectory_map = np.zeros((h, w * 2, 3), dtype=np.uint8)
    text_map = np.zeros((h, w * 2, 3), dtype=np.uint8)

    frame_counter = 0

    for frame_path in frames[1:]:
        frame = cv2.imread(frame_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_new, good_old = track_keypoints(old_gray, frame_gray, p0, lk_params)

        if good_new is not None:
            for new, old in zip(good_new, good_old):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())
                cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        p0 = update_keypoints_if_needed(p0, frame_gray, detector)

        # Обробка кадру SLAM
        slam.process_frame(frame)
        trajectory_slam = slam.get_trajectory()
        smoothed_trajectory_slam = apply_moving_average(trajectory_slam)
        global_trajectory = slam.get_global_trajectory()
        smoothed_global_trajectory = apply_moving_average(global_trajectory)

        draw_trajectory(smoothed_global_trajectory, smoothed_trajectory_slam, trajectory_map, scale=1, max_length=100)
        draw_line_if_in_bounds(trajectory_map, (w, 0), (w, h), (255, 255, 255), 2)

        text_map.fill(0)
        if smoothed_trajectory_slam:
            camera_position = slam.get_camera_position().flatten()
            global_position = slam.get_global_position().flatten()

            cv2.putText(text_map, f"Камера: ({camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f})",
                        (w + 10, h - 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(text_map, f"Глобальні: ({global_position[0]:.1f}, {global_position[1]:.1f}, {global_position[2]:.1f})",
                        (10, h - 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        print(f"Кадр №{frame_counter}:")

        cv2.putText(text_map, f"Кадр {frame_counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Кадр {frame_counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        combined_map = cv2.add(trajectory_map, text_map)

        cv2.imshow('Frames', frame)
        cv2.imshow('Trajectory Map', combined_map)

        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
