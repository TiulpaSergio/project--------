import cv2
import numpy as np
import os
from collections import deque
from vo import VO
from feature_detection import detect_keypoints, track_keypoints, update_keypoints_if_needed
from sensor_loader import SensorLoader
import matplotlib.pyplot as plt
from ground_truth import plot_ground_truth

def apply_moving_average(trajectory, window_size=5):
    smoothed_trajectory = []
    window = deque(maxlen=window_size)
    for point in trajectory:
        window.append(point)
        avg_x = np.mean([p[0] for p in window])
        avg_y = np.mean([p[1] for p in window])
        avg_z = np.mean([p[2] for p in window])
        smoothed_trajectory.append((avg_x, avg_y, avg_z))
    return smoothed_trajectory

def load_frames_from_folder(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    return [os.path.join(folder_path, f) for f in frame_files]

def plot_trajectories(ax2d, ax3d, trajectory, frame_counter):
    ax3d.cla()  # Очищення осей 3D графіка
    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Траєкторія', color='b')
        ax3d.set_xlabel('X координати')
        ax3d.set_ylabel('Y координати')
        ax3d.set_zlabel('Z координати')
        ax3d.set_title('3D Траєкторія')
        ax3d.legend()

    # Малювання 2D траєкторії
    ax2d.cla()  # Очищення осей 2D графіка
    ax2d.set_title('2D Траєкторія')
    ax2d.set_xlabel('X координати')
    ax2d.set_ylabel('Y координати')

    if len(trajectory) > 0:
        trajectory_2d = trajectory[:, :2]
        ax2d.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], label='Траєкторія', color='g')
        ax2d.legend()

        # Додавання номера кадру до графіка
        ax2d.text(0.05, 0.95, f"Кадр: {frame_counter}", transform=ax2d.transAxes, 
                  fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.5))

def main():
    plot_ground_truth()
    
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

    print(imu_data)
    print(camera_data)
    print(leica_data)
    print(imu_csv_data)
    print(camera_csv_data)
    print(leica_csv_data)

    # Ініціалізація SLAM з даними сенсорів(поки не інтегровано)
    vo = VO(imu_data, camera_data, leica_data, imu_csv_data, camera_csv_data, leica_csv_data)

    frames = load_frames_from_folder('mav0/cam0/data')
    if not frames:
        print("Не знайдено жодного фрейму в каталозі.")
        return

    detector = cv2.ORB_create(nfeatures=3000)
    lk_params = {'winSize': (21, 21), 'maxLevel': 3, 
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)}

    old_frame = cv2.imread(frames[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detect_keypoints(detector, old_gray)

    fig = plt.figure(figsize=(10, 8))
    ax2d = fig.add_subplot(211)
    ax3d = fig.add_subplot(212, projection='3d')
    plt.ion()

    cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Frames', cv2.WND_PROP_TOPMOST, 1)  # Завжди на передньому плані

    frame_counter = 0
    plot_frequency = 5
    global_trajectory = []

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
        global_trajectory
        smoothed_trajectory = apply_moving_average(global_trajectory)

        if frame_counter % plot_frequency == 0:
            plot_trajectories(ax2d, ax3d, smoothed_trajectory, frame_counter)

        print(f"Кадр №",frame_counter)
        cv2.putText(frame, f"Кадр: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Frames', frame)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            cv2.destroyAllWindows()
            plt.close('all')
            return

        plt.pause(0.001)
        frame_counter += 1

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
