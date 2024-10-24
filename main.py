import cv2
import numpy as np
import os
from collections import deque
from vo import VO
from feature_detection import detect_keypoints, track_keypoints, update_keypoints_if_needed
from sensor_loader import SensorLoader
import matplotlib.pyplot as plt

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

def load_ground_truth(file_path):
    # Завантаження Ground Truth даних
    ground_truth = []
    with open(file_path, 'r') as f:
        next(f)  # Пропустити перший рядок (коментарі)
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 4:  # Перевірка на кількість стовпців
                x = float(data[1])
                y = float(data[2])
                z = float(data[3])
                ground_truth.append((x, y, z))
    return ground_truth
    
def plot_trajectories(ax2d_vo, ax3d_vo, vo_trajectory, ax2d_gt, ax3d_gt, gt_trajectory, frame_counter):
    # Очищення попередніх даних
    ax2d_vo.clear()
    ax2d_gt.clear()
    ax3d_vo.cla()
    ax3d_gt.cla()

    # Налаштування заголовків
    ax2d_vo.set_title(f'2D Trajectory VO (Кадр {frame_counter})')
    ax2d_gt.set_title('2D Ground Truth')
    ax3d_vo.set_title(f'3D Trajectory VO (Кадр {frame_counter})')
    ax3d_gt.set_title('3D Ground Truth')

    # Побудова 2D траєкторій
    ax2d_vo.plot([p[0] for p in vo_trajectory], [p[1] for p in vo_trajectory], 'b-', label='VO')
    ax2d_gt.plot([p[0] for p in gt_trajectory], [p[1] for p in gt_trajectory], 'g-', label='GT')

    # Додавання початкових та кінцевих точок на 2D графіках
    if vo_trajectory:
        ax2d_vo.plot(vo_trajectory[0][0], vo_trajectory[0][1], 'ro', label='Start')
        ax2d_vo.plot(vo_trajectory[-1][0], vo_trajectory[-1][1], 'bo', label='End')
    if gt_trajectory:
        ax2d_gt.plot(gt_trajectory[0][0], gt_trajectory[0][1], 'ro', label='Start')
        ax2d_gt.plot(gt_trajectory[-1][0], gt_trajectory[-1][1], 'bo', label='End')

    # Побудова 3D траєкторій
    ax3d_vo.plot([p[0] for p in vo_trajectory], [p[1] for p in vo_trajectory], [p[2] for p in vo_trajectory], 'b-')
    ax3d_gt.plot([p[0] for p in gt_trajectory], [p[1] for p in gt_trajectory], [p[2] for p in gt_trajectory], 'g-')

    # Додавання початкових та кінцевих точок на 3D графіках
    if vo_trajectory:
        ax3d_vo.scatter(vo_trajectory[0][0], vo_trajectory[0][1], vo_trajectory[0][2], color='r', s=50, label='Start')
        ax3d_vo.scatter(vo_trajectory[-1][0], vo_trajectory[-1][1], vo_trajectory[-1][2], color='b', s=50, label='End')
    if gt_trajectory:
        ax3d_gt.scatter(gt_trajectory[0][0], gt_trajectory[0][1], gt_trajectory[0][2], color='r', s=50, label='Start')
        ax3d_gt.scatter(gt_trajectory[-1][0], gt_trajectory[-1][1], gt_trajectory[-1][2], color='b', s=50, label='End')

    # Оновлення графіків
    ax2d_vo.legend()
    ax2d_gt.legend()
    plt.draw()

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

    # Ініціалізація SLAM з даними сенсорів
    vo = VO(imu_data, camera_data, leica_data, imu_csv_data, camera_csv_data, leica_csv_data)

    frames = load_frames_from_folder('mav0/cam0/data')
    if not frames:
        print("Не знайдено жодного фрейму в каталозі.")
        return

    # Завантаження даних Ground Truth
    ground_truth = load_ground_truth('mav0/state_groundtruth_estimate0/data.csv')
    smoothed_ground_truth = apply_moving_average(ground_truth)

    detector = cv2.ORB_create(nfeatures=3000)
    lk_params = {'winSize': (21, 21), 'maxLevel': 3, 
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)}

    old_frame = cv2.imread(frames[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detect_keypoints(detector, old_gray)

    # Ініціалізація вікна з 4 графіками
    fig = plt.figure(figsize=(12, 8))
    ax2d_vo = fig.add_subplot(221)  # 2D VO
    ax2d_gt = fig.add_subplot(222)  # 2D Ground Truth
    ax3d_vo = fig.add_subplot(223, projection='3d')  # 3D VO
    ax3d_gt = fig.add_subplot(224, projection='3d')  # 3D Ground Truth

    plt.ion()  # Увімкнення інтерактивного режиму

    cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Frames', cv2.WND_PROP_TOPMOST, 1)  # Вікно завжди зверху

    frame_counter = 0
    plot_frequency = 5  # Частота оновлення графіків
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

        # Виклик методу VO для обчислення траєкторії
        global_trajectory = vo.compute_trajectory(frame_gray)
        smoothed_trajectory = apply_moving_average(global_trajectory)

        if frame_counter % plot_frequency == 0:
            plot_trajectories(
                ax2d_vo, ax3d_vo, smoothed_trajectory, 
                ax2d_gt, ax3d_gt, smoothed_ground_truth, frame_counter
            )

        print(f"Кадр № {frame_counter}")
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
