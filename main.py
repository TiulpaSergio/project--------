import cv2
import numpy as np
from collections import deque
from slam import SLAM
from draw_drawing import draw_trajectory

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
    # Перевіряємо, чи обидві точки в межах зображення
    if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
        0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
        cv2.line(img, pt1, pt2, color, thickness)

def main():
    slam = SLAM()

    cap = cv2.VideoCapture('output/output_video2.avi')
    if not cap.isOpened():
        print("Не вдалося відкрити відеофайл")
        return

    lk_params = {
        'winSize': (21, 21),
        'maxLevel': 3,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
    }

    detector = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE, edgeThreshold=15)

    ret, old_frame = cap.read()
    if not ret:
        print("Не вдалося прочитати перший кадр відео")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(old_gray, None)
    p0 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    h, w, _ = old_frame.shape
    trajectory_slam = []
    trajectory_map = np.zeros((h, w * 2, 3), dtype=np.uint8)  # Створення за межами циклу
    text_map = np.zeros((h, w * 2, 3), dtype=np.uint8)  # Мапа для тексту

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) >= 4:
                print(f"Кадр №{frame_counter}:")

                for new, old in zip(good_new, good_old):
                    a, b = map(int, new.ravel())
                    c, d = map(int, old.ravel())
                    cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

            # Оновлення старого зображення та точок
            old_gray = frame_gray.copy()
            if len(good_new) > 0:
                p0 = good_new.reshape(-1, 1, 2)

        # Перевірка кількості точок
        if len(p0) < 300:
            keypoints = detector.detect(frame_gray, None)
            new_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

            if len(new_points) > 0:
                p0 = np.vstack((p0, new_points))
                old_gray = frame_gray.copy()

        slam.process_frame(frame)
        trajectory_slam = slam.get_trajectory()
        smoothed_trajectory_slam = apply_moving_average(trajectory_slam)
        global_trajectory = slam.get_global_trajectory()
        smoothed_global_trajectory = apply_moving_average(global_trajectory)

        draw_trajectory(smoothed_global_trajectory, smoothed_trajectory_slam, trajectory_map, scale=1, max_length=100)

        draw_line_if_in_bounds(trajectory_map, (w, 0), (w, h), (255, 255, 255), 2)

        # Очищаємо текстову мапу
        text_map.fill(0)

        if smoothed_trajectory_slam:
            # Виводимо координати камери та глобальні координати
            camera_position = slam.get_camera_position()  # Отримуємо позицію камери
            global_position = slam.get_global_position()  # Отримуємо глобальні координати

            # Виводимо позицію камери з правого боку
            cv2.putText(text_map, f"Позицiя камери: ({camera_position[0][0]:.1f}, {camera_position[1][0]:.1f}, {camera_position[2][0]:.1f})", (w + 10, h - 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Виводимо глобальні координати з лівого боку
            cv2.putText(text_map, f"Глобальнi координати: ({global_position[0][0]:.1f}, {global_position[1][0]:.1f}, {global_position[2][0]:.1f})", (10, h - 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # Додаємо номер кадру
        cv2.putText(frame, f"Кадр {frame_counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Об'єднуємо мапу координат з основною мапою
        combined_map = cv2.add(trajectory_map, text_map)

        # Додаємо номер кадру на мапу
        cv2.putText(combined_map, f"Кадр {frame_counter}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Візуальна одометрія", frame)
        cv2.imshow('Trajectory Map', combined_map)

        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
