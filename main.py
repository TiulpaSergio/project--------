from collections import deque
import cv2
import numpy as np
import torch

class SLAM:
    def __init__(self):
        # Ініціалізація ORB детектора
        self.orb = cv2.ORB_create()
        
        # Список для збереження точок мапи
        self.map_points = []
        
        # Початкова позиція камери
        self.camera_position = np.zeros((3, 1))
        
        # Глобальні координати
        self.global_position = np.zeros((3, 1))
        
        # Попередній кадр та точки
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Ініціалізація списку для збереження траєкторії
        self.trajectory = []

    def process_frame(self, frame):
        # Перетворення зображення у відтінки сірого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Якщо це перший кадр, просто визначаємо ключові точки
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = self.detect_features(gray)
            return

        # Визначення ключових точок та дескрипторів для поточного кадру
        keypoints, descriptors = self.detect_features(gray)

        # Пошук відповідностей між поточними і попередніми дескрипторами
        matches = self.match_features(self.prev_descriptors, descriptors)

        # Фільтрація надійних відповідностей
        matched_pts_prev, matched_pts_curr = self.get_matched_points(matches, self.prev_keypoints, keypoints)

        # Локалізація камери на основі відповідностей
        if len(matched_pts_prev) >= 8:
            self.localize_camera(matched_pts_prev, matched_pts_curr)
            self.update_trajectory()

        # Оновлення попереднього кадру
        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def detect_features(self, gray):
        # Використання ORB для визначення ключових точок та дескрипторів
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return np.array([kp.pt for kp in keypoints]), descriptors

    def match_features(self, descriptors1, descriptors2):
        # Використання BFMatcher для знаходження відповідностей
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    def get_matched_points(self, matches, keypoints1, keypoints2):
        # Витягування координат відповідних точок
        matched_pts_prev = np.float32([keypoints1[m.queryIdx] for m in matches])
        matched_pts_curr = np.float32([keypoints2[m.trainIdx] for m in matches])
        return matched_pts_prev, matched_pts_curr

    def localize_camera(self, pts_prev, pts_curr):
        # Знайти фундаментальну матрицю
        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.RANSAC, 0.1, 0.99)
        if F is None:
            print("Не вдалося знайти фундаментальну матрицю.")
            return
        
        # Очищення точок на основі маски
        pts_prev = pts_prev[mask.ravel() == 1]
        pts_curr = pts_curr[mask.ravel() == 1]
        
        # Оцінка відносного руху камери
        E, _ = cv2.findEssentialMat(pts_curr, pts_prev, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("Не вдалося знайти основну матрицю.")
            return
        
        # Визначення R та t
        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev)
        
        # Оновлення позиції камери
        self.camera_position += t
        self.trajectory.append(self.camera_position.copy())
        
        # Оновлення глобальних координат
        self.update_global_position(R, t)

        # Вивід для трасування
        print(f"Оновлена позиція камери: {self.camera_position.T}")
        print(f"Глобальні координати: {self.global_position.T}")

    def update_global_position(self, R, t):
        # Оновлення глобальних координат
        self.global_position += R @ t

        # Вивід для трасування
        print(f"Оновлені глобальні координати: {self.global_position.T}")

    def update_trajectory(self):
        # Вивід поточної позиції камери
        print(f"Поточна позиція камери: {self.camera_position.T}")

    def get_trajectory(self):
        return np.array(self.trajectory)
    
# Функція для малювання траєкторії
def draw_trajectory(trajectory1, trajectory2, trajectory_map, scale=1, max_length=100):
    # Обробка першої траєкторії
    if len(trajectory1) > max_length:
        trajectory1 = trajectory1[-max_length:]

    if len(trajectory1) > 1:
        for i in range(1, len(trajectory1)):
            start_point = (int(trajectory1[i - 1][0]), int(trajectory1[i - 1][1])) 
            end_point = (int(trajectory1[i][0]), int(trajectory1[i][1]))  

            # Перевірка меж точок
            if (0 <= start_point[0] < trajectory_map.shape[1] and 
                0 <= start_point[1] < trajectory_map.shape[0] and
                0 <= end_point[0] < trajectory_map.shape[1] and 
                0 <= end_point[1] < trajectory_map.shape[0]):
                
                cv2.line(trajectory_map, start_point, end_point, (0, 255, 0), 2) 
                cv2.circle(trajectory_map, end_point, 3, (0, 0, 255), -1) 

    # Обробка другої траєкторії
    if len(trajectory2) > max_length:
        trajectory2 = trajectory2[-max_length:]

    if len(trajectory2) > 1:
        for i in range(1, len(trajectory2)):
            start_point = (int(trajectory2[i - 1][0]), int(trajectory2[i - 1][1])) 
            end_point = (int(trajectory2[i][0]), int(trajectory2[i][1])) 

            # Перевірка меж точок
            if (0 <= start_point[0] < trajectory_map.shape[1] and 
                0 <= start_point[1] < trajectory_map.shape[0] and
                0 <= end_point[0] < trajectory_map.shape[1] and 
                0 <= end_point[1] < trajectory_map.shape[0]):
                
                cv2.line(trajectory_map, start_point, end_point, (255, 0, 0), 2)
                cv2.circle(trajectory_map, end_point, 3, (255, 255, 0), -1) 

    # Виведення поточних позицій обох траєкторій
    if trajectory1:
        current_position1 = (int(trajectory1[-1][0]), int(trajectory1[-1][1]))
        cv2.circle(trajectory_map, current_position1, 5, (0, 255, 255), -1) 

    if trajectory2:
        current_position2 = (int(trajectory2[-1][0]), int(trajectory2[-1][1])) 
        cv2.circle(trajectory_map, current_position2, 5, (0, 128, 255), -1)  

    return trajectory_map

# Функція для обчислення ковзаючого середнього
from collections import deque

def apply_moving_average(trajectory1, trajectory2, window_size=5):
    smoothed_trajectory1 = []
    smoothed_trajectory2 = []
    window1 = deque(maxlen=window_size)
    window2 = deque(maxlen=window_size)

    # Згладжування першої траєкторії
    for point in trajectory1:
        window1.append(point)
        avg_x = np.mean([p[0] for p in window1])
        avg_y = np.mean([p[1] for p in window1])
        smoothed_trajectory1.append((avg_x, avg_y))

    # Згладжування другої траєкторії
    for point in trajectory2:
        window2.append(point)
        avg_x = np.mean([p[0] for p in window2])
        avg_y = np.mean([p[1] for p in window2])
        smoothed_trajectory2.append((avg_x, avg_y))

    return smoothed_trajectory1, smoothed_trajectory2

def main():
    # Ініціалізація об'єкта SLAM
    slam = SLAM()

    cap = cv2.VideoCapture('output/output_video3.avi')
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
    initial_position = [w // 2, h // 2]
    global_position = initial_position.copy()
    trajectory = []
    trajectory_slam = [] 
    trajectory_map = np.zeros((h, w, 3), dtype=np.uint8)

    # Відзначаємо початкову точку
    cv2.circle(trajectory_map, (initial_position[0], initial_position[1]), 5, (255, 0, 0), -1)

    frame_counter = 0
    last_H = None 

    # Встановіть пороги для стабілізації по осі Y
    y_threshold = 5 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_old) >= 4 and len(good_new) >= 4:
                distances = np.linalg.norm(good_new - good_old, axis=1)
                median_distance = np.median(distances)
                valid_indices = np.where(distances < 2 * median_distance)[0]

                if len(valid_indices) > 4:
                    good_old = good_old[valid_indices]
                    good_new = good_new[valid_indices]

                H, status = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                
                if H is not None:
                    if last_H is not None:
                        # Порівнюємо з останньою матрицею H для обмеження
                        dx, dy = H[0, 2] - last_H[0, 2], H[1, 2] - last_H[1, 2]
                        if abs(dx) > 5 or abs(dy) > 5:  
                            H = last_H 

                    last_H = H 
                    dx, dy = H[0, 2], H[1, 2]
                    
                    # Стабілізація по осі Y
                    if abs(dy) < y_threshold:
                        global_position[0] -= dx * 0.05 
                        global_position[1] += dy * 0.05

                    trajectory.append((global_position[0], global_position[1]))

                    print(f"Кадр №{frame_counter}: Глобальні координати: {global_position}, "
                          f"Трансформаційна матриця H:\n{H}")
                else:
                    print(f"Кадр №{frame_counter}: Не вдалося знайти гомографію для поточного кадру")
            else:
                print(f"Кадр №{frame_counter}: Недостатньо точок для обчислення гомографії")

            for new, old in zip(good_new, good_old):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())
                cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        if len(p0) < 300:
            keypoints = detector.detect(frame_gray, None)
            new_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            
            if len(new_points) > 0:
                p0 = np.vstack((p0, new_points))
                old_gray = frame_gray.copy()

        # Обробка кадру з використанням SLAM
        slam.process_frame(frame)
        trajectory_slam = slam.get_trajectory()  # Отримання SLAM траєкторії

        # Згладжування траєкторій
        smoothed_trajectory, smoothed_trajectory_slam = apply_moving_average(trajectory, trajectory_slam)

        # Малюємо траєкторії на мапі
        trajectory_map = draw_trajectory(smoothed_trajectory, smoothed_trajectory_slam, trajectory_map, scale=1, max_length=100)

        # Відзначаємо початкову точку на мапі
        cv2.circle(trajectory_map, (initial_position[0], initial_position[1]), 5, (255, 0, 0), -1)

        # Поєднуємо кадр і мапу траєкторії для відображення
        combined_frame = np.hstack((frame, trajectory_map))
        
        # Малюємо ключові точки з SLAM
        for kp in slam.prev_keypoints:  
            pt = (int(kp[0]), int(kp[1]))  
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)

        # Додаємо траєкторію SLAM
        if len(trajectory_slam) > 0:
            for i in range(1, len(trajectory_slam)):
                pt1 = (int(trajectory_slam[i - 1][0]), int(trajectory_slam[i - 1][1]))
                pt2 = (int(trajectory_slam[i][0]), int(trajectory_slam[i][1]))
                cv2.line(combined_frame, pt1, pt2, (0, 255, 255), 2)

        cv2.imshow("Візуальна одометрія", combined_frame)

        # Зберігаємо мапу кожні 1000 кадрів
        if frame_counter % 1000 == 0:
            cv2.imwrite(f'output/trajectory_map_{frame_counter}.png', trajectory_map)

        frame_counter += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
