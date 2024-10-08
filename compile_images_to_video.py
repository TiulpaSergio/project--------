import os
import cv2

def load_image_paths(image_folder):
    """Завантажити шляхи до зображень з папки."""
    files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    files.sort(key=lambda x: int(x[:-4]))  # Сортування за номером зображення
    return [os.path.join(image_folder, f) for f in files]

def create_video_from_images(image_folder, output_file):
    """Створити відео з зображень у заданій папці."""
    images = load_image_paths(image_folder)
    
    if not images:
        print("Не знайдено зображень у папці.")
        return

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(image))
    
    video.release()
    print(f"Відео збережено в {output_file}")

def main():
    # Вкажіть шлях до папки з зображеннями та назву вихідного відеофайлу
    image_folder = 'mav1/cam0/data/'  # Задайте ваш шлях до зображень
    output_file = 'output/output_video3.avi'  # Назва вихідного відеофайлу
    
    create_video_from_images(image_folder, output_file)

if __name__ == "__main__":
    main()
