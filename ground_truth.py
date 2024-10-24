import pandas as pd
import matplotlib.pyplot as plt
import time

def plot_ground_truth():
    # Визначаємо назви стовпців
    column_names = [
        'timestamp', 'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z',
        'q_RS_w', 'q_RS_x', 'q_RS_y', 'q_RS_z',
        'v_RS_R_x', 'v_RS_R_y', 'v_RS_R_z',
        'b_w_RS_S_x', 'b_w_RS_S_y', 'b_w_RS_S_z',
        'b_a_RS_S_x', 'b_a_RS_S_y', 'b_a_RS_S_z'
    ]

    # Завантажуємо дані з CSV файлу з заданими назвами стовпців і типами
    data = pd.read_csv('mav0/state_groundtruth_estimate0/data.csv', header=0, names=column_names, dtype={
        'timestamp': 'int64',
        'p_RS_R_x': 'float64',
        'p_RS_R_y': 'float64',
        'p_RS_R_z': 'float64',
        'q_RS_w': 'float64',
        'q_RS_x': 'float64',
        'q_RS_y': 'float64',
        'q_RS_z': 'float64',
        'v_RS_R_x': 'float64',
        'v_RS_R_y': 'float64',
        'v_RS_R_z': 'float64',
        'b_w_RS_S_x': 'float64',
        'b_w_RS_S_y': 'float64',
        'b_w_RS_S_z': 'float64',
        'b_a_RS_S_x': 'float64',
        'b_a_RS_S_y': 'float64',
        'b_a_RS_S_z': 'float64'
    })

    # Витягуємо координати
    x = data['p_RS_R_x']
    y = data['p_RS_R_y']
    z = data['p_RS_R_z']

    # Створюємо фігуру та підплоти
    fig = plt.figure(figsize=(12, 6))

    # 2D графік
    ax1 = fig.add_subplot(121)
    ax1.set_title('2D Trajectory')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.axis('equal')
    ax1.grid()

    # 3D графік
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('3D Trajectory')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')

    # Додаємо початкову точку
    start_point = ax1.scatter(x.iloc[0], y.iloc[0], color='g', s=100, label='Start Point')
    end_point = ax1.scatter(x.iloc[0], y.iloc[0], color='r', s=100, label='End Point')  # Тимчасова кінцева точка

    ax1.legend()  # Визначаємо легенду один раз

    # Додаємо початкову точку для 3D графіка
    ax2.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='g', s=100)  # Початкова точка
    end_point_3d = ax2.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='r', s=100)  # Тимчасова кінцева точка для 3D

    # Поступово оновлюємо графіки
    for i in range(len(x)):
        # Оновлюємо 2D графік
        ax1.plot(x[:i+1], y[:i+1], color='b')  # Додаємо нову точку
        
        # Оновлюємо кінцеву точку
        end_point.set_offsets((x.iloc[i], y.iloc[i]))  # Оновлюємо положення кінцевої точки
        
        # Оновлюємо 3D графік
        ax2.plot(x[:i+1], y[:i+1], z[:i+1], color='r')  # Додаємо нову точку
        
        # Оновлюємо кінцеву точку для 3D
        end_point_3d._offsets3d = (x.iloc[i:i+1], y.iloc[i:i+1], z.iloc[i:i+1])  # Оновлюємо положення кінцевої точки 3D

        # Показуємо графіки
        plt.pause(0.00005)  # Затримка 50 мс

    plt.tight_layout()
    plt.show()

plot_ground_truth()
