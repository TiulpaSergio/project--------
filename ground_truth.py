import pandas as pd
import matplotlib.pyplot as plt

def plot_ground_truth():
    # Завантажте дані з CSV файлу
    leica_data = pd.read_csv('mav0/leica0/data.csv', header=0)  # Вказуємо, що перший рядок - це заголовки

    # Вибираємо колонки
    x_coords = leica_data['p_RS_R_x [m]']
    y_coords = leica_data['p_RS_R_y [m]']
    z_coords = leica_data['p_RS_R_z [m]']

    # Візуалізація
    fig = plt.figure(figsize=(10, 8))

    # 2D графік
    plt.subplot(211)  # Перший графік
    plt.plot(x_coords, y_coords, label='Ground Truth', color='blue')
    plt.scatter(x_coords.iloc[0], y_coords.iloc[0], color='green', label='Start', zorder=5)
    plt.scatter(x_coords.iloc[-1], y_coords.iloc[-1], color='red', label='End', zorder=5)
    plt.text(x_coords.iloc[0], y_coords.iloc[0], 'Start', fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='green')
    plt.text(x_coords.iloc[-1], y_coords.iloc[-1], 'End', fontsize=12, verticalalignment='bottom', horizontalalignment='left', color='red')
    plt.title('Ground Truth 2D')
    plt.xlabel('X Coordinate [m]')
    plt.ylabel('Y Coordinate [m]')
    plt.legend()
    plt.grid()

    # 3D графік
    ax = fig.add_subplot(212, projection='3d')  # Другий графік
    ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o')
    ax.scatter(x_coords.iloc[0], y_coords.iloc[0], z_coords.iloc[0], color='green', s=100, label='Start')
    ax.scatter(x_coords.iloc[-1], y_coords.iloc[-1], z_coords.iloc[-1], color='red', s=100, label='End')
    ax.text(x_coords.iloc[0], y_coords.iloc[0], z_coords.iloc[0], 'Start', fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='green')
    ax.text(x_coords.iloc[-1], y_coords.iloc[-1], z_coords.iloc[-1], 'End', fontsize=12, verticalalignment='bottom', horizontalalignment='left', color='red')
    ax.set_title('Ground Truth 3D')
    ax.set_xlabel('X Coordinate [m]')
    ax.set_ylabel('Y Coordinate [m]')
    ax.set_zlabel('Z Coordinate [m]')
    ax.legend()
