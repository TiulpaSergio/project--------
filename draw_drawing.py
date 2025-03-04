#це старй код новий потрібно перенести з main
import cv2
import numpy as np

def draw_trajectory(smoothed_trajectory1, trajectory_map, scale=1, max_length=100):
    height, width, _ = trajectory_map.shape

    if len(smoothed_trajectory1) > max_length:
        smoothed_trajectory1 = smoothed_trajectory1[-max_length:]
    if len(smoothed_trajectory2) > max_length:
        smoothed_trajectory2 = smoothed_trajectory2[-max_length:]


    offset_y = height // 2 
    left_offset = width // 4

    for i in range(1, len(smoothed_trajectory1)):
        pt1 = (
            int(smoothed_trajectory1[i - 1][0] * scale) + left_offset, 
            int(smoothed_trajectory1[i - 1][1] * scale) + offset_y
        )
        pt2 = (
            int(smoothed_trajectory1[i][0] * scale) + left_offset, 
            int(smoothed_trajectory1[i][1] * scale) + offset_y
        )

        pt1 = (max(0, min(pt1[0], width - 1)), max(0, min(pt1[1], height - 1)))
        pt2 = (max(0, min(pt2[0], width - 1)), max(0, min(pt2[1], height - 1)))

        cv2.line(trajectory_map, pt1, pt2, (0, 255, 0), 2)

    return trajectory_map
