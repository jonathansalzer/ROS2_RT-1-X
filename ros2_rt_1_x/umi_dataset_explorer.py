import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # data = np.load('/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/episodes/epi_1720712200.npy', allow_pickle=True)
    # for i in range(0, len(data)):
    #     print(f"Step {i}")
    #     print(data[i]['state'])
    #     print(data[i]['action'])

    print("Hello from umi_dataset_explorer")

    directory = os.fsencode("/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/episodes")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        print(f"Analyzing episode {filename}...")

        data = np.load(f"/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/episodes/{filename}", allow_pickle=True)

        highest_x = 0
        lowest_x = 0
        highest_y = 0
        highest_z = 0

        for index, step in enumerate(data):
            world_vec = step['state']
            x = world_vec[0]
            y = world_vec[1]
            z = world_vec[2]

            if x > highest_x:
                highest_x = x
            if x < lowest_x:
                lowest_x = x
            if y > highest_y:
                highest_y = y
            if z > highest_z:
                highest_z = z

        print(f"Highest x: {highest_x}")
        print(f"Lowest x: {lowest_x}")
        if highest_x > 0.4 or lowest_x < -0.4:
            print(f" ================> Episode {filename} has x values out of range.")
        # print(f"Highest y: {highest_y}")
        # print(f"Highest z: {highest_z}")

def calculate_final_state(prev_state, action):
    return [
        prev_state[0] + action[0],
        prev_state[1] + action[1],
        prev_state[2] + action[2],
        prev_state[3] + action[3],
        prev_state[4] + action[4],
        prev_state[5] + action[5],
        prev_state[6] + action[6],
        True
    ]

ACTION_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False]
STATE_0 = [0.0, 0.5, 0.5, 90.0, 0.0, 0.0, 0.02, False]