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

        print(f"Correcting episode {filename}...")

        data = np.load(f"/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/episodes/{filename}", allow_pickle=True)

        corrected_data = []

        for index, step in enumerate(data):
            if index == len(data) - 1:
                corrected_data.append({
                    'image': step['image'],
                    'action': step['action'],
                    'language_instruction': step['language_instruction'],
                    'state': calculate_final_state(data[index - 1]['state'], step['action'])
                })
            else:
                corrected_data.append({
                    'image': step['image'],
                    'action': step['action'],
                    'language_instruction': step['language_instruction'],
                    'state': data[index + 1]['state']
                })

        # corrected_data[0]['state'] = STATE_0
        # corrected_data[0]['action'] = ACTION_0

        np.save(f"/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/corrected_episodes/{filename}", corrected_data)

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