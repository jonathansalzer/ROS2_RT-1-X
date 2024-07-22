import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
import time
import numpy as np
import os

class UmiDatasetRunner(Node):

    def __init__(self):
        super().__init__('umi_dataset_runner')
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        self.cur_x = 0.0
        self.cur_y = 0.5
        self.cur_z = 0.5
        self.cur_roll = 0.0
        self.cur_pitch = 0.0
        self.cur_yaw = 90.0
        self.cur_grip = 0.02

        self.init_pose()
        time.sleep(10)
        print('Pose initialized.')

        self.counter = 0

        # self.dataset = np.load('/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/corrected_episodes/epi_1720181013_0_1.npy', allow_pickle=True)

        directory = os.fsencode("/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/corrected_episodes")

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if "_0_1" in filename:
                continue

            print(f"Analyzing episode {filename}...")

            self.init_pose()
            time.sleep(2)
            print('Pose initialized.')

            self.dataset = np.load(f"/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/corrected_episodes/{filename}", allow_pickle=True)

            for index, step in enumerate(self.dataset):
                pose = Pose()
                pose.position.x = step['state'][0]
                pose.position.y = step['state'][1]
                pose.position.z = step['state'][2]
                pose.orientation.x = step['state'][3]
                pose.orientation.y = step['state'][4]
                pose.orientation.z = step['state'][5]
                pose.orientation.w = 1.0

                grip = Float32()
                grip.data = step['state'][6]

                self.pose_publisher.publish(pose)
                self.grip_publisher.publish(grip)

                if step['state'][7]:
                    break

                time.sleep(0.5)
        # self.timer = self.create_timer(5, self.execute_action)

    def execute_state(self):
        step = self.dataset[self.counter]
        state = step['state']

        pose = Pose()
        pose.position.x = state[0]
        pose.position.y = state[1]
        pose.position.z = state[2]
        pose.orientation.x = state[3]
        pose.orientation.y = state[4]
        pose.orientation.z = state[5]
        pose.orientation.w = 1.0

        grip = Float32()
        grip.data = state[6]

        self.pose_publisher.publish(pose)
        self.grip_publisher.publish(grip)

        self.get_logger().info(str(pose))

        self.counter += 1

        if state[7]:
            self.get_logger().info('Episode finished.')
            self.timer.cancel()

    def execute_action(self):
        step = self.dataset[self.counter]
        action = step['action']

        print(action)
        
        world_vector = action[:3]
        rotation_delta = action[3:6]
        gripper_closedness_action = action[6]
        terminate_episode = action[7]

        self.cur_x += float(world_vector[0])
        self.cur_y += float(world_vector[1])
        self.cur_z += float(world_vector[2])
        self.cur_roll += float(rotation_delta[0])
        self.cur_pitch += float(rotation_delta[1])
        self.cur_yaw += float(rotation_delta[2])
        self.cur_grip = float(gripper_closedness_action)

        self.cur_x = min(max(self.cur_x, -0.5), 0.5)
        self.cur_y = min(max(self.cur_y, 0.2), 0.7)
        self.cur_z = min(max(self.cur_z, 0.2), 0.6)
        self.cur_roll = min(max(self.cur_roll, 0.0), 90.0)
        self.cur_pitch = min(max(self.cur_pitch, 0.0), 90.0)
        self.cur_yaw = min(max(self.cur_yaw, -10.0), 170.0)
        self.cur_grip = min(max(self.cur_grip, 0.02), 0.08)

        pose = Pose()
        pose.position.x = self.cur_x
        pose.position.y = self.cur_y
        pose.position.z = self.cur_z
        pose.orientation.x = self.cur_yaw
        pose.orientation.y = self.cur_pitch
        pose.orientation.z = self.cur_roll
        pose.orientation.w = 1.0

        grip = Float32()
        grip.data = self.cur_grip

        self.pose_publisher.publish(pose)
        self.grip_publisher.publish(grip)

        self.get_logger().info(str(pose))

        self.counter += 1

        if terminate_episode:
            self.get_logger().info('Episode finished.')

    def init_pose(self):
        pose = Pose()
        pose.position.x = self.cur_x
        pose.position.y = self.cur_y
        pose.position.z = self.cur_z
        pose.orientation.x = self.cur_yaw
        pose.orientation.y = self.cur_pitch
        pose.orientation.z = self.cur_roll
        pose.orientation.w = 1.0

        grip = Float32()
        grip.data = self.cur_grip

        self.pose_publisher.publish(pose)
        self.grip_publisher.publish(grip)
        
def main(args=None):
    rclpy.init(args=args)

    umi_dataset_runner = UmiDatasetRunner()

    rclpy.spin(umi_dataset_runner)

    umi_dataset_runner.destroy_node()
    rclpy.shutdown()