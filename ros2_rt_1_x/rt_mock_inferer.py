import rclpy
from rclpy.node import Node
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import tensorflow as tf
import copy
import rlds

import ros2_rt_1_x.models.rt1_inference as jax_models
import ros2_rt_1_x.output_logging as output_log
import ros2_rt_1_x.umi_rescale as umi_rescale


class RtMockInferer(Node):
    def __init__(self):
        tf.config.experimental.set_visible_devices([], "GPU")

        super().__init__('rt_mock_inferer')

        # "bridge" or "umi"
        self.dataset = "umi"

        if self.dataset == "umi":
            self.natural_language_instruction = "Pick up the yellow banana."
        else:
            self.natural_language_instruction = "Place the can to the left of the pot."

        self.epi_steps_iterator = self.load_dataset()
        
        self.rt1_inferer = jax_models.RT1Inferer(self.natural_language_instruction)

        self.run_inference()

    def load_dataset(self):
        if self.dataset == "umi":
            dataset_builder = tfds.builder_from_directory('/home/jonathan/tensorflow_datasets/episodes/1.0.0')
        else:
            dataset_builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
        dataset = dataset_builder.as_dataset(split='train')
        iter_dataset = iter(dataset)

        episode_count = 0
        for episode in iter_dataset:
            episode_count += 1
        print(f'Episode count: {episode_count}')

        iter_dataset = iter(dataset)

        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)
        first_episode = next(iter_dataset)


        episode_steps = first_episode['steps']
        step_iterator = iter(episode_steps)
        return step_iterator

    def run_inference(self):
        actions = []
        ground_truth_actions = []

        for index, step in enumerate(self.epi_steps_iterator):

            image = Image.fromarray(np.array(step['observation']['image']))

            act = self.rt1_inferer.run_umi_mock_inference(image, self.natural_language_instruction, index)

            print("OG ACTION: ", act)

            # actions.append(act)

            if self.dataset == "umi":
                act = umi_rescale.rt1_outputs_to_umi_states(act)
            if self.dataset == "bridge":
                act = umi_rescale.scale_rt1_to_bridge(act)

            actions.append(act)
            print(act)

            print("DIFFERENCE:")
            print(act['world_vector'][1])
            print(step['action']['world_vector'][1])
            print(act['world_vector'][1] - step['action']['world_vector'][1])

            # transform ground truth action to match the output of the model, so it can be plotted

            if self.dataset == "umi":
                step['action']['gripper_closedness_action'] = [step['action']['gripper_closedness_action']]
                # step['action']['world_vector'] = [i * 2 - 0.45 for i in step['action']['world_vector']]

            if self.dataset == "bridge":
                if step['action']['open_gripper'] == True:
                    step['action']['gripper_closedness_action'] = [-1.0]
                else:
                    step['action']['gripper_closedness_action'] = [1.0]

            if step['action']['terminate_episode'] == 0.0:
                step['action']['terminate_episode'] = [0,1,0]
            else:
                step['action']['terminate_episode'] = [1,0,0]

            ground_truth_actions.append(step['action'])

            print(f'Inference step {index+1}')

        output_log.draw_compare_model_output(actions, ground_truth_actions, 'umi_mock_inference')



def main(args=None):
    rclpy.init(args=args)

    rt_mock_inferer = RtMockInferer()

    rclpy.spin(rt_mock_inferer)
