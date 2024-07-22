import numpy as np
import copy

# UMI COORDINATE LIMITS
X_MIN = -0.5
X_MAX = 0.5
Y_MIN = 0.2
Y_MAX = 0.7
Z_MIN = 0.2
Z_MAX = 0.7
O_X_MIN = 45.0
O_X_MAX = 135.0
O_Y_MIN = 0.0
O_Y_MAX = 90.0
O_Z_MIN = 0.0
O_Z_MAX = 90.0
GRIP_MIN = 0.02 # gripper closed
GRIP_MAX = 0.08 # gripper open

# UMI ACTION LIMITS
X_ACTION_MIN = (X_MIN - X_MAX) / 2
X_ACTION_MAX = (X_MAX - X_MIN) / 2
Y_ACTION_MIN = (Y_MIN - Y_MAX) / 2
Y_ACTION_MAX = (Y_MAX - Y_MIN) / 2
Z_ACTION_MIN = (Z_MIN - Z_MAX) / 2
Z_ACTION_MAX = (Z_MAX - Z_MIN) / 2
O_X_ACTION_MIN = (O_X_MIN - O_X_MAX) / 2
O_X_ACTION_MAX = (O_X_MAX - O_X_MIN) / 2
O_Y_ACTION_MIN = (O_Y_MIN - O_Y_MAX) / 2
O_Y_ACTION_MAX = (O_Y_MAX - O_Y_MIN) / 2
O_Z_ACTION_MIN = (O_Z_MIN - O_Z_MAX) / 2
O_Z_ACTION_MAX = (O_Z_MAX - O_Z_MIN) / 2

# RT-1 ACTION SPACE
RT1_POS_MIN = -2.0
RT1_POS_MAX = 2.0
RT1_ROT_MIN = -np.pi
RT1_ROT_MAX = np.pi
RT1_GRIP_MIN = 1.0 # gripper closed
RT1_GRIP_MAX = -1.0 # gripper open

def rt1_outputs_to_umi_actions(rt1_outputs):
    gripper_closedness = rt1_outputs["gripper_closedness_action"]
    rotation_delta = rt1_outputs["rotation_delta"]
    terminate_episode = rt1_outputs["terminate_episode"]
    world_vector = rt1_outputs["world_vector"]

    pos_x = float(world_vector[0])
    pos_y = float(world_vector[1])
    pos_z = float(world_vector[2])
    roll = float(rotation_delta[0])
    pitch = float(rotation_delta[1])
    yaw = float(rotation_delta[2])
    grip = float(gripper_closedness[0])

    umi_pos_x = _rescale_dimension(pos_x, RT1_POS_MIN, RT1_POS_MAX, X_ACTION_MIN, X_ACTION_MAX)
    umi_pos_y = _rescale_dimension(pos_y, RT1_POS_MIN, RT1_POS_MAX, Y_ACTION_MIN, Y_ACTION_MAX)
    umi_pos_z = _rescale_dimension(pos_z, RT1_POS_MIN, RT1_POS_MAX, Z_ACTION_MIN, Z_ACTION_MAX)
    umi_roll = _rescale_dimension(roll, RT1_ROT_MIN, RT1_ROT_MAX, O_X_ACTION_MIN, O_X_ACTION_MAX)
    umi_pitch = _rescale_dimension(pitch, RT1_ROT_MIN, RT1_ROT_MAX, O_Y_ACTION_MIN, O_Y_ACTION_MAX)
    umi_yaw = _rescale_dimension(yaw, RT1_ROT_MIN, RT1_ROT_MAX, O_Z_ACTION_MIN, O_Z_ACTION_MAX)
    umi_grip = _rescale_dimension(grip, RT1_GRIP_MIN, RT1_GRIP_MAX, GRIP_MIN, GRIP_MAX)

    umi_action = copy.deepcopy(rt1_outputs)

    umi_action["world_vector"] = [umi_pos_x, umi_pos_y, umi_pos_z]
    umi_action["rotation_delta"] = [umi_roll, umi_pitch, umi_yaw]
    umi_action["gripper_closedness_action"] = [umi_grip]

    return umi_action

def umi_actions_to_rt1_inputs(umi_actions):
    umi_pos = umi_actions["world_vector"]
    umi_rot = umi_actions["rotation_delta"]
    umi_grip = umi_actions["gripper_closedness_action"]

    pos_x = float(umi_pos[0])
    pos_y = float(umi_pos[1])
    pos_z = float(umi_pos[2])
    roll = float(umi_rot[0])
    pitch = float(umi_rot[1])
    yaw = float(umi_rot[2])
    grip = float(umi_grip[0])

    rt1_pos_x = _rescale_dimension(pos_x, X_ACTION_MIN, X_ACTION_MAX, RT1_POS_MIN, RT1_POS_MAX)
    rt1_pos_y = _rescale_dimension(pos_y, Y_ACTION_MIN, Y_ACTION_MAX, RT1_POS_MIN, RT1_POS_MAX)
    rt1_pos_z = _rescale_dimension(pos_z, Z_ACTION_MIN, Z_ACTION_MAX, RT1_POS_MIN, RT1_POS_MAX)
    rt1_roll = _rescale_dimension(roll, O_X_ACTION_MIN, O_X_ACTION_MAX, RT1_ROT_MIN, RT1_ROT_MAX)
    rt1_pitch = _rescale_dimension(pitch, O_Y_ACTION_MIN, O_Y_ACTION_MAX, RT1_ROT_MIN, RT1_ROT_MAX)
    rt1_yaw = _rescale_dimension(yaw, O_Z_ACTION_MIN, O_Z_ACTION_MAX, RT1_ROT_MIN, RT1_ROT_MAX)
    rt1_grip = _rescale_dimension(grip, GRIP_MIN, GRIP_MAX, RT1_GRIP_MIN, RT1_GRIP_MAX)

    rt1_inputs = copy.deepcopy(umi_actions)

    rt1_inputs["world_vector"] = [rt1_pos_x, rt1_pos_y, rt1_pos_z]
    rt1_inputs["rotation_delta"] = [rt1_roll, rt1_pitch, rt1_yaw]
    rt1_inputs["gripper_closedness_action"] = [rt1_grip]

    return rt1_inputs

def rt1_outputs_to_umi_states(rt1_outputs):
    pos = rt1_outputs["world_vector"]
    rot = rt1_outputs["rotation_delta"]
    grip = rt1_outputs["gripper_closedness_action"]

    pos_x = float(pos[0])
    pos_y = float(pos[1])
    pos_z = float(pos[2])
    yaw = float(rot[0])
    pitch = float(rot[1])
    roll = float(rot[2])
    gripper_closedness = float(grip[0])
    
    umi_pos_x = _rescale_dimension(pos_x, -1.75, 1.75, X_MIN, X_MAX)
    umi_pos_y = _rescale_dimension(pos_y, -1.75, 1.75, Y_MIN, Y_MAX)
    umi_pos_z = _rescale_dimension(pos_z, -1.75, 1.75, Z_MIN, Z_MAX)

    # umi_pos_x = pos_x #/ 1.984
    # umi_pos_y = pos_y #/ 1.989
    # umi_pos_z = pos_z #/ 1.989

    umi_yaw = _rescale_dimension(yaw, -1.4, 1.4, O_X_MIN, O_X_MAX)
    umi_pitch = _rescale_dimension(pitch, -1.4, 1.4, O_Y_MIN, O_Y_MAX)
    umi_roll = _rescale_dimension(roll, -1.4, 1.4, O_Z_MIN, O_Z_MAX)
    umi_gripper_closedness = _rescale_dimension(gripper_closedness, RT1_GRIP_MIN, RT1_GRIP_MAX, GRIP_MIN, GRIP_MAX)

    print("Y OG: ", pos_y)
    print("Y UMI: ", umi_pos_y)

    umi_states = copy.deepcopy(rt1_outputs)

    umi_states["world_vector"] = [umi_pos_x, umi_pos_y, umi_pos_z]
    umi_states["rotation_delta"] = [umi_yaw, umi_pitch, umi_roll]
    umi_states["gripper_closedness_action"] = [umi_gripper_closedness]

    return umi_states

def _rescale_dimension(
    value: float,
    low: float,
    high: float,
    post_scaling_min: float = -1.0,
    post_scaling_max: float = 1.0,
) -> float:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    if value < low:
        print(f"VALUE BELOW LOW: {value} < {low}")
    if value > high:
        print(f"VALUE ABOVE HIGH: {value} > {high}")

    # val = value
    val = (value - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    if val < post_scaling_min:
        print("VALUE BELOW MIN")
        # return post_scaling_min
    if val > post_scaling_max:
        print("VALUE ABOVE MAX")
        # return post_scaling_max
    return val

def scale_back_to_umi(action):
        # in the finetuning code, we scale our actions to the UMI range,
        # which is 2;2 for coordinates and pi/2 for rotations. We need
        # to do the opposite of that here, so we get back to UMI range.
        # The world vector as existed in the dataset on disk ranges from -0.01 to 0.01

        scaled_action = copy.deepcopy(action)
        
        # We scale by 200.0 so that the action better spans the limit of the
        # world_vector action, from -2.0 to 2.0.
        scaled_action['world_vector'] = [i / 12.0 for i in action['world_vector']]

        # Similarly, the rotation_delta in the dataset on disk ranges from -2.0 to
        # 2.0
        # We scale by 0.79 so that the rotation_delta almost spans the limit of
        # rotation_delta, from -pi/2 to pi/2.
        scaled_action['rotation_delta'] = [i / 0.025 for i in action['rotation_delta']]

        # scale grip from space 0.02 to 0.08, to -1 to 1, with 0.02 being 1, and 0.08 being -1
        scaled_action['gripper_closedness_action'] = (
            _rescale_dimension(
                value=action['gripper_closedness_action'],
                low=1.0,
                high=-1.0,
                post_scaling_min=0.02,
                post_scaling_max=0.08,
            )
        )

        return scaled_action

def scale_rt1_to_bridge(action):
    scaled_action = copy.deepcopy(action)

    scaled_action['world_vector'][0] = _rescale_dimension(
        value=action['world_vector'][0],
        post_scaling_min=-0.05,
        post_scaling_max=0.05,
        low=-1.75,
        high=1.75,
    )

    scaled_action['world_vector'][1] = _rescale_dimension(
        value=action['world_vector'][1],
        post_scaling_min=-0.05,
        post_scaling_max=0.05,
        low=-1.75,
        high=1.75,
    )

    scaled_action['world_vector'][2] = _rescale_dimension(
        value=action['world_vector'][2],
        post_scaling_min=-0.05,
        post_scaling_max=0.05,
        low=-1.75,
        high=1.75,
    )

    scaled_action['rotation_delta'][0] = _rescale_dimension(
        value=action['rotation_delta'][0],
        post_scaling_min=-0.25,
        post_scaling_max=0.25,
        low=-1.4,
        high=1.4,
    )

    scaled_action['rotation_delta'][1] = _rescale_dimension(
        value=action['rotation_delta'][1],
        post_scaling_min=-0.25,
        post_scaling_max=0.25,
        low=-1.4,
        high=1.4,
    )

    scaled_action['rotation_delta'][2] = _rescale_dimension(
        value=action['rotation_delta'][2],
        post_scaling_min=-0.25,
        post_scaling_max=0.25,
        low=-1.4,
        high=1.4,
    )

    return scaled_action
    