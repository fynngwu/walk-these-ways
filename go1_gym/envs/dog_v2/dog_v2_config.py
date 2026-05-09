from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1


def config_dog_v2(Cnfg: Union[Cfg, Meta]):
    config_go1(Cnfg)

    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 0.34]
    _.default_joint_angles = {
        'FL_hip_joint': 0.0,
        'RL_hip_joint': 0.0,
        'FR_hip_joint': 0.0,
        'RR_hip_joint': 0.0,
        'FL_thigh_joint': 0.0,
        'RL_thigh_joint': 0.0,
        'FR_thigh_joint': 0.0,
        'RR_thigh_joint': 0.0,
        'FL_calf_joint': 0.0,
        'RL_calf_joint': 0.0,
        'FR_calf_joint': 0.0,
        'RR_calf_joint': 0.0,
    }

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/dog_v2_description/urdf/dog_v2_2_4.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base"]

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 20.0}
    _.damping = {'joint': 0.5}

    _ = Cnfg.rewards
    _.base_height_target = 0.30
