import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.dog_v2.dog_v2_config import config_dog_v2
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    dirs = glob.glob(str(root / "runs" / label / "*"))
    logdir = sorted(dirs)[-1]
    print(f"Loading policy from {logdir}")

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


class KeyboardController:
    def __init__(self):
        from pynput import keyboard as _kb
        self._kb = _kb
        self.keys = set()
        self.listener = _kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            self.keys.add(key.char.lower())
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            self.keys.discard(key.char.lower())
        except AttributeError:
            if key == self._kb.Key.esc:
                self.listener.stop()
                return False

    def get_vel(self, max_x=1.5, max_y=0.6, max_yaw=2.0):
        x = max_x * int('w' in self.keys) - max_x * int('s' in self.keys)
        y = max_y * int('a' in self.keys) - max_y * int('d' in self.keys)
        yaw = max_yaw * int('q' in self.keys) - max_yaw * int('e' in self.keys)
        return float(x), float(y), float(yaw)


def play_go1(headless=True):
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/2026-05-09/train"

    env, policy = load_env(label, headless=headless)

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}
    gait_names = list(gaits.keys())

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait_idx = [1]
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    kb = KeyboardController()

    print("=== Keyboard Control ===")
    print("W/S: forward/backward")
    print("A/D: left/right")
    print("Q/E: turn left/right")
    print("1/2/3/4: pronking/trotting/bounding/pacing")
    print("Z/X: stance width down/up")
    print("F/R: frequency down/up")
    print("G/T: foot swing height down/up")
    print("ESC: quit")
    print("========================")

    obs = env.reset()
    step = 0

    while True:
        if not kb.listener.is_alive():
            break

        if '1' in kb.keys: gait_idx[0] = 0
        if '2' in kb.keys: gait_idx[0] = 1
        if '3' in kb.keys: gait_idx[0] = 2
        if '4' in kb.keys: gait_idx[0] = 3
        if 'z' in kb.keys: stance_width_cmd = max(0.10, stance_width_cmd - 0.002)
        if 'x' in kb.keys: stance_width_cmd = min(0.45, stance_width_cmd + 0.002)
        if 'f' in kb.keys: step_frequency_cmd = max(2.0, step_frequency_cmd - 0.01)
        if 'r' in kb.keys: step_frequency_cmd = min(4.0, step_frequency_cmd + 0.01)
        if 'g' in kb.keys: footswing_height_cmd = max(0.03, footswing_height_cmd - 0.001)
        if 't' in kb.keys: footswing_height_cmd = min(0.35, footswing_height_cmd + 0.001)

        x_vel, y_vel, yaw_vel = kb.get_vel()
        gait = torch.tensor(gaits[gait_names[gait_idx[0]]])

        with torch.no_grad():
            actions = policy(obs)

        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        if step % 50 == 0:
            vel = env.base_lin_vel[0, 0].item()
            print(f"step={step} gait={gait_names[gait_idx[0]]} cmd=({x_vel:.1f},{y_vel:.1f},{yaw_vel:.1f}) freq={step_frequency_cmd:.2f} swing={footswing_height_cmd:.3f} stance_width={stance_width_cmd:.3f} vel_x={vel:.2f} action=[{actions[0,:].min().item():.3f},{actions[0,:].max().item():.3f}] obs_range=[{obs['obs_history'][0].min().item():.3f},{obs['obs_history'][0].max().item():.3f}]")
        step += 1


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
