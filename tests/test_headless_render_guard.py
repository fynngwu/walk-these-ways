import numpy as np

from go1_gym.envs.base.legged_robot import LeggedRobot


class _DummyGym:
    def set_camera_location(self, *_args, **_kwargs):
        return None

    def get_camera_image(self, *_args, **_kwargs):
        return np.array([], dtype=np.uint8)


class _DummyCameraProps:
    height = 240
    width = 360


class _DummyEnv:
    pass


class _DummyRobot:
    def __init__(self):
        self.record_now = True
        self.complete_video_frames = []
        self.root_states = np.zeros((1, 3), dtype=np.float32)
        self.gym = _DummyGym()
        self.sim = object()
        self.envs = [_DummyEnv()]
        self.rendering_camera = object()
        self.camera_props = _DummyCameraProps()
        self.video_frames = []

        self.record_eval_now = False
        self.complete_video_frames_eval = []
        self.eval_cfg = None


def test_render_headless_skips_empty_camera_frames_without_crashing():
    robot = _DummyRobot()

    LeggedRobot._render_headless(robot)

    assert robot.video_frames == []
