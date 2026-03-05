from scripts.train import build_wandb_config, configure_headless_runtime, parse_args


class _DummyEnvCfg:
    record_video = True


class _DummyCfg:
    env = _DummyEnvCfg()


class _DummyRunnerArgs:
    save_video_interval = 100


def test_parse_headless_and_wandb_flags():
    args, unknown = parse_args(
        [
            "--headless",
            "--wandb",
            "--wandb-project",
            "walk-these-ways",
            "--wandb-entity",
            "robotics-team",
            "--wandb-run-name",
            "nightly-run",
            "--wandb-mode",
            "offline",
        ]
    )
    assert args.headless is True
    assert args.wandb is True
    assert args.wandb_project == "walk-these-ways"
    assert args.wandb_entity == "robotics-team"
    assert args.wandb_run_name == "nightly-run"
    assert args.wandb_mode == "offline"
    assert unknown == []


def test_build_wandb_config_contains_sections():
    cfg = build_wandb_config(
        {"a": 1},
        {"b": 2},
        {"c": 3},
        {"env": {"num_envs": 4000}},
    )
    assert set(cfg.keys()) == {"AC_Args", "PPO_Args", "RunnerArgs", "Cfg"}


def test_configure_headless_runtime_disables_video_and_interval():
    _DummyCfg.env.record_video = True
    _DummyRunnerArgs.save_video_interval = 100

    configure_headless_runtime(headless=True, cfg=_DummyCfg, runner_args_cls=_DummyRunnerArgs)

    assert _DummyCfg.env.record_video is False
    assert _DummyRunnerArgs.save_video_interval == 0


def test_configure_headless_runtime_keeps_defaults_when_not_headless():
    _DummyCfg.env.record_video = True
    _DummyRunnerArgs.save_video_interval = 100

    configure_headless_runtime(headless=False, cfg=_DummyCfg, runner_args_cls=_DummyRunnerArgs)

    assert _DummyCfg.env.record_video is True
    assert _DummyRunnerArgs.save_video_interval == 100
