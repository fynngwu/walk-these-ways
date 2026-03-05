from scripts.train import build_wandb_config, parse_args


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
