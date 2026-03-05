# Headless + W&B Training Entry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/train.py` support true headless training and optional W&B logging of core metrics and run config.

**Architecture:** Add a small CLI layer in `scripts/train.py` to control runtime flags (`--headless`, `--wandb*`) and pass a W&B run handle into the PPO runner. Keep existing `ml_logger/ml_dash` behavior intact and add W&B as an optional secondary sink. Emit per-iteration metrics from `Runner.learn` and finalize W&B run cleanly.

**Tech Stack:** Python 3.8, Isaac Gym, PyTorch, `ml_logger`, `params_proto`, `wandb`, pytest

---

### Task 1: Add CLI parsing and true headless propagation

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/test_train_cli_flags.py`

**Step 1: Write the failing test**

```python
from scripts.train import parse_args


def test_parse_headless_and_wandb_flags():
    args = parse_args([
        "--headless",
        "--wandb",
        "--wandb-project", "walk-these-ways",
        "--wandb-mode", "offline",
    ])
    assert args.headless is True
    assert args.wandb is True
    assert args.wandb_project == "walk-these-ways"
    assert args.wandb_mode == "offline"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_train_cli_flags.py::test_parse_headless_and_wandb_flags -v`
Expected: FAIL with import error for `parse_args` (not defined yet)

**Step 3: Write minimal implementation**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="walk-these-ways")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    return parser.parse_args(argv)
```

Also wire headless into env creation:

```python
env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
```

And call `train_go1(...)` from parsed args in `if __name__ == '__main__':`.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_train_cli_flags.py::test_parse_headless_and_wandb_flags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train.py tests/test_train_cli_flags.py
git commit -m "feat: add train CLI flags and true headless mode"
```

### Task 2: Add optional W&B run initialization in train entry

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/test_train_cli_flags.py`

**Step 1: Write the failing test**

```python
from scripts.train import build_wandb_config


def test_build_wandb_config_contains_sections():
    cfg = build_wandb_config(
        {"a": 1},
        {"b": 2},
        {"c": 3},
        {"env": {"num_envs": 4000}},
    )
    assert set(cfg.keys()) == {"AC_Args", "PPO_Args", "RunnerArgs", "Cfg"}
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_train_cli_flags.py::test_build_wandb_config_contains_sections -v`
Expected: FAIL with missing `build_wandb_config`

**Step 3: Write minimal implementation**

```python
def build_wandb_config(ac_args, ppo_args, runner_args, cfg_dict):
    return {
        "AC_Args": ac_args,
        "PPO_Args": ppo_args,
        "RunnerArgs": runner_args,
        "Cfg": cfg_dict,
    }
```

And in `train_go1(...)`:
- if `use_wandb`:
  - import `wandb`
  - call `wandb.init(project=..., entity=..., name=..., mode=..., config=build_wandb_config(...))`
  - pass resulting run handle into `Runner(...)`
- if import fails: raise clear RuntimeError with install hint.
- ensure `wandb_run.finish()` is called after training.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_train_cli_flags.py::test_build_wandb_config_contains_sections -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train.py tests/test_train_cli_flags.py
git commit -m "feat: add optional wandb initialization for train entry"
```

### Task 3: Emit per-iteration core metrics to W&B in PPO runner

**Files:**
- Modify: `go1_gym_learn/ppo_cse/__init__.py`
- Test: `tests/test_runner_wandb_payload.py`

**Step 1: Write the failing test**

```python
from go1_gym_learn.ppo_cse import _build_wandb_metrics_payload


def test_build_wandb_metrics_payload_has_core_keys():
    payload = _build_wandb_metrics_payload(
        iteration=7,
        timesteps=12345,
        adaptation_loss=1.2,
        mean_value_loss=0.3,
        mean_surrogate_loss=0.1,
        mean_adaptation_module_test_loss=1.1,
        time_elapsed=10.0,
        time_iter=0.5,
        train_reward_mean=2.3,
        train_ep_len_mean=120.0,
        eval_reward_mean=1.8,
        eval_ep_len_mean=100.0,
    )
    assert payload["iterations"] == 7
    assert payload["timesteps"] == 12345
    assert payload["adaptation_loss"] == 1.2
    assert payload["train/episode/rew_total/mean"] == 2.3
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_runner_wandb_payload.py::test_build_wandb_metrics_payload_has_core_keys -v`
Expected: FAIL with missing `_build_wandb_metrics_payload`

**Step 3: Write minimal implementation**

```python
def _build_wandb_metrics_payload(...):
    payload = {
        "iterations": iteration,
        "timesteps": timesteps,
        "adaptation_loss": adaptation_loss,
        "mean_value_loss": mean_value_loss,
        "mean_surrogate_loss": mean_surrogate_loss,
        "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss,
        "time_elapsed": time_elapsed,
        "time_iter": time_iter,
    }
    if train_reward_mean is not None:
        payload["train/episode/rew_total/mean"] = train_reward_mean
    if train_ep_len_mean is not None:
        payload["train/episode/ep_len/mean"] = train_ep_len_mean
    if eval_reward_mean is not None:
        payload["eval/episode/rew_total/mean"] = eval_reward_mean
    if eval_ep_len_mean is not None:
        payload["eval/episode/ep_len/mean"] = eval_ep_len_mean
    return payload
```

Update `Runner.__init__` to accept `wandb_run=None`, store it, and in `learn(...)` call:

```python
if self.wandb_run is not None:
    self.wandb_run.log(payload, step=it)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_runner_wandb_payload.py::test_build_wandb_metrics_payload_has_core_keys -v`
Expected: PASS

**Step 5: Commit**

```bash
git add go1_gym_learn/ppo_cse/__init__.py tests/test_runner_wandb_payload.py
git commit -m "feat: log core PPO metrics to wandb each iteration"
```

### Task 4: End-to-end CLI verification (headless + W&B offline)

**Files:**
- Modify: none
- Test: runtime command verification

**Step 1: Write the failing test**

```python
def test_smoke_command_documented():
    assert True
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_train_cli_flags.py::test_smoke_command_documented -v`
Expected: Optional for documentation-only smoke task

**Step 3: Run minimal implementation verification command**

Run: `python3 scripts/train.py --headless --wandb --wandb-mode offline`
Expected:
- training initializes without GUI
- no crash on W&B init
- iteration logs appear in console

**Step 4: Verify artifacts**

Run: `ls wandb`
Expected: local offline run directory exists

**Step 5: Commit**

```bash
git add scripts/train.py go1_gym_learn/ppo_cse/__init__.py tests/
git commit -m "feat: support headless CLI training with optional wandb logging"
```
