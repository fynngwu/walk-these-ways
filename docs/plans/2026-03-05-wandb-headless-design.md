# Train Script Headless + W&B Design

## Goal

Enable direct headless launch from `scripts/train.py` and optional Weights & Biases logging for core training metrics and run configuration.

## Current State

- `scripts/train.py` exposes `train_go1(headless=True)` but hardcodes `headless=False` when creating `VelocityTrackingEasyEnv`.
- `scripts/train.py` has no CLI argument parsing, so `--headless` is ignored today.
- Training logs are emitted through `ml_logger` and visualized by `ml_dash`.
- No `wandb` integration exists in the repository.

## Options Considered

1. Patch only `scripts/train.py` and keep W&B out of runner internals.
   - Pro: minimal surface area.
   - Con: poor real-time metric fidelity and awkward metric extraction.

2. Add CLI in `scripts/train.py` and add W&B hooks inside PPO runner update loop. (selected)
   - Pro: direct `--headless` usability and accurate per-iteration metrics.
   - Con: modest changes in both entry script and runner.

3. Keep training untouched and add offline post-sync script to W&B.
   - Pro: safest for training path.
   - Con: not real-time and does not match requested workflow.

## Selected Design

### 1) CLI and Headless Behavior

- Add `argparse` to `scripts/train.py`.
- Support `--headless` and pass it through to env creation.
- Keep default behavior backward-compatible: no flag means GUI mode, `--headless` means no rendering window.

### 2) W&B Optional Logging

- Add CLI flags:
  - `--wandb`
  - `--wandb-project`
  - `--wandb-entity`
  - `--wandb-run-name`
  - `--wandb-mode` (`online` or `offline`)
- If `--wandb` is set, initialize a W&B run in `scripts/train.py` and pass the run object into `Runner`.
- Keep `ml_logger`/`ml_dash` unchanged; W&B is additive.

### 3) Metrics Scope (requested: basic metrics + config)

- Log config once at startup: `AC_Args`, `PPO_Args`, `RunnerArgs`, and flattened `Cfg` snapshot.
- Log per-iteration core metrics from `Runner.learn`:
  - `adaptation_loss`
  - `mean_value_loss`
  - `mean_surrogate_loss`
  - `mean_adaptation_module_test_loss`
  - `time_elapsed`
  - `time_iter`
  - `timesteps`
  - `iterations`
- Also log episode summary means when available:
  - train/eval episode return mean and episode length mean.

### 4) Failure and UX

- If `--wandb` is set but `wandb` is missing, fail early with a clear error message.
- Always `finish()` W&B run after training ends (including interruption-safe finalize path).

## Non-Goals

- No checkpoint artifact upload to W&B.
- No TensorBoard integration in this change.
- No changes to core PPO update math or environment dynamics.

## Validation Plan

1. CLI parse check: `python scripts/train.py --help` shows new flags.
2. Headless launch check: `python scripts/train.py --headless` starts training without GUI.
3. W&B smoke check: `python scripts/train.py --headless --wandb --wandb-mode offline` runs and writes W&B logs locally.
