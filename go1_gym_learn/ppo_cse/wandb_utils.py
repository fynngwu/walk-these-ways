def mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def build_wandb_metrics_payload(
    iteration,
    timesteps,
    adaptation_loss,
    mean_value_loss,
    mean_surrogate_loss,
    mean_adaptation_module_test_loss,
    time_elapsed,
    time_iter,
    train_reward_mean=None,
    train_ep_len_mean=None,
    eval_reward_mean=None,
    eval_ep_len_mean=None,
):
    payload = {
        "iterations": int(iteration),
        "timesteps": int(timesteps),
        "adaptation_loss": float(adaptation_loss),
        "mean_value_loss": float(mean_value_loss),
        "mean_surrogate_loss": float(mean_surrogate_loss),
        "mean_adaptation_module_test_loss": float(mean_adaptation_module_test_loss),
        "time_elapsed": float(time_elapsed),
        "time_iter": float(time_iter),
    }
    if train_reward_mean is not None:
        payload["train/episode/rew_total/mean"] = float(train_reward_mean)
    if train_ep_len_mean is not None:
        payload["train/episode/ep_len/mean"] = float(train_ep_len_mean)
    if eval_reward_mean is not None:
        payload["eval/episode/rew_total/mean"] = float(eval_reward_mean)
    if eval_ep_len_mean is not None:
        payload["eval/episode/ep_len/mean"] = float(eval_ep_len_mean)
    return payload
