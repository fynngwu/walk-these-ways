from go1_gym_learn.ppo_cse.wandb_utils import build_wandb_metrics_payload


def test_build_wandb_metrics_payload_has_core_keys():
    payload = build_wandb_metrics_payload(
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
    assert payload["eval/episode/rew_total/mean"] == 1.8
