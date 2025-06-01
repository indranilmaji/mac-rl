import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from envs.factory import make_env

def train(config_path: str):
    """Load YAML config, build env, train PPO, and save checkpoints.

    Args:
        config_path: Path to experiment YAML (includes env_id, wrappers, hyperparams).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Create directories
    os.makedirs(os.path.join(cfg["exp_name"] + cfg["log_dir"]), exist_ok=True)
    os.makedirs(os.path.join(cfg["exp_name"] + cfg["checkpoint_dir"]), exist_ok=True)
    
    # Build env
    env = make_env(
        cfg["env_id"],
        wrappers=cfg.get("wrappers", []),
        render_mode=cfg.get("render_mode", None),
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)

    # Instantiate model
    model = PPO(
        cfg["policy"],
        vec_env,
        policy_kwargs=cfg.get("policy_kwargs", {}),
        verbose=1,
        tensorboard_log=cfg["log_dir"],
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["save_freq"], save_path=cfg["checkpoint_dir"], name_prefix="iter"
    )
    eval_cb = EvalCallback(
        vec_env,
        best_model_save_path=cfg["best_dir"],
        log_path=cfg["best_dir"],
        eval_freq=cfg["eval_freq"],
        deterministic=True,
    )

    # Train
    model.learn(
        total_timesteps=cfg["timesteps"],
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=cfg.get("tb_name", "run"),
    )
    model.save(os.path.join(cfg["best_dir"], "final_model"))
