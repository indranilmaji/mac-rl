# trainers

This folder contains **training scripts** that orchestrate experiments using stable-baselines3 or custom RL loops. Each trainer reads a YAML config to build envs, instantiate models, and log metrics.

## trainers/train_sb3.py

A general trainer for SB3 algorithms (PPO, DQN, etc.):

```bash
python -m trainers.train_sb3 --config config/base.yaml
```