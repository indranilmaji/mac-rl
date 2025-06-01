import gymnasium as gym
import torch
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ppo_unlock_baseline.zip", help="Path to trained model")
parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Evaluating on device: {device}")

env_id = "MiniGrid-Unlock-v0"

env = gym.make(env_id, render_mode="human")
# env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

model = PPO.load(args.model, device=device)

for ep in range(args.episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        time.sleep(0.05)

    print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

env.close()
