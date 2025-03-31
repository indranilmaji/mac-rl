import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import argparse

# --------------------------
# Custom CNN Feature Extractor
# --------------------------
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically infer the number of flattened features
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# --------------------------
# CLI Config
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Enable RGB rendering during training")
parser.add_argument("--timesteps", type=int, default=200000, help="Training steps")
args = parser.parse_args()

# --------------------------
# Environment Setup
# --------------------------
env_id = "MiniGrid-Fetch-8x8-N3-v0"
render_mode = "rgb_array" if args.render else None
print("Render Mode: ", render_mode)
env = gym.make(env_id, render_mode=render_mode)
env = RGBImgObsWrapper(env)     # adds 'image' key
env = ImgObsWrapper(env)        # makes observation a 3D tensor (HWC)

# --------------------------
# Policy Config
# --------------------------
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)

# --------------------------
# PPO + Training
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = PPO("CnnPolicy", env, learning_rate=1e-3, policy_kwargs=policy_kwargs, verbose=1, device=device, tensorboard_log="./logs")

model.learn(total_timesteps=args.timesteps)
model.save("ppo_minigrid_fetch")
