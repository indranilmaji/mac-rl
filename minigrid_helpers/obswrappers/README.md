# obswrappers

This folder holds **custom `ObservationWrapper`** subclasses for MiniGrid and other Gym environments. Each wrapper transforms the observation dict or array into a richer, model‐friendly format :contentReference[oaicite:3]{index=3}.

## Included Wrappers

- `MissionObsWrapper`: encodes textual missions into fixed‐size one-hot vectors.  
- `DirectionObsWrapper`: (if used) encodes agent’s orientation as a numeric or one-hot vector.  
- `ComposeObsWrapper`: chains multiple wrappers in a specified order.

## Usage

```python
from obswrappers.mission import MissionObsWrapper
import gymnasium as gym

env = gym.make("MiniGrid-Empty-5x5-v0")
env = MissionObsWrapper(env)
obs, _ = env.reset()
print(obs["mission_vec"].shape)  # e.g. (9,)
```