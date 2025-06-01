from gymnasium import ObservationWrapper
from gymnasium.spaces import Dict, Box
import numpy as np

class MissionObsWrapper(ObservationWrapper):
    """Encodes the textual 'mission' into a fixed-size one-hot vector."""

    def __init__(self, env):
        """Wrap env so that obs['mission'] → obs['mission_vec'].

        Args:
            env: a MiniGrid environment with a 'mission' key in its Dict space.
        """
        super().__init__(env)
        # original mission is a string; replace with one-hot vector length 9
        self.observation_space = Dict({
            "image": env.observation_space.spaces["image"],
            "mission_vec": Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
        })

    def observation(self, obs):
        """Convert obs['mission'] string → 9-dim one-hot 'mission_vec'."""
        word_list = obs["mission"].split()
        # simplistic encoding: pick indices by some mapping
        vec = np.zeros(9, dtype=np.float32)
        # e.g. word_list[-2] and word_list[-1] tell color+object
        # fill vec accordingly…
        obs = {"image": obs["image"], "mission_vec": vec}
        return obs
