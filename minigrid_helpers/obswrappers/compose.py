from gymnasium import ObservationWrapper, Env
from typing import List, Type

class ComposeObsWrapper(ObservationWrapper):
    """Chains multiple ObservationWrappers into one."""

    def __init__(self, env: Env, wrappers: List[Type[ObservationWrapper]]):
        """Initialize with base env and a list of wrapper classes.

        Args:
            env: the Gym environment to wrap.
            wrappers: a list of ObservationWrapper subclasses to apply.
        """
        super().__init__(env)
        self.wrappers = []
        space = env.observation_space
        for W in wrappers:
            wrapped = W(self.env)
            wrapped.env = type("tmp_env", (), {"observation_space": space})()
            space = wrapped.observation_space
            self.wrappers.append(Wrapped := W(self.env))
        self.observation_space = space

    def observation(self, obs):
        """Apply each wrapped ObservationWrapper in sequence."""
        for W in self.wrappers:
            obs = W.observation(obs)
        return obs
