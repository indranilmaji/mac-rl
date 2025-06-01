from gymnasium import make
from obswrappers.compose import ComposeObsWrapper
from typing import List
import logging
import minigrid

# Registry mapping wrapper names to import paths
AVAILABLE_WRAPPERS = {
    "mission": "obswrappers.mission:MissionObsWrapper",
    "direction": "obswrappers.direction:DirectionObsWrapper",
    # add more as needed
}


def make_env(env_id: str, wrappers: List[str] = None, **kwargs):
    """Factory to create and wrap Gym environments.

    Args:
        env_id: the Gymnasium environment ID (e.g. "MiniGrid-Fetch-8x8-N3-v0").
        wrappers: list of keys from AVAILABLE_WRAPPERS to apply, in order.
        **kwargs: passed to gym.make (e.g. render_mode).

    Returns:
        A wrapped Gym environment ready for training or evaluation.
    """
    print("Env Name:", env_id)
    env = make(env_id)
    if wrappers:
        wrapper_classes = []
        for name in wrappers:
            print(name)
            module_path, cls_name = AVAILABLE_WRAPPERS[name].split(":")
            module = __import__(module_path, fromlist=[cls_name])
            wrapper_classes.append(getattr(module, cls_name))
        env = ComposeObsWrapper(env, wrapper_classes)
    return env
