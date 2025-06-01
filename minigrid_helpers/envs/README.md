# envs

This directory contains the **environment factory** and related utilities for creating and wrapping Gym environments.

## Overview

The `make_env` function provides a unified interface to:
- Instantiate any Gym environment by its ID
- Apply a configurable sequence of `ObservationWrapper` classes
- Pass through additional `gym.make` arguments (e.g., `render_mode`)

Using a factory ensures consistency across experiments and makes it trivial to add new environments or wrappers without changing training code :contentReference[oaicite:0]{index=0}.

## Usage

```python
from envs.factory import make_env

# Create a MiniGrid Fetch env with mission and direction wrappers
env = make_env(
    "MiniGrid-Fetch-8x8-N3-v0",
    wrappers=["mission", "direction"],
    render_mode="rgb_array"
)
```
