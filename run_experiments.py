#!/usr/bin/env python
"""CLI entrypoint to run any experiment defined in config/*.yaml"""

import argparse
from trainers.train_sb3 import train

def main():
    """Parse args and dispatch to the chosen trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, help="Path to experiment YAML config"
    )
    args = parser.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
