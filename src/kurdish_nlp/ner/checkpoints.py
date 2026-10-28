"""Read-only restoration/inspection of immutable NER backup generations.

Restoration creates a fresh local directory. This module never trains a model
or writes to the source backup. Use the isolated training Python for full-state
restores so stack compatibility can be checked without loading model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .colab_artifacts import restore_run
from .training_state import latest_generation, restore_training, validate_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("restore-weights")
    legacy.add_argument("--drive-run", type=Path, required=True)
    legacy.add_argument("--generation", required=True)
    legacy.add_argument("--output", type=Path, required=True)
    full = commands.add_parser("restore-state")
    full.add_argument("--state-root", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("verify-state")
    choice = inspect.add_mutually_exclusive_group(required=True)
    choice.add_argument("--generation", type=Path)
    choice.add_argument("--state-root", type=Path)
    args = parser.parse_args()
    if args.command == "restore-weights":
        result = restore_run(args.drive_run, args.output, generation=args.generation)
        print(
            json.dumps(
                {
                    "restored": str(result),
                    "resume_state": "weights_only_no_optimizer_or_scheduler",
                }
            )
        )
    elif args.command == "restore-state":
        print(
            json.dumps(
                {"restored": str(restore_training(args.state_root, args.output))}
            )
        )
    else:
        generation = (
            latest_generation(args.state_root) if args.state_root else args.generation
        )
        print(json.dumps(validate_generation(generation), indent=2))


if __name__ == "__main__":
    main()
