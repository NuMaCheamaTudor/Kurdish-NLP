"""Bootstrap an isolated Linux/CUDA environment without mutating Colab's kernel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def install(repo: Path, environment: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Install only in the Linux x86_64 Colab runtime")
    print("Notebook Python:", platform.python_version(), flush=True)
    python = environment / "bin/python"
    if not python.exists():
        if sys.version_info[:2] == (3, 12):
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        else:
            # New Colab Python versions need not force an unvalidated model stack.
            # uv is used only to obtain an isolated compatible interpreter.
            bootstrap = environment.parent / "ner-python-bootstrap"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--target",
                    str(bootstrap),
                    "uv==0.8.22",
                ],
                check=True,
            )
            env = dict(os.environ, PYTHONPATH=str(bootstrap))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "uv",
                    "venv",
                    "--seed",
                    "--python",
                    "3.12",
                    str(environment),
                ],
                env=env,
                check=True,
            )
            print(
                "Active kernel Python differs from 3.12; selected isolated Python 3.12 (NetworkX 3.7 requires it). Model versions are unchanged.",
                flush=True,
            )
    version = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import sys,json;print(json.dumps(list(sys.version_info[:2])))",
            ],
            text=True,
        )
    )
    if version != [3, 12]:
        raise RuntimeError(
            "Existing training interpreter is incompatible; choose a new environment path"
        )
    requirement = repo / "configs/ner/requirements-colab-cu124.lock.txt"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "-r",
            str(requirement),
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print(
        "Training Python:",
        subprocess.check_output([str(python), "-V"], text=True).strip(),
    )
    print(
        "Kernel restart not required: all ML imports and training execute in fresh isolated subprocesses."
    )
    print(
        "If the driver/CUDA probe fails, choose a fresh T4 runtime; do not ignore the preflight failure."
    )
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    install(args.repo, args.environment)
