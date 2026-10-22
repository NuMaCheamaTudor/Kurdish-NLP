"""Discover pinned wheel CUDA libraries without importing any ML framework.

LD_LIBRARY_PATH must reach the dynamic loader at process launch, not after
CuPy/PyTorch have already been imported in a notebook kernel.
"""

from __future__ import annotations

import json
import os
import platform
import site
import sys
from pathlib import Path


def runtime_environment(environ=None, roots=None):
    env = dict(os.environ if environ is None else environ)
    roots = [Path(p) for p in (site.getsitepackages() if roots is None else roots)]
    libraries = []
    for root in roots:
        libraries.extend(sorted(root.glob("nvidia/*/lib")))
        libraries.append(root / "torch/lib")
    libraries = [str(p.resolve()) for p in libraries if p.is_dir()]
    previous = env.get("LD_LIBRARY_PATH", "").split(":")
    env["LD_LIBRARY_PATH"] = ":".join(
        dict.fromkeys(libraries + [p for p in previous if p])
    )
    # CuPy needs CUDA's headers as well as shared libraries for NVRTC kernels.
    for root in roots:
        runtime = root / "nvidia/cuda_runtime"
        if (runtime / "include/cuda_runtime.h").is_file():
            env["CUDA_PATH"] = str(runtime.resolve())
            break
    return env


def bootstrap():
    if platform.system() == "Linux":
        env = runtime_environment()
        if any(
            os.environ.get(k) != env.get(k) for k in ("LD_LIBRARY_PATH", "CUDA_PATH")
        ):
            os.execve(
                sys.executable,
                [sys.executable, "-m", "kurdish_nlp.ner.colab", *sys.argv[1:]],
                env,
            )


if __name__ == "__main__":
    env = runtime_environment()
    print(json.dumps({k: env[k] for k in ("LD_LIBRARY_PATH", "CUDA_PATH") if k in env}))
