"""Official fastText supervised classifier baseline.

Training reads a UTF-8 supervised file produced by the shared formatter
(``dataset.format_fasttext_example``). With ``thread=1`` fastText's SGD runs
sequentially, so a fixed seed yields a deterministic model.

fastText 0.9.3 is refused for training with fewer than 11 threads. Its input
matrix became an ``intgemm::AlignedVector`` allocated with ``posix_memalign``
(uninitialized), while ``DenseMatrix::uniform`` initializes only ``thread``
tenths of it, so most embedding rows start as leftover heap bytes: runs with
identical seeds differ and can diverge to NaN. 0.9.2 keeps a zero-initialized
``std::vector``; with ``thread=1`` it initializes the first tenth uniformly and
the rest to zero, deterministically. Loading saved models is unaffected.

Probabilities come from fastText's list prediction API with ``k=-1`` (all
labels). fastText reports ``p + 1e-5`` per label; the offset is removed and each
row is renormalized, then columns are reordered to the experiment's labels.
"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.dataset import FASTTEXT_LABEL_PREFIX
from kurdish_nlp.langid.experiments.evaluation import renormalize
from kurdish_nlp.langid.fasttext_detector import fasttext_probability

FAMILY = "fasttext"
# Arguments of fasttext.train_supervised this experiment may set.
TRAIN_ARGUMENTS = frozenset(
    {
        "lr",
        "dim",
        "ws",
        "epoch",
        "minCount",
        "minCountLabel",
        "minn",
        "maxn",
        "neg",
        "wordNgrams",
        "loss",
        "bucket",
        "thread",
        "lrUpdateRate",
        "t",
        "verbose",
        "seed",
    }
)


def _fasttext():
    try:
        import fasttext  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "the fastText baseline needs the official 'fasttext' package; install "
            "the project with the 'experiments' extra"
        ) from error
    return fasttext


UNINITIALIZED_MATRIX_VERSIONS = frozenset({"0.9.3"})
FULL_INIT_MIN_THREADS = 11


def check_training_safety(version: str, threads: int) -> None:
    """Refuse fastText builds that would train on uninitialized memory."""

    if version in UNINITIALIZED_MATRIX_VERSIONS and threads < FULL_INIT_MIN_THREADS:
        raise RuntimeError(
            f"fastText {version} leaves most of its input matrix uninitialized when "
            f"thread < {FULL_INIT_MIN_THREADS} (posix_memalign + partial uniform "
            "init), making seeded training non-deterministic and NaN-prone; install "
            "the official fasttext==0.9.2 (the 'experiments' extra) for training"
        )


def library_versions() -> dict[str, str]:
    return {
        "fasttext": importlib.metadata.version("fasttext"),
        "numpy": importlib.metadata.version("numpy"),
        "python": platform.python_version(),
    }


def memory_estimate(params: Mapping[str, Any], vocabulary: int) -> int:
    """Input + output matrices of a supervised model, in bytes (float32)."""

    rows = vocabulary + int(params["bucket"])
    return int(rows * int(params["dim"]) * 4 * 2)


class FastTextModel:
    """A trained fastText classifier whose probability columns follow ``labels``."""

    family = FAMILY

    def __init__(self, model: Any, labels: Sequence[str]) -> None:
        self.model = model
        self.labels = tuple(labels)
        model_labels = [
            label.removeprefix(FASTTEXT_LABEL_PREFIX) for label in model.get_labels()
        ]
        if sorted(model_labels) != sorted(self.labels):
            raise ValueError(
                f"fastText labels {sorted(model_labels)} differ from {sorted(self.labels)}"
            )
        self._index = {label: i for i, label in enumerate(self.labels)}

    @classmethod
    def train(
        cls,
        train_file: str | Path,
        params: Mapping[str, Any],
        *,
        seed: int,
        labels: Sequence[str],
    ) -> FastTextModel:
        unknown = set(params) - TRAIN_ARGUMENTS
        if unknown:
            raise ValueError(f"unsupported fastText arguments: {sorted(unknown)}")
        if params.get("thread") != 1:
            raise ValueError("fastText baselines train with thread=1 (determinism)")
        module = _fasttext()
        check_training_safety(
            importlib.metadata.version("fasttext"), int(params["thread"])
        )
        model = module.train_supervised(
            input=str(train_file), seed=int(seed), **dict(params)
        )
        return cls(model, labels)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        batch = list(texts)
        result = np.zeros((len(batch), len(self.labels)), dtype=np.float64)
        if not batch:
            return result
        all_labels, all_scores = self.model.predict(batch, k=-1, threshold=0.0)
        for row, (labels, scores) in enumerate(
            zip(all_labels, all_scores, strict=True)
        ):
            for label, score in zip(labels, scores, strict=True):
                column = self._index[label.removeprefix(FASTTEXT_LABEL_PREFIX)]
                result[row, column] = fasttext_probability(score)
        return renormalize(result)

    @property
    def vocabulary_size(self) -> int:
        return len(self.model.get_words())

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(destination))
        return {
            "format": "fasttext-bin",
            "family": FAMILY,
            "labels": list(self.labels),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "vocabulary_size": self.vocabulary_size,
            "versions": library_versions(),
        }

    @classmethod
    def load(cls, path: str | Path, labels: Sequence[str]) -> FastTextModel:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"fastText model not found: {source}")
        return cls(_fasttext().load_model(str(source)), labels)
