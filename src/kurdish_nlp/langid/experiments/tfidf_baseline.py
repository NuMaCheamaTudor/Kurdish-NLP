"""Character TF-IDF + multinomial logistic regression baseline.

The vectorizer runs on the project's conservatively normalized text with
``lowercase=False`` and ``strip_accents=None``, so no Unicode distinction is
folded. Features stay in a sparse CSR matrix throughout; nothing is densified.

Artifacts are a joblib pickle plus a JSON sidecar recording its SHA-256 and the
library versions. joblib/pickle is a *trusted local* format only: ``load``
refuses a file whose hash differs from its sidecar or that was written by a
different scikit-learn version.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.experiments.evaluation import renormalize

FAMILY = "tfidf_lr"
SIDECAR_SUFFIX = ".meta.json"
FORMAT = "joblib-sklearn-pipeline-v1"


def library_versions() -> dict[str, str]:
    names = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
    versions = {name: importlib.metadata.version(name) for name in names}
    versions["python"] = platform.python_version()
    return versions


def build_vectorizer(fixed: Mapping[str, Any], params: Mapping[str, Any]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    options = dict(fixed)
    options.update(params)
    options["ngram_range"] = tuple(options["ngram_range"])
    options["dtype"] = np.dtype(options.get("dtype", "float32")).type
    return TfidfVectorizer(**options)


def build_classifier(fixed: Mapping[str, Any], params: Mapping[str, Any], seed: int):
    from sklearn.linear_model import LogisticRegression

    options = dict(fixed)
    options.update(params)
    return LogisticRegression(random_state=seed, **options)


def lbfgs_memory_estimate(n_features: int, n_classes: int, history: int = 10) -> int:
    """Rough upper bound (bytes) of L-BFGS state for a multinomial model."""

    return int(n_features * n_classes * 8 * (2 * history + 6))


def csr_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def fit_classifier(classifier, features, y: Sequence[int]) -> dict[str, Any]:
    """Fit and report convergence instead of hiding ConvergenceWarning."""

    from sklearn.exceptions import ConvergenceWarning

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features, np.asarray(y, dtype=np.int64))
    elapsed = time.perf_counter() - start
    iterations = int(np.max(classifier.n_iter_))
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {
        "fit_seconds": round(elapsed, 3),
        "iterations": iterations,
        "converged": converged and iterations < classifier.max_iter,
    }


class TfidfLogisticModel:
    """A fitted pipeline whose probability columns follow ``labels``."""

    family = FAMILY

    def __init__(self, pipeline, labels: Sequence[str]) -> None:
        self.pipeline = pipeline
        self.labels = tuple(labels)
        classes = [int(c) for c in pipeline.named_steps["classifier"].classes_]
        if sorted(classes) != list(range(len(self.labels))):
            raise ValueError(
                f"classifier classes {classes} do not cover labels {self.labels}"
            )
        # classes_ holds label indices; this maps each column to its label index.
        self._columns = np.asarray(classes, dtype=np.int64)

    @classmethod
    def from_parts(cls, vectorizer, classifier, labels: Sequence[str]):
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
        return cls(pipeline, labels)

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        y: Sequence[int],
        *,
        vectorizer_fixed: Mapping[str, Any],
        vectorizer_params: Mapping[str, Any],
        classifier_fixed: Mapping[str, Any],
        classifier_params: Mapping[str, Any],
        labels: Sequence[str],
        seed: int,
    ) -> TfidfLogisticModel:
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("tfidf", build_vectorizer(vectorizer_fixed, vectorizer_params)),
                (
                    "classifier",
                    build_classifier(classifier_fixed, classifier_params, seed),
                ),
            ]
        )
        pipeline.fit(list(texts), np.asarray(y, dtype=np.int64))
        return cls(pipeline, labels)

    def align(self, raw: np.ndarray) -> np.ndarray:
        """Reorder ``predict_proba`` columns (``classes_`` order) to ``labels``."""

        aligned = np.zeros((raw.shape[0], len(self.labels)), dtype=np.float64)
        aligned[:, self._columns] = raw
        return renormalize(aligned)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.align(self.pipeline.predict_proba(list(texts)))

    @property
    def n_features(self) -> int:
        return int(self.pipeline.named_steps["classifier"].coef_.shape[1])

    def save(self, path: str | Path) -> dict[str, Any]:
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, destination, compress=0)
        meta = {
            "format": FORMAT,
            "family": FAMILY,
            "labels": list(self.labels),
            "class_index_order": self._columns.tolist(),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "n_features": self.n_features,
            "versions": library_versions(),
            "trust": "joblib/pickle executes code on load; load only this local, hash-verified file",
        }
        sidecar = destination.with_name(destination.name + SIDECAR_SUFFIX)
        sidecar.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return meta

    @classmethod
    def load(cls, path: str | Path, labels: Sequence[str] | None = None):
        import joblib

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"TF-IDF model not found: {source}")
        sidecar = source.with_name(source.name + SIDECAR_SUFFIX)
        if not sidecar.is_file():
            raise FileNotFoundError(f"TF-IDF model sidecar not found: {sidecar}")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(
                f"unsupported TF-IDF artifact format: {meta.get('format')}"
            )
        actual = file_sha256(source)
        if actual != meta["sha256"]:
            raise ValueError(
                f"refusing to unpickle {source}: sha256 {actual} != sidecar {meta['sha256']}"
            )
        current = library_versions()["scikit-learn"]
        if meta["versions"]["scikit-learn"] != current:
            raise RuntimeError(
                f"model was saved with scikit-learn {meta['versions']['scikit-learn']}, "
                f"running {current}; retrain instead of unpickling across versions"
            )
        if labels is not None and list(labels) != meta["labels"]:
            raise ValueError("label order differs from the saved model")
        return cls(joblib.load(source), meta["labels"])
