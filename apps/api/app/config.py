"""Runtime settings, read once from the environment.

Every model path comes from here; nothing in a request can point the server at a
file. Values can also be placed in ``apps/api/.env`` (simple ``KEY=VALUE`` lines);
variables already set in the environment take precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent.parent

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "langid" / "experiments" / "baseline-v1"
DEFAULT_NER_MODEL = REPO_ROOT / "kurdish-ner-v1"
DEFAULT_GRAMMAR_MODEL = REPO_ROOT / "kurdish-pos-output (1)" / "model-best"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` lines without overriding the real environment."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _path(name: str, default: Path | None) -> Path | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _origins(name: str) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(item.strip() for item in value.split(",") if item.strip())
    if "*" in origins:
        raise ValueError(f"{name} must list explicit origins, not '*'")
    return origins


@dataclass(frozen=True, slots=True)
class Settings:
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    default_model: str = "tfidf_lr"
    ner_model_path: Path | None = DEFAULT_NER_MODEL
    grammar_model_path: Path | None = DEFAULT_GRAMMAR_MODEL
    max_text_chars: int = 5000
    max_body_bytes: int = 64 * 1024
    inference_concurrency: int = 4
    preload: bool = True
    cors_origins: tuple[str, ...] = field(default=DEFAULT_CORS_ORIGINS)

    @classmethod
    def from_env(cls) -> Settings:
        load_env_file(API_DIR / ".env")
        max_text_chars = _int("KNLP_MAX_TEXT_CHARS", 5000, 1, 100_000)
        return cls(
            artifacts_dir=_path("KNLP_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
            or DEFAULT_ARTIFACTS_DIR,
            default_model=os.environ.get("KNLP_DEFAULT_MODEL", "tfidf_lr").strip(),
            ner_model_path=_path("KNLP_NER_MODEL_PATH", DEFAULT_NER_MODEL),
            grammar_model_path=_path("KNLP_GRAMMAR_MODEL_PATH", DEFAULT_GRAMMAR_MODEL),
            max_text_chars=max_text_chars,
            # Room for the JSON envelope and up to 4 UTF-8 bytes per character.
            max_body_bytes=_int(
                "KNLP_MAX_BODY_BYTES",
                max(64 * 1024, max_text_chars * 4 + 4096),
                1024,
                4 * 1024 * 1024,
            ),
            inference_concurrency=_int("KNLP_INFERENCE_CONCURRENCY", 4, 1, 64),
            preload=_bool("KNLP_PRELOAD", True),
            cors_origins=_origins("KNLP_CORS_ORIGINS"),
        )
