"""Verified loading of the frozen dataset, its metadata and the UD benchmark.

Nothing here rebuilds or rewrites a dataset. Every input file is hashed and
compared with both the experiment configuration and the dataset lock before a
single record is used, and verification failures raise instead of degrading.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kurdish_nlp.langid.build.config import canonical_json
from kurdish_nlp.langid.build.grouping import stable_key
from kurdish_nlp.langid.build.sources import file_sha256
from kurdish_nlp.langid.build.splitting import SplitGroup
from kurdish_nlp.langid.build.textkeys import script_class
from kurdish_nlp.langid.dataset import (
    DatasetRecord,
    analyze_records,
    load_jsonl,
    write_fasttext_examples,
)
from kurdish_nlp.langid.experiments.config import (
    DEV_ROLES,
    ExperimentConfig,
)
from kurdish_nlp.langid.normalization import normalize_text
from kurdish_nlp.langid.schemas import LanguageCode

SPLITS = ("train", "dev", "test")
UNKNOWN = "unknown"


class DatasetVerificationError(RuntimeError):
    """Raised when the frozen dataset or benchmark is missing or altered."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(
            "dataset verification failed:\n" + "\n".join(f"- {p}" for p in problems)
        )


@dataclass(frozen=True, slots=True)
class Example:
    """One evaluation unit with the metadata used for slicing."""

    id: str
    text: str
    label: LanguageCode
    role: str
    group: str
    source_family: str
    domain: str
    tokens: int
    length_bucket: str
    script: str
    contributor_tier: str = UNKNOWN
    variety: str = UNKNOWN
    review_flags: tuple[str, ...] = ()
    giant_group: bool = False


@dataclass(slots=True)
class ExampleSet:
    """An ordered collection of examples sharing one protocol role."""

    name: str
    examples: list[Example] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def texts(self) -> list[str]:
        return [example.text for example in self.examples]

    @property
    def ids(self) -> list[str]:
        return [example.id for example in self.examples]

    def label_indices(self, config: ExperimentConfig) -> list[int]:
        index = config.label_index()
        return [index[example.label.value] for example in self.examples]

    def subset(self, name: str, keep: Iterable[bool]) -> ExampleSet:
        return ExampleSet(
            name, [e for e, flag in zip(self.examples, keep, strict=True) if flag]
        )


@dataclass(slots=True)
class LoadedData:
    train: ExampleSet
    dev_tune: ExampleSet
    dev_calib: ExampleSet
    test: ExampleSet
    calibration_folds: list[str]
    role_report: dict[str, Any]
    verification: dict[str, Any]
    contributor_report: dict[str, Any]

    def role(self, name: str) -> ExampleSet:
        sets = {
            "train": self.train,
            "dev_tune": self.dev_tune,
            "dev_calib": self.dev_calib,
            "test": self.test,
        }
        return sets[name]


@dataclass(slots=True)
class ExternalData:
    strict: ExampleSet
    exploratory: ExampleSet
    report: dict[str, Any]


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_frozen_files(config: ExperimentConfig) -> dict[str, Any]:
    """Compare every frozen input with the experiment config and the lock."""

    spec = config.dataset
    problems: list[str] = []
    if not spec.build_dir.is_dir():
        raise DatasetVerificationError(
            [
                (
                    f"dataset build directory is missing: {spec.build_dir}. "
                    "Dataset V1 is not rebuilt by experiments; restore the frozen build."
                )
            ]
        )
    if not spec.lock_file.is_file():
        raise DatasetVerificationError([f"dataset lock is missing: {spec.lock_file}"])
    lock = json.loads(spec.lock_file.read_text(encoding="utf-8"))
    manifest_path = spec.build_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetVerificationError(
            [f"dataset manifest is missing: {manifest_path}"]
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if lock.get("dataset") != spec.name or lock.get("dataset_version") != spec.version:
        problems.append("lock dataset name/version differs from the experiment config")
    if lock.get("config_sha256") != spec.config_sha256:
        problems.append("lock config_sha256 differs from the experiment config")
    if manifest.get("dataset") != spec.name or (
        manifest.get("dataset_version") != spec.version
    ):
        problems.append(
            "manifest dataset name/version differs from the experiment config"
        )
    if manifest.get("config", {}).get("sha256") != spec.config_sha256:
        problems.append("manifest build-config hash differs from the experiment config")
    if spec.build_config.is_file():
        build_config = json.loads(spec.build_config.read_text(encoding="utf-8"))
        actual = hashlib.sha256(canonical_json(build_config).encode()).hexdigest()
        if actual != spec.config_sha256:
            problems.append(
                f"build configuration {spec.build_config} canonical hash is {actual}"
            )
    else:
        problems.append(f"build configuration is missing: {spec.build_config}")

    files: dict[str, str] = {}
    for name, expected in sorted(spec.expected_sha256.items()):
        path = spec.build_dir / name
        if not path.is_file():
            problems.append(f"frozen file is missing: {path}")
            continue
        actual = file_sha256(path)
        files[name] = actual
        if actual != expected:
            problems.append(f"{name}: sha256 {actual} != configured {expected}")
        locked = lock.get("outputs", {}).get(name)
        if locked != expected:
            problems.append(f"{name}: configured hash differs from lock ({locked})")
    if problems:
        raise DatasetVerificationError(problems)
    return {
        "dataset": spec.name,
        "dataset_version": spec.version,
        "build_config_sha256": spec.config_sha256,
        "lock_file": str(spec.lock_file),
        "lock_file_sha256": file_sha256(spec.lock_file),
        "manifest_sha256": file_sha256(manifest_path),
        "files_sha256": files,
        "lock_counts": lock.get("counts", {}),
    }


def _load_split_records(config: ExperimentConfig) -> dict[str, list[DatasetRecord]]:
    records = {}
    for split in SPLITS:
        path = config.dataset.build_dir / f"{split}.jsonl"
        records[split] = load_jsonl(path)
    return records


def verify_records(
    config: ExperimentConfig,
    records: Mapping[str, Sequence[DatasetRecord]],
    lock_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Validate canonical records, label mapping, normalization and isolation."""

    problems: list[str] = []
    labels = set(config.label_values)
    for split, rows in records.items():
        expected = config.dataset.expected_counts.get(split)
        if len(rows) != expected:
            problems.append(f"{split}: {len(rows)} records, expected {expected}")
        wrong_split = [r.id for r in rows if r.split != split]
        if wrong_split:
            problems.append(f"{split}: {len(wrong_split)} records carry another split")
        unknown = {r.label.value for r in rows} - labels
        if unknown:
            problems.append(
                f"{split}: labels outside the experiment: {sorted(unknown)}"
            )
        not_normalized = sum(normalize_text(r.text) != r.text for r in rows)
        if not_normalized:
            problems.append(f"{split}: {not_normalized} texts are not normalized")
        benchmark_ids = [r.id for r in rows if r.id.startswith("ud:")]
        if benchmark_ids:
            problems.append(
                f"{split}: contains UD benchmark records {benchmark_ids[:3]}"
            )

    by_label_split: dict[str, Counter[str]] = defaultdict(Counter)
    for split, rows in records.items():
        for record in rows:
            by_label_split[record.label.value][split] += 1
    for label in labels:
        for split in SPLITS:
            locked = lock_counts.get(label, {}).get(split)
            if by_label_split[label][split] != locked:
                problems.append(
                    f"{label}/{split}: {by_label_split[label][split]} records, lock {locked}"
                )

    combined = [record for split in SPLITS for record in records[split]]
    analysis = analyze_records(combined)
    if analysis.issues:
        codes = Counter(issue.code for issue in analysis.issues)
        problems.append(f"canonical validation issues: {dict(codes)}")
    groups = {split: {r.document_id for r in records[split]} for split in SPLITS}
    overlaps = {
        f"{a}|{b}": len(groups[a] & groups[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    if any(overlaps.values()):
        problems.append(f"document groups cross splits: {overlaps}")
    if problems:
        raise DatasetVerificationError(problems)
    return {
        "records": {split: len(rows) for split, rows in records.items()},
        "label_counts": {
            label: dict(sorted(by_label_split[label].items()))
            for label in config.label_values
        },
        "und_in_training_labels": False,
        "canonical_validation_issues": 0,
        "group_overlap_between_splits": overlaps,
        "groups_per_split": {split: len(groups[split]) for split in SPLITS},
        "all_texts_normalized": True,
    }


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Provenance:
    source_id: str
    contributor: str | None
    variety: str | None
    giant_group: bool


def _load_provenance(path: Path) -> dict[str, _Provenance]:
    """Stream the provenance sidecar and keep only fields used for slicing."""

    result: dict[str, _Provenance] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            build = next(
                (t for t in row["transformations"] if t["name"] == "dataset_build"),
                None,
            )
            giant = bool(
                build and build["parameters"].get("grouping", {}).get("giant_group")
            )
            contributor = row.get("contributor_id") or row.get("translator_id")
            result[row["canonical_record_id"]] = _Provenance(
                source_id=row["source_id"],
                contributor=str(contributor) if contributor else None,
                variety=row.get("language_variety"),
                giant_group=giant,
            )
    return result


def _load_review_flags(path: Path) -> dict[str, tuple[str, ...]]:
    flags: dict[str, tuple[str, ...]] = {}
    if not path.is_file():
        return flags
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            flags[row["record_id"]] = tuple(sorted(row.get("flags", ())))
    return flags


def pseudonymous_contributor(experiment_id: str, source: str, contributor: str) -> str:
    """Stable pseudonym; raw usernames are never written to reports."""

    digest = hashlib.sha256(
        f"{experiment_id}|{source}|{contributor}".encode()
    ).hexdigest()
    return f"c-{digest[:10]}"


def contributor_tiers(
    config: ExperimentConfig,
    train_records: Sequence[DatasetRecord],
    provenance: Mapping[str, _Provenance],
) -> tuple[dict[tuple[str, str, str], str], dict[str, Any]]:
    """Rank contributors within (label, source) by their TRAIN record counts.

    ``top`` is the single largest contributor, ``head`` the next ranks up to the
    configured head size, ``tail`` the rest; ``unseen`` (assigned later) marks a
    contributor with no training records. Ranks come from train only, so test
    composition never influences tier definitions.
    """

    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in train_records:
        prov = provenance[record.id]
        if prov.contributor:
            counts[(record.label.value, prov.source_id)][prov.contributor] += 1
    tiers: dict[tuple[str, str, str], str] = {}
    report: dict[str, Any] = {}
    top, head = config.evaluation.contributor_top, config.evaluation.contributor_head
    for (label, source), counter in sorted(counts.items()):
        ranked = sorted(
            counter.items(),
            key=lambda item: (-item[1], stable_key(config.seed, "contrib", item[0])),
        )
        total = sum(counter.values())
        for rank, (contributor, _) in enumerate(ranked, start=1):
            tier = "top" if rank <= top else "head" if rank <= head else "tail"
            tiers[(label, source, contributor)] = tier
        report[f"{label}|{source}"] = {
            "train_records_with_contributor": total,
            "distinct_contributors": len(ranked),
            "top_share": round(ranked[0][1] / total, 6),
            "head_share": round(sum(c for _, c in ranked[:head]) / total, 6),
            "top_pseudonym": pseudonymous_contributor(
                config.experiment_id, source, ranked[0][0]
            ),
        }
    return tiers, report


def _example(
    config: ExperimentConfig,
    record: DatasetRecord,
    role: str,
    provenance: Mapping[str, _Provenance],
    tiers: Mapping[tuple[str, str, str], str],
    review: Mapping[str, tuple[str, ...]],
) -> Example:
    prov = provenance.get(record.id)
    if prov is None:
        raise DatasetVerificationError([f"record {record.id} has no provenance row"])
    if prov.contributor:
        tier = tiers.get(
            (record.label.value, prov.source_id, prov.contributor), "unseen"
        )
    else:
        tier = "none"
    tokens = len(record.text.split())
    return Example(
        id=record.id,
        text=record.text,
        label=record.label,
        role=role,
        group=record.document_id,
        source_family=prov.source_id,
        domain=record.domain,
        tokens=tokens,
        length_bucket=config.evaluation.length_bucket(tokens),
        script=script_class(record.text),
        contributor_tier=tier,
        variety=prov.variety or "none",
        review_flags=review.get(record.id, ()),
        giant_group=prov.giant_group,
    )


# --------------------------------------------------------------------------
# Group-disjoint development roles
# --------------------------------------------------------------------------


def assign_group_roles(
    groups: Sequence[SplitGroup],
    ratios: Mapping[str, float],
    *,
    seed: int,
    salt: str,
) -> dict[str, str]:
    """Greedy group-level stratified assignment to arbitrary named roles.

    Same algorithm as ``build.splitting.assign_splits`` (largest groups first,
    ties by a seeded stable hash, minimum weighted fill) but for any role names;
    the frozen builder module itself is not modified. Every group lands in exactly
    one role, so roles are group-disjoint by construction.
    """

    roles = list(ratios)
    if len({g.group_id for g in groups}) != len(groups):
        raise ValueError("group IDs must be unique")
    totals: Counter[tuple[str, ...]] = Counter()
    for group in groups:
        for stratum, count in group.strata:
            totals[stratum] += count
    targets = {
        role: {stratum: ratios[role] * total for stratum, total in totals.items()}
        for role in roles
    }
    assigned: dict[str, Counter[tuple[str, ...]]] = {role: Counter() for role in roles}
    result: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda g: (-g.size, stable_key(seed, salt, g.group_id), g.group_id),
    )
    for group in ordered:
        strata = sorted(group.strata)

        def cost(
            role: str, strata: list[tuple[tuple[str, ...], int]] = strata
        ) -> float:
            return sum(
                count * (assigned[role][stratum] + count) / targets[role][stratum]
                for stratum, count in strata
            )

        best = min(roles, key=lambda role: (cost(role), roles.index(role)))
        result[group.group_id] = best
        for stratum, count in group.strata:
            assigned[best][stratum] += count
    return result


def _split_groups(
    examples: Sequence[Example], strata_fields: Sequence[str]
) -> list[SplitGroup]:
    by_group: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for example in examples:
        values = []
        for name in strata_fields:
            values.append(
                example.label.value if name == "label" else example.source_family
            )
        by_group[example.group][tuple(values)] += 1
    return [
        SplitGroup(group_id, tuple(sorted(counter.items())))
        for group_id, counter in sorted(by_group.items())
    ]


def partition_development(
    config: ExperimentConfig, dev: Sequence[Example]
) -> tuple[list[Example], list[Example], list[str], dict[str, Any]]:
    """Split dev into group-disjoint dev_tune / dev_calib and calibration folds."""

    protocol = config.development
    groups = _split_groups(dev, protocol.strata)
    assignment = assign_group_roles(
        groups, protocol.roles, seed=config.seed, salt="dev-role"
    )
    tune = [
        replace_role(e, "dev_tune") for e in dev if assignment[e.group] == "dev_tune"
    ]
    calib = [
        replace_role(e, "dev_calib") for e in dev if assignment[e.group] == "dev_calib"
    ]
    fold_names = [f"fold_{i}" for i in range(protocol.calibration_crossfit_folds)]
    fold_assignment = assign_group_roles(
        _split_groups(calib, protocol.strata),
        {name: 1.0 / len(fold_names) for name in fold_names},
        seed=config.seed,
        salt="dev-calib-fold",
    )
    folds = [fold_assignment[e.group] for e in calib]

    tune_groups = {e.group for e in tune}
    calib_groups = {e.group for e in calib}
    overlap = tune_groups & calib_groups
    if overlap or len(tune) + len(calib) != len(dev):
        raise DatasetVerificationError(
            [f"development partition is not group-disjoint ({len(overlap)} shared)"]
        )
    report = {
        "method": protocol.method,
        "group_field": protocol.group_field,
        "strata": list(protocol.strata),
        "ratios": dict(protocol.roles),
        "records": {"dev_tune": len(tune), "dev_calib": len(calib)},
        "groups": {"dev_tune": len(tune_groups), "dev_calib": len(calib_groups)},
        "shared_groups": 0,
        "by_label": {
            role: dict(sorted(Counter(e.label.value for e in items).items()))
            for role, items in (("dev_tune", tune), ("dev_calib", calib))
        },
        "by_source": {
            role: dict(sorted(Counter(e.source_family for e in items).items()))
            for role, items in (("dev_tune", tune), ("dev_calib", calib))
        },
        "calibration_folds": dict(sorted(Counter(folds).items())),
        "assignment_sha256": hashlib.sha256(
            canonical_json(sorted(assignment.items())).encode()
        ).hexdigest(),
    }
    return tune, calib, folds, report


def replace_role(example: Example, role: str) -> Example:
    return replace(example, role=role)


# --------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------


def load_experiment_data(config: ExperimentConfig) -> LoadedData:
    """Verify Dataset V1 and return the four protocol roles with metadata."""

    files = verify_frozen_files(config)
    records = _load_split_records(config)
    checks = verify_records(config, records, files["lock_counts"])
    provenance = _load_provenance(config.dataset.build_dir / "provenance.jsonl")
    review = _load_review_flags(config.dataset.build_dir / "review_queue.jsonl")
    missing = [
        r.id for split in SPLITS for r in records[split] if r.id not in provenance
    ]
    if missing:
        raise DatasetVerificationError(
            [f"{len(missing)} records lack provenance, e.g. {missing[:3]}"]
        )
    tiers, contributor_report = contributor_tiers(config, records["train"], provenance)

    def build(split: str, role: str) -> list[Example]:
        return [
            _example(config, record, role, provenance, tiers, review)
            for record in records[split]
        ]

    train = build("train", "train")
    dev = build("dev", "dev")
    test = build("test", "test")
    tune, calib, folds, role_report = partition_development(config, dev)
    if set(role_report["records"]) != set(DEV_ROLES):
        raise AssertionError("unexpected development roles")
    verification = {**files, **checks}
    verification.pop("lock_counts", None)
    return LoadedData(
        train=ExampleSet("train", train),
        dev_tune=ExampleSet("dev_tune", tune),
        dev_calib=ExampleSet("dev_calib", calib),
        test=ExampleSet("test", test),
        calibration_folds=folds,
        role_report=role_report,
        verification=verification,
        contributor_report=contributor_report,
    )


def load_external_benchmark(
    config: ExperimentConfig, training_texts: Iterable[str] = ()
) -> ExternalData:
    """Load the strict and exploratory UD slices as two separate sets.

    The strict slice is exactly the records marked ``strict_external`` with no
    exclusion reason; the exploratory slice admits only the configured prior
    exposure reason. Counts must match the configuration. Any exact normalized
    overlap with Dataset V1 texts aborts evaluation.
    """

    spec = config.external
    problems = []
    for hashed in (spec.records, spec.manifest):
        if not hashed.path.is_file():
            raise DatasetVerificationError(
                [f"UD benchmark file missing: {hashed.path}"]
            )
        actual = file_sha256(hashed.path)
        if actual != hashed.sha256:
            problems.append(f"{hashed.path}: sha256 {actual} != {hashed.sha256}")
    if problems:
        raise DatasetVerificationError(problems)
    rows = [
        json.loads(line)
        for line in spec.records.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    strict_rows, exploratory_rows, excluded = [], [], Counter()
    for row in rows:
        reasons = tuple(row.get("exclusion_reasons", ()))
        if row.get("benchmark") != spec.benchmark_id:
            problems.append(f"{row.get('id')}: wrong benchmark id")
        if row.get("strict_external") is True and not reasons:
            strict_rows.append(row)
        elif (
            row.get("strict_external") is False
            and reasons
            and set(reasons) <= set(spec.exploratory.allowed_exclusion_reasons)
        ):
            exploratory_rows.append(row)
        else:
            excluded["|".join(reasons) or "unspecified"] += 1
    for name, chosen, slice_spec in (
        ("strict", strict_rows, spec.strict),
        ("exploratory", exploratory_rows, spec.exploratory),
    ):
        languages = {row["expected_language"] for row in chosen}
        if languages != {slice_spec.expected_language.value}:
            problems.append(f"{name}: unexpected languages {sorted(languages)}")
        if len(chosen) != slice_spec.expected_count:
            problems.append(
                f"{name}: {len(chosen)} records, expected {slice_spec.expected_count}"
            )
    strict_ids = {row["id"] for row in strict_rows}
    if strict_ids & {row["id"] for row in exploratory_rows}:
        problems.append("a record is in both the strict and exploratory slices")
    training = {normalize_text(text) for text in training_texts}
    overlap = [
        row["id"]
        for row in strict_rows + exploratory_rows
        if normalize_text(row["text"]) in training
    ]
    if overlap:
        problems.append(f"UD texts overlap Dataset V1 exactly: {overlap[:5]}")
    if problems:
        raise DatasetVerificationError(problems)

    def to_example(row: Mapping[str, Any], role: str) -> Example:
        text = normalize_text(row["text"])
        tokens = len(text.split())
        return Example(
            id=row["id"],
            text=text,
            label=LanguageCode(row["expected_language"]),
            role=role,
            group=row["id"],
            source_family=f"ud-{row['treebank']}",
            domain=row["slice"],
            tokens=tokens,
            length_bucket=config.evaluation.length_bucket(tokens),
            script=script_class(text),
            contributor_tier="none",
            variety=row["treebank"],
            review_flags=tuple(sorted(row.get("exclusion_reasons", ()))),
        )

    strict = ExampleSet(
        spec.strict.name, [to_example(r, spec.strict.name) for r in strict_rows]
    )
    exploratory = ExampleSet(
        spec.exploratory.name,
        [to_example(r, spec.exploratory.name) for r in exploratory_rows],
    )
    report = {
        "benchmark_id": spec.benchmark_id,
        "records_sha256": spec.records.sha256,
        "manifest_sha256": spec.manifest.sha256,
        "records_total": len(rows),
        "strict": {
            "name": spec.strict.name,
            "title": spec.strict.title,
            "records": len(strict),
            "scripts": dict(sorted(Counter(e.script for e in strict.examples).items())),
        },
        "exploratory": {
            "name": spec.exploratory.name,
            "title": spec.exploratory.title,
            "records": len(exploratory),
            "scripts": dict(
                sorted(Counter(e.script for e in exploratory.examples).items())
            ),
            "genres": dict(
                sorted(Counter(e.domain for e in exploratory.examples).items())
            ),
        },
        "excluded_from_both": dict(sorted(excluded.items())),
        "exact_overlap_with_dataset_v1": 0,
        "restrictions": list(spec.restrictions),
    }
    return ExternalData(strict=strict, exploratory=exploratory, report=report)


def write_fasttext_training_file(examples: ExampleSet, path: Path) -> dict[str, Any]:
    """Write the supervised fastText file through the shared formatter."""

    count = write_fasttext_examples(
        path, ((e.label, e.text) for e in examples.examples)
    )
    with path.open("r", encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    if lines != count or count != len(examples):
        raise RuntimeError(
            f"fastText conversion wrote {lines} lines for {len(examples)} examples"
        )
    return {"path": str(path), "lines": lines, "sha256": file_sha256(path)}
