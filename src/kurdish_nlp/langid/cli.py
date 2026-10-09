"""Command-line tooling for canonical language-ID datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kurdish_nlp.langid.acquisition.importers import tatoeba, ud, wikimedia
from kurdish_nlp.langid.acquisition.importers.parme import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_DIR,
    plan_acquisition,
)
from kurdish_nlp.langid.acquisition.importers.parme import (
    acquire as acquire_parme,
)
from kurdish_nlp.langid.acquisition.importers.parme import (
    archive_path as parme_archive_path,
)
from kurdish_nlp.langid.acquisition.importers.parme import (
    build as build_parme,
)
from kurdish_nlp.langid.acquisition.manifests import load_manifest
from kurdish_nlp.langid.acquisition.policy import evaluate
from kurdish_nlp.langid.acquisition.schemas import DatasetRole, PolicyProfile
from kurdish_nlp.langid.dataset import (
    DatasetValidationError,
    analyze_records,
    convert_jsonl_to_fasttext,
    load_jsonl,
    validate_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kurdish-langid-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate canonical JSONL")
    validate.add_argument("input")

    summarize = subparsers.add_parser("summarize", help="summarize canonical JSONL")
    summarize.add_argument("input")

    convert = subparsers.add_parser(
        "convert-fasttext",
        help="validate and convert canonical JSONL to fastText format",
    )
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument(
        "--split",
        action="append",
        choices=("train", "dev", "test"),
        dest="splits",
        help="include one split; repeat to include multiple (default: all)",
    )

    manifest = subparsers.add_parser(
        "manifest", help="inspect offline source manifests"
    )
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_validate = manifest_commands.add_parser(
        "validate", help="validate a source manifest"
    )
    manifest_validate.add_argument("input")
    manifest_show = manifest_commands.add_parser(
        "show", help="print a normalized source manifest"
    )
    manifest_show.add_argument("input")
    policy_check = manifest_commands.add_parser(
        "policy-check", help="evaluate a source for a dataset role"
    )
    policy_check.add_argument("input")
    policy_check.add_argument(
        "--role", required=True, choices=tuple(role.value for role in DatasetRole)
    )
    policy_check.add_argument(
        "--profile",
        default="commercial",
        choices=tuple(profile.value for profile in PolicyProfile),
    )

    acquire = subparsers.add_parser(
        "acquire", help="explicitly acquire a pinned source archive"
    )
    acquire.add_argument("source", choices=("parme", "tatoeba", "wikimedia", "ud"))
    acquire.add_argument("--manifest")
    acquire.add_argument("--raw-dir")
    acquire.add_argument(
        "--profile", choices=("commercial", "research"), default="commercial"
    )
    acquire.add_argument("--dry-run", action="store_true")
    acquire.add_argument("--languages", nargs="+")
    acquire.add_argument("--max-pages", type=int, default=100)
    acquire.add_argument("--seed", type=int, default=42)
    acquire.add_argument("--contact")
    acquire.add_argument("--request-interval", type=float, default=1.1)

    import_source = subparsers.add_parser(
        "import", help="import a local, verified source archive"
    )
    import_source.add_argument(
        "source", choices=("parme", "tatoeba", "wikimedia", "ud")
    )
    import_source.add_argument("--manifest")
    import_source.add_argument("--raw-dir")
    import_source.add_argument(
        "--archive", help="explicit local archive override, still checksum-verified"
    )
    import_source.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    import_source.add_argument(
        "--profile", choices=("commercial", "research"), default="commercial"
    )
    import_source.add_argument("--seed", type=int, default=42)
    import_source.add_argument("--train-ratio", type=float, default=0.8)
    import_source.add_argument("--dev-ratio", type=float, default=0.1)
    import_source.add_argument("--test-ratio", type=float, default=0.1)
    import_source.add_argument(
        "--languages", nargs="+", help="Tatoeba canonical language labels"
    )
    import_source.add_argument("--max-per-language", type=int)
    import_source.add_argument(
        "--cross-label-duplicates", choices=("report", "exclude"), default="report"
    )
    import_source.add_argument(
        "--segment-mode",
        choices=("sentence", "paragraph", "bounded"),
        default="bounded",
    )
    import_source.add_argument("--max-tokens", type=int, default=80)
    import_source.add_argument("--max-segments-per-page", type=int, default=30)
    import_source.add_argument("--max-segments-per-language", type=int, default=50000)

    build = subparsers.add_parser(
        "build", help="build a versioned, reproducible dataset release"
    )
    build.add_argument(
        "dataset", help="dataset name; reads configs/langid/<dataset>.json by default"
    )
    build.add_argument("--config", help="explicit build configuration path")
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="verify sources, licenses, receipts, hashes and benchmarks only",
    )
    build.add_argument("--output-dir", help="override the configured output directory")
    build.add_argument("--work-dir", help="override the configured work directory")
    build.add_argument("--keep-work", action="store_true")
    build.add_argument(
        "--write-lock",
        action="store_true",
        help="(re)write the release lock instead of comparing against it",
    )
    build.add_argument("--workers", type=int, help="provenance validation processes")
    build.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            analysis = validate_jsonl(args.input)
            print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "summarize":
            analysis = analyze_records(load_jsonl(args.input))
            print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
            return 1 if analysis.issues else 0
        if args.command == "convert-fasttext":
            count = convert_jsonl_to_fasttext(
                args.input,
                args.output,
                splits=args.splits,
            )
            print(f"wrote {count} records to {args.output}")
            return 0
        if args.command == "manifest":
            source_manifest = load_manifest(args.input)
            if args.manifest_command == "validate":
                print(
                    f"valid source manifest: {source_manifest.source_id}@{source_manifest.manifest_version}"
                )
                return 0
            if args.manifest_command == "show":
                print(
                    json.dumps(source_manifest.to_dict(), ensure_ascii=False, indent=2)
                )
                return 0
            if args.manifest_command == "policy-check":
                decision = evaluate(
                    source_manifest, role=args.role, profile=args.profile
                )
                print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
                return 0 if decision.allowed else 1
        if args.command == "acquire":
            if args.source == "ud":
                if args.manifest:
                    raise ValueError(
                        "UD uses two reviewed manifests; select --languages"
                    )
                raw_dir = Path(args.raw_dir or ud.DEFAULT_RAW_DIR)
                if args.dry_run:
                    plan = ud.plan_acquisition(
                        args.languages, raw_dir=raw_dir, profile=args.profile
                    )
                    print(json.dumps(plan, ensure_ascii=False, indent=2))
                    return (
                        0
                        if all(
                            item["policy"]["allowed"]
                            for item in plan["sources"].values()
                        )
                        else 1
                    )
                print(
                    json.dumps(
                        ud.acquire(
                            args.languages, raw_dir=raw_dir, profile=args.profile
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            if args.source == "wikimedia":
                if args.manifest:
                    raise ValueError(
                        "Wikimedia uses six reviewed manifests; select --languages"
                    )
                raw_dir = Path(args.raw_dir or wikimedia.DEFAULT_RAW_DIR)
                if args.dry_run:
                    plan = wikimedia.plan_acquisition(
                        args.languages,
                        raw_dir=raw_dir,
                        max_pages=args.max_pages,
                        profile=args.profile,
                    )
                    print(json.dumps(plan, ensure_ascii=False, indent=2))
                    return (
                        0
                        if all(
                            item["policy"]["allowed"]
                            for item in plan["sources"].values()
                        )
                        else 1
                    )
                receipts = wikimedia.acquire(
                    args.languages,
                    raw_dir=raw_dir,
                    max_pages=args.max_pages,
                    seed=args.seed,
                    contact=args.contact or wikimedia.CONTACT_DEFAULT,
                    interval=args.request_interval,
                    profile=args.profile,
                )
                print(
                    json.dumps(
                        {
                            label: {
                                "snapshot_id": receipt["snapshot_id"],
                                "pages": len(receipt["pages"]),
                                "receipt": str(
                                    raw_dir
                                    / wikimedia.PROJECTS[label][0]
                                    / "receipt.json"
                                ),
                            }
                            for label, receipt in receipts.items()
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            default_manifest = (
                DEFAULT_MANIFEST if args.source == "parme" else tatoeba.DEFAULT_MANIFEST
            )
            default_raw_dir = (
                DEFAULT_RAW_DIR if args.source == "parme" else tatoeba.DEFAULT_RAW_DIR
            )
            source_manifest = load_manifest(args.manifest or default_manifest)
            raw_dir = Path(args.raw_dir or default_raw_dir)
            if args.dry_run:
                plan_fn = (
                    plan_acquisition
                    if args.source == "parme"
                    else tatoeba.plan_acquisition
                )
                plan = plan_fn(source_manifest, raw_dir=raw_dir, profile=args.profile)
                print(json.dumps(plan, ensure_ascii=False, indent=2))
                return 0 if plan["policy"]["allowed"] else 1
            if args.source == "tatoeba":
                receipt, reused = tatoeba.acquire(
                    source_manifest, raw_dir=raw_dir, profile=args.profile
                )
                print(
                    json.dumps(
                        {
                            "receipt": str(raw_dir / tatoeba.RECEIPT_NAME),
                            "snapshot_id": receipt["snapshot_id"],
                            "reused": reused,
                        },
                        indent=2,
                    )
                )
                return 0
            path, reused = acquire_parme(
                source_manifest, raw_dir=raw_dir, profile=args.profile
            )
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "reused": reused,
                        "sha256": source_manifest.checksum_sha256,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "import":
            if args.source == "ud":
                if args.manifest or args.archive:
                    raise ValueError(
                        "UD import uses reviewed manifests and verified cached archives"
                    )
                output_dir = Path(args.output_dir)
                if output_dir == DEFAULT_OUTPUT_DIR:
                    output_dir = ud.DEFAULT_OUTPUT_DIR
                report = ud.build(
                    args.languages,
                    raw_dir=Path(args.raw_dir or ud.DEFAULT_RAW_DIR),
                    output_dir=output_dir,
                    profile=args.profile,
                )
                print(
                    f"UD external benchmark: {report['audit']['record_count']} records, {report['audit']['strict_external_count']} strict; audit: {report['outputs']['audit']}"
                )
                return 0
            if args.source == "wikimedia":
                if args.manifest or args.archive:
                    raise ValueError(
                        "Wikimedia import uses six reviewed manifests and verified local receipts"
                    )
                report = wikimedia.build(
                    args.languages,
                    raw_dir=Path(args.raw_dir or wikimedia.DEFAULT_RAW_DIR),
                    output_dir=Path(args.output_dir),
                    profile=args.profile,
                    seed=args.seed,
                    ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
                    segment_mode=args.segment_mode,
                    max_tokens=args.max_tokens,
                    max_segments_per_page=args.max_segments_per_page,
                    max_segments_per_language=args.max_segments_per_language,
                )
                print(
                    f"Wikimedia import complete: {report['segments_accepted']} accepted, "
                    f"{report['segments_rejected']} rejected"
                )
                print(f"Audit: {report['outputs']['audit_text']}")
                return 0
            default_manifest = (
                DEFAULT_MANIFEST if args.source == "parme" else tatoeba.DEFAULT_MANIFEST
            )
            default_raw_dir = (
                DEFAULT_RAW_DIR if args.source == "parme" else tatoeba.DEFAULT_RAW_DIR
            )
            source_manifest = load_manifest(args.manifest or default_manifest)
            if args.source == "tatoeba":
                if args.archive:
                    raise ValueError(
                        "Tatoeba uses three verified archives; use --raw-dir, not --archive"
                    )
                report = tatoeba.build(
                    source_manifest,
                    raw_dir=Path(args.raw_dir or default_raw_dir),
                    output_dir=Path(args.output_dir),
                    languages=args.languages or sorted(tatoeba.TARGETS),
                    profile=args.profile,
                    max_per_language=args.max_per_language,
                    seed=args.seed,
                    ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
                    cross_label_duplicates=args.cross_label_duplicates,
                )
                print(
                    f"Tatoeba import complete: {report['accepted_records']} accepted, {report['rejected_records']} rejected"
                )
                print(f"Audit: {report['outputs']['audit_text']}")
                return 0
            archive = (
                Path(args.archive)
                if args.archive
                else parme_archive_path(
                    source_manifest, Path(args.raw_dir or default_raw_dir)
                )
            )
            report = build_parme(
                source_manifest,
                archive,
                output_dir=Path(args.output_dir),
                profile=args.profile,
                seed=args.seed,
                ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
            )
            print(
                f"PARME import complete: {report['accepted_records']} accepted, {report['rejected_records']} rejected"
            )
            print(f"Audit: {report['outputs']['audit_text']}")
            return 0
        if args.command == "build":
            from kurdish_nlp.langid.build import (
                BuildOptions,
                build_dataset,
                load_config,
            )

            config_path = Path(args.config or f"configs/langid/{args.dataset}.json")
            if not config_path.is_file():
                raise FileNotFoundError(f"build configuration not found: {config_path}")
            if load_config(config_path).dataset_name != args.dataset:
                raise ValueError(
                    f"{config_path} does not configure dataset {args.dataset!r}"
                )
            result = build_dataset(
                config_path,
                BuildOptions(
                    dry_run=args.dry_run,
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    keep_work=args.keep_work,
                    write_lock=args.write_lock,
                    workers=args.workers,
                    progress=None
                    if args.quiet
                    else lambda message: print(message, file=sys.stderr, flush=True),
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except (DatasetValidationError, OSError, ValueError, TypeError) as error:
        print(str(error))
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
