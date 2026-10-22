"""CLI: python -m kurdish_nlp.ner.cli ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build, dump, verify, write_jsonl
from .dataset import audit, parse_bio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "audit",
        "build",
        "verify",
        "convert",
        "smoke",
        "train",
        "select",
        "evaluate",
        "benchmark",
    ):
        p = commands.add_parser(name)
        p.add_argument("--source", type=Path, default=Path("data/ner/raw/adyan.txt"))
        p.add_argument(
            "--data", type=Path, default=Path("data/ner/builds/sorani-ner-v2.0")
        )
        p.add_argument(
            "--lock", type=Path, default=Path("configs/ner/ner-v2.lock.json")
        )
        if name == "build":
            p.add_argument(
                "--policy", type=Path, default=Path("configs/ner/policy-v2.json")
            )
        if name == "audit":
            p.add_argument("--output", type=Path, required=True)
        if name in {"convert", "smoke", "train", "select", "evaluate", "benchmark"}:
            p.add_argument("--output", type=Path, required=True)
        if name in {"smoke", "train", "benchmark"}:
            p.add_argument(
                "--encoder",
                type=Path,
                required=True,
                help="Local pinned XLM-R snapshot; no implicit download",
            )
            p.add_argument(
                "--config", type=Path, default=Path("configs/ner/xlmr-v2.cfg")
            )
            p.add_argument("--gpu-id", type=int, default=-1)
        if name in {"select", "evaluate", "benchmark"}:
            p.add_argument("--model", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build(args.source, args.policy, args.data, args.lock)
        print(
            json.dumps(
                {
                    "raw": {
                        k: report["raw"][k]
                        for k in (
                            "segments",
                            "parsed_token_rows",
                            "invalid_bio_transitions",
                        )
                    },
                    "cleaning": report["cleaning"],
                    "splits": report["splits"],
                },
                indent=2,
            )
        )
    elif args.command == "audit":
        from collections import Counter

        from .grouping import group_records

        records = parse_bio(args.source)
        report, issues = audit(records)
        groups, edges = group_records(records)
        report["grouping"] = {
            "method": "normalized token trigrams, exact Jaccard threshold join",
            "threshold": 0.85,
            "minimum_tokens": 8,
            "groups_including_all_records": len(set(groups.values())),
            "edge_counts": dict(Counter(e["kind"] for e in edges)),
        }
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / "audit.json", report)
        write_jsonl(args.output / "issues.jsonl", issues)
        write_jsonl(args.output / "group-edges.jsonl", edges)
    elif args.command == "verify":
        verify(args.data, args.lock, args.source)
        print("Frozen checksums, BIO and split isolation verified")
    else:
        from . import training

        getattr(training, args.command)(args)


if __name__ == "__main__":
    main()
