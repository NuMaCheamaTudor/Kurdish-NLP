"""Offline data correctness / leakage regression tests; no network or model weights."""

import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from kurdish_nlp.ner.build import build, read_jsonl, verify
from kurdish_nlp.ner.dataset import audit, bio_errors, parse_bio, spans
from kurdish_nlp.ner.evaluation import metrics
from kurdish_nlp.ner.grouping import group_records, jaccard_pairs, normalized_tokens

POLICY = Path(__file__).resolve().parents[1] / "configs/ner/policy-v2.json"


def source(tmp_path, text):
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_strict_parse_preserves_malformed_records_and_eof(tmp_path):
    records = parse_bio(
        source(
            tmp_path, "ناسا B-ORG\nپلانی O\n\nwrong three columns\nbad\n\nهەولێر I-LOC"
        )
    )
    assert len(records) == 3
    assert records[1]["line_start"] == 4
    assert len(records[1]["malformed"]) == 2
    assert records[2]["row_lines"] == [7]
    report, _ = audit(records)
    assert report["malformed_rows"] == 2
    assert report["invalid_bio_transitions"] == 1


def test_spans_never_silently_fix_iob():
    assert spans(["B-PER", "I-PER", "B-PER", "O", "B-LOC"]) == [
        (0, 2, "PER"),
        (2, 3, "PER"),
        (4, 5, "LOC"),
    ]
    for tags in (["I-PER"], ["B-LOC", "I-PER"], ["o"], ["B-per"], ["B-"]):
        assert bio_errors(tags)
        with pytest.raises(ValueError):
            spans(tags)


def test_exact_threshold_join_matches_bruteforce():
    rng = random.Random(1729)
    sets = [frozenset(rng.sample(range(30), rng.randint(1, 20))) for _ in range(90)]
    sets += [
        frozenset(range(20)),
        frozenset(range(18)),
        frozenset(),
        frozenset(range(20)),
    ]
    for threshold in (0.5, 0.85, 1.0):
        expected = {
            tuple(sorted((a, b)))
            for a, b in itertools.combinations(range(len(sets)), 2)
            if sets[a]
            and sets[b]
            and len(sets[a] & sets[b]) / len(sets[a] | sets[b]) >= threshold
        }
        actual = {tuple(sorted((a, b))) for a, b, _ in jaccard_pairs(sets, threshold)}
        assert actual == expected


def test_normalization_does_not_rewrite_source(tmp_path):
    path = source(tmp_path, "كورد B-LOC\n\nکورد B-LOC")
    records = parse_bio(path)
    groups, _ = group_records(records)
    assert len(set(groups.values())) == 1
    assert records[0]["tokens"] == ["كورد"]
    assert normalized_tokens(["كورد", "ى", "ي", "ـ"]) == ("کورد", "ی", "ی")


def test_clean_quarantine_dedup_provenance_and_determinism(tmp_path):
    path = source(
        tmp_path,
        "ناسا B-ORG\n\nناسا B-ORG\n\nهەولێر B-LOC\n\nهەولێر O\n\nکورد I-MISC\n\nبەهار B-SEASON\n\nپارتی B-POLITICAL_PARY\n\nسەربازی o\n\ninvalid row fields",
    )
    original = path.read_bytes()
    reports = []
    for name in ("a", "b"):
        out, lock = tmp_path / name, tmp_path / f"{name}.lock.json"
        reports.append(build(path, POLICY, out, lock))
        verify(out, lock, path)
    assert json.loads((tmp_path / "a.lock.json").read_text()) == json.loads(
        (tmp_path / "b.lock.json").read_text()
    )
    assert path.read_bytes() == original
    assert reports[0]["cleaning"]["status_counts"] == {
        "retained": 4,
        "deduplicated": 1,
        "quarantined": 4,
    }
    kept = read_jsonl(tmp_path / "a/cleaned.jsonl")
    assert kept[0]["source_records"] == ["seg-000001", "seg-000002"]
    assert {e[2] for r in kept for e in spans(r["tags"])} == {
        "ORG",
        "SEASON",
        "POLITICAL_PARTY",
    }
    assert any(r["tags"] == ["O"] for r in kept)
    assert len(read_jsonl(tmp_path / "a/provenance.jsonl")) == 9
    with pytest.raises(FileExistsError):
        build(path, POLICY, tmp_path / "a", tmp_path / "a.lock.json")
    (tmp_path / "a/train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path / "a", tmp_path / "a.lock.json")


def test_near_grouping_retains_transitive_bridge(tmp_path):
    words = [f"w{i}" for i in range(60)]
    variants = [words, words[:-1] + ["a"], words[:-2] + ["b", "a"]]
    path = source(
        tmp_path, "\n\n".join("\n".join(f"{w} O" for w in ws) for ws in variants)
    )
    records = parse_bio(path)
    groups, edges = group_records(records)
    assert len(set(groups.values())) == 1
    assert any(e["kind"] == "near" for e in edges)


def test_strict_entity_metrics_and_error_pairing():
    gold = [{(0, 3, "PER"), (4, 8, "LOC"), (9, 14, "ORG")}]
    predicted = [{(0, 3, "PER"), (4, 8, "ORG"), (10, 14, "ORG"), (16, 18, "PER")}]
    result = metrics(gold, predicted, ["PER", "LOC", "ORG", "SEASON"])
    assert result["micro"]["precision"] == 1 / 4
    assert result["micro"]["recall"] == 1 / 3
    assert result["micro"]["f1"] == 2 / 7
    assert result["per_label"]["SEASON"]["f1"] is None
    assert result["span_errors"]["type_only"] == 1
    assert result["span_errors"]["boundary_only"] == 1
    assert result["span_errors"]["unmatched_predictions"] == 1


def test_spacy_conversion_matches_inference_tokenization(tmp_path):
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import DocBin

    from kurdish_nlp.ner.training import convert_records, gold_doc

    record = {
        "id": "test",
        "group": "g",
        "tokens": ["😀", "هەولێر،", "ناسا"],
        "tags": ["O", "B-LOC", "B-ORG"],
    }
    nlp = spacy.blank("xx")
    doc = gold_doc(nlp, record)
    assert [t.text for t in doc] == [t.text for t in nlp.make_doc(doc.text)]
    assert [(e.text, e.start_char, e.end_char) for e in doc.ents] == [
        ("هەولێر،", 2, 9),
        ("ناسا", 10, 14),
    ]
    result = convert_records([record], tmp_path / "test.spacy", nlp)
    loaded = list(DocBin().from_disk(tmp_path / "test.spacy").get_docs(nlp.vocab))
    assert loaded[0].text == doc.text
    assert result["entities"] == 2
    from kurdish_nlp.ner.contract import entities_payload

    payload = entities_payload(doc, "test-annotations", "fixture", ["LOC", "ORG"])
    assert payload["entities"][0] == {
        "text": "هەولێر،",
        "label": "LOC",
        "start": 2,
        "end": 9,
    }
    with pytest.raises(ValueError, match="inventory"):
        entities_payload(doc, "test-annotations", "fixture", ["ORG"])


def test_smoke_cannot_be_selected(tmp_path):
    from kurdish_nlp.ner.training import select

    path = source(tmp_path, "ناسا B-ORG")
    data, lock = tmp_path / "data", tmp_path / "lock.json"
    build(path, POLICY, data, lock)
    run = tmp_path / "run"
    model = run / "checkpoints/model-best"
    model.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"kind": "smoke", "status": "smoke_validated"})
    )
    args = SimpleNamespace(
        data=data,
        lock=lock,
        source=path,
        model=model,
        output=tmp_path / "selection.json",
    )
    with pytest.raises(ValueError, match="completed full run"):
        select(args)
