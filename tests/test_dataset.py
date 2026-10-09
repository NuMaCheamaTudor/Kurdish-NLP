from dataclasses import replace
from pathlib import Path

import pytest

from kurdish_nlp.langid.dataset import (
    DatasetRecord,
    DatasetValidationError,
    analyze_records,
    convert_jsonl_to_fasttext,
    find_document_split_leakage,
    find_exact_duplicate_texts,
    find_text_split_leakage,
    load_jsonl,
    validate_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "langid" / "synthetic.jsonl"


def _record(**updates: str) -> DatasetRecord:
    value = {
        "id": "record-1",
        "text": "valid example",
        "label": "en",
        "source": "test",
        "domain": "synthetic",
        "document_id": "doc-1",
        "license": "test-only",
        "split": "train",
    }
    value.update(updates)
    return DatasetRecord.from_mapping(value)


def test_synthetic_fixture_is_valid_and_has_all_target_labels() -> None:
    analysis = validate_jsonl(FIXTURE)
    assert analysis.record_count == 7
    assert set(analysis.label_distribution) == {
        "ckb",
        "kmr",
        "sdh",
        "ar",
        "fa",
        "tr",
        "en",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("text", "  "),
        ("source", ""),
        ("document_id", ""),
        ("split", "validation"),
        ("label", "und"),
        ("label", "ku"),
    ],
)
def test_invalid_dataset_records(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        _record(**{field: value})


def test_jsonl_loader_reports_all_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("\nnot-json\n{}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError) as captured:
        load_jsonl(path)
    assert len(captured.value.issues) == 3


def test_duplicate_and_text_leakage_detection() -> None:
    first = _record()
    duplicate = replace(
        first,
        id="record-2",
        split="dev",
        document_id="doc-2",
        text="  valid\nexample ",
    )
    records = [first, duplicate]
    assert find_exact_duplicate_texts(records) == (("record-1", "record-2"),)
    assert find_text_split_leakage(records) == (("record-1", "record-2"),)
    analysis = analyze_records(records)
    assert {issue.code for issue in analysis.issues} >= {
        "duplicate_text",
        "text_split_leakage",
    }


def test_document_id_overlap_across_splits() -> None:
    first = _record()
    second = replace(
        first,
        id="record-2",
        text="different example",
        split="test",
    )
    assert find_document_split_leakage([first, second]) == (("record-1", "record-2"),)


def test_fasttext_conversion_can_select_splits(tmp_path: Path) -> None:
    output = tmp_path / "train.txt"
    count = convert_jsonl_to_fasttext(FIXTURE, output, splits=("train",))
    lines = output.read_text(encoding="utf-8").splitlines()
    assert count == 3
    assert lines[0].startswith("__label__ckb ")
    assert lines[1].startswith("__label__kmr ")
    assert "\n" not in lines[0]
