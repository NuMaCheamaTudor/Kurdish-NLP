"""Shared offline fixture world for dataset-builder integration tests.

The world mirrors the repository layout under a temporary directory. Its source
exports are produced by the real PARME, Tatoeba and Wikimedia importers from
small synthetic acquisitions, so the builder is exercised on genuine export
formats, receipts and provenance sidecars. Every text is synthetic test data.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from kurdish_nlp.langid.acquisition.benchmark import ExternalBenchmarkRecord
from kurdish_nlp.langid.acquisition.importers import (
    parme,
    tatoeba,
    wikimedia,
    wikimedia_acquire,
)
from kurdish_nlp.langid.acquisition.manifests import load_manifest, save_manifest
from kurdish_nlp.langid.normalization import normalize_text

REPO = Path(__file__).parents[1]
SOURCES = REPO / "configs/langid/sources"

UD_TEXTS = {
    "ud:kmr_kurmanji:test:u1": (
        "kmr",
        "Ez ê sibê biçim bajêr û hevalên xwe bibînim.",
        False,
    ),
    "ud:kmr_kurmanji:test:u2": (
        "kmr",
        "Pirtûka min li ser maseya mezin a odeyê ye.",
        False,
    ),
    "ud:kmr_kurmanji:test:u3": (
        "kmr",
        "Zarok di baxçeyê dibistanê de bi kêfxweşî dilîzin.",
        False,
    ),
    "ud:kmr_kurmanji:test:u4": (
        "kmr",
        "Bavê min her roj bi trênê diçe kar û êvarê vedigere malê.",
        False,
    ),
    "ud:sdh_garrusi:test:u5": ("sdh", "Min nan xward û çim bo bazar.", True),
}

PARME_ROWS = [
    # (split, english, persian, southern kurdish, variety, county, translator)
    (
        "train",
        "I go to school every morning.",
        "من هر صبح به مدرسه می‌روم.",
        "مِن هەر سوو چم ئەڕا مەدرەسە.",
        "Kirmashani",
        "Kermanshah",
        "Vol_0",
    ),
    (
        "val",
        "I go to school every morning.",
        "من هر صبح به مدرسه می‌روم.",
        "هەر سوحی مەچم ئەڕا مەکتەو.",
        "Kalhori",
        "Eslamabad",
        "EK",
    ),
    (
        "test",
        "The weather is cold today.",
        "امروز هوا سرد است.",
        "ئیمڕوو هەوا سەردە.",
        "Pehley",
        "Ilam",
        "AG",
    ),
    (
        "train",
        "This is my book.",
        "این کتاب من است.",
        "ئەیە کتاوەگەی منە.",
        "Kirmashani",
        "Kermanshah",
        "Vol_0",
    ),
    (
        "train",
        "My brother works in the city.",
        "برادرم در شهر کار می‌کند.",
        "براگەم لە شار کار کەێد.",
        "Kirmashani",
        "Kermanshah",
        "Vol_0",
    ),
    (
        "val",
        "We ate bread and cheese.",
        "ما نان و پنیر خوردیم.",
        "ئێمە نان و پەنیر خواردیم.",
        "Badrei",
        "Badra",
        "Vol_3",
    ),
    (
        "test",
        "The children are playing outside.",
        "بچه‌ها بیرون بازی می‌کنند.",
        "منداڵەیل لە دەیشت کایە کەن.",
        "Garusi",
        "Bijar",
        "AG",
    ),
    (
        "train",
        "Where is the market?",
        "بازار کجاست؟",
        "بازاڕ لە کوورەسە؟",
        "Kirmashani",
        "Kermanshah",
        "MO",
    ),
    (
        "train",
        "My mother is cooking dinner.",
        "مادرم شام می‌پزد.",
        "دالگم شیو مەپەزێد.",
        "Pehley",
        "Ilam",
        "Vol_0",
    ),
    (
        "val",
        "The road to the village is long.",
        "راه روستا طولانی است.",
        "ڕێ دێ دوورە.",
        "Kalhori",
        "Eslamabad",
        "EK",
    ),
    (
        "test",
        "I will come tomorrow.",
        "من فردا می‌آیم.",
        "مِن سوو تێەم.",
        "Kirmashani",
        "Kermanshah",
        "Vol_0",
    ),
    (
        "train",
        "The river is full of water.",
        "رودخانه پر از آب است.",
        "ڕووخانە پڕ لە ئاوە.",
        "Garusi",
        "Bijar",
        "Vol_5",
    ),
]

ENGLISH_NOUNS = [
    "teacher",
    "farmer",
    "doctor",
    "painter",
    "baker",
    "sailor",
    "driver",
    "singer",
]
ENGLISH_PLACES = [
    "market",
    "harbour",
    "library",
    "station",
    "garden",
    "bridge",
    "temple",
    "museum",
]
TURKISH_NOUNS = [
    "öğretmen",
    "çiftçi",
    "doktor",
    "ressam",
    "fırıncı",
    "denizci",
    "şoför",
    "şarkıcı",
]
TURKISH_PLACES = [
    "pazara",
    "limana",
    "kütüphaneye",
    "istasyona",
    "bahçeye",
    "köprüye",
    "tapınağa",
    "müzeye",
]


def _tatoeba_rows() -> tuple[list[list[str]], list[list[str]]]:
    rows: list[list[str]] = []

    def add(sentence_id: int, code: str, text: str, user: str) -> None:
        rows.append(
            [
                str(sentence_id),
                code,
                text,
                user,
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
            ]
        )

    giant = [
        ("eng", "I love learning new languages.", "user_a"),
        ("tur", "Yeni diller öğrenmeyi seviyorum.", "user_b"),
        ("ara", "أحب تعلم لغات جديدة كثيرا.", "user_c"),
        ("pes", "من یادگیری زبان‌های جدید را دوست دارم.", "user_d"),
        ("ckb", "من حەزم لە فێربوونی زمانی نوێیە.", "user_e"),
        ("kmr", "Ez ji fêrbûna zimanên nû hez dikim.", "user_f"),
        ("sdh", "مِن حەز لە فێربوین زوانە تازەیل دیرم.", "user_g"),
        ("eng", "I like to learn new languages.", "user_a"),
        ("tur", "Yeni diller öğrenmeyi severim.", "user_b"),
        ("eng", "Learning languages is fun for me.", "user_h"),
    ]
    for offset, (code, text, user) in enumerate(giant, 1):
        add(offset, code, text, user)
    links = [[str(index), str(index + 1)] for index in range(1, len(giant))]
    for index in range(30):
        noun = ENGLISH_NOUNS[index % 8]
        place = ENGLISH_PLACES[(index * 3) % 8]
        user = "big_contributor" if index < 20 else f"user_en_{index}"
        add(
            100 + index,
            "eng",
            f"The {noun} number {index} walked slowly to the {place}.",
            user,
        )
        add(
            200 + index,
            "tur",
            f"{index} numaralı {TURKISH_NOUNS[index % 8]} yavaşça {TURKISH_PLACES[(index * 3) % 8]} yürüdü.",
            f"user_tr_{index % 7}",
        )
    add(130, "eng", "Hello there my friend.", "user_x")
    add(131, "eng", "hello there, my friend!", "user_y")
    add(132, "eng", "The weather is nice today.", "user_x")
    add(133, "eng", "OK", "user_x")
    arabic = [
        "ذهب الولد إلى المدرسة صباحا.",
        "الكتاب على الطاولة الكبيرة.",
        "نحن نحب السفر إلى البحر.",
        "السماء زرقاء والجو جميل.",
        "شربت القهوة مع أصدقائي.",
        "المدينة مزدحمة في المساء.",
        "قرأت قصة طويلة أمس.",
        "الحديقة مليئة بالزهور.",
        "يعمل أبي في المستشفى.",
        "تعلمت السباحة في الصيف.",
        "البيت قريب من النهر.",
        "اشترينا خبزا طازجا.",
    ]
    for index, text in enumerate(arabic):
        add(300 + index, "ara", text, f"user_ar_{index % 3}")
    persian = [
        "امروز به کتابخانه رفتم.",
        "خواهرم نقاشی را دوست دارد.",
        "ما در تابستان به سفر رفتیم.",
        "این شهر بسیار زیبا است.",
        "دوستم برای من نامه نوشت.",
        "باران دیشب خیلی شدید بود.",
    ]
    for index, text in enumerate(persian):
        add(400 + index, "pes", text, f"user_fa_{index % 2}")
    add(406, "pes", "القاهرة مدينة كبيرة جدا.", "user_fa_9")
    sorani = [
        "ئەمڕۆ کەشوهەوا زۆر خۆشە.",
        "من کتێبێکی نوێم کڕی.",
        "باوکم لە بازاڕ کار دەکات.",
        "ئێمە بۆ گەشت دەچین.",
        "خوشکەکەم مامۆستایە.",
        "شار زۆر جەنجاڵە.",
        "منداڵەکان لە باخچە یاری دەکەن.",
        "نان و چا دەخۆین.",
    ]
    for index, text in enumerate(sorani):
        add(500 + index, "ckb", text, f"user_ckb_{index % 2}")
    add(508, "ckb", "من دەچمە بازاڕ بۆ کڕینی نان.", "user_ckb_0")
    add(509, "ckb", "سڵاو چۆنی هاوڕێ؟", "user_ckb_1")
    southern = [
        "ئیمە چیمنە باخ.",
        "دالگم نان پەتێد.",
        "کوڕەگە لە ماڵە.",
        "هەوا وەفر کەفتێیە.",
    ]
    for index, text in enumerate(southern):
        add(600 + index, "sdh", text, "user_sdh")
    add(604, "sdh", "ئەیە کتاوەگەی منە.", "user_sdh")
    add(605, "sdh", "سڵاو چۆنی هاوڕێ", "user_sdh")
    kurmanji = [
        "Ez dixwazim pirtûkekê bixwînim.",
        "Bajarê me pir xweş e.",
        "Em diçin çiyê.",
        "Diya min nan çêdike.",
        "Hevalê min mamoste ye.",
        "Baran dibare.",
        "Zarok di dibistanê de ne.",
    ]
    for index, text in enumerate(kurmanji):
        add(700 + index, "kmr", text, f"user_kmr_{index % 2}")
    add(707, "kmr", "Ez ê sibê biçim bajêr û hevalên xwe bibînim.", "user_kmr_0")
    add(708, "kmr", "pirtûka min li ser maseya mezin a odeyê ye!", "user_kmr_1")
    add(
        709,
        "kmr",
        "Bavê min her roj bi trênê diçe kar û êvarê vedigere malê xwe.",
        "user_kmr_0",
    )
    links += [["600", "100"], ["500", "300"]]
    return rows, links


WIKI_PAGES = {
    "ckb": [
        "ئەمە وتارێکە دەربارەی مێژووی شاری هەولێر و قەڵاکەی.",
        "من دەچمە بازاڕ بۆ کڕینی نان.",
        "زمانی کوردی چەندین زاراوەی هەیە.\n\nکوردستان ناوچەیەکی شاخاوییە لە ڕۆژهەڵاتی ناوەڕاست.",
    ],
    "kmr": [
        "Ev gotarek li ser dîroka bajarê Amedê ye. Zarok di baxçeyê dibistanê de bi kêfxweşî dilîzin.",
        "Çiyayên Kurdistanê di zivistanê de bi berfê dinixumin.",
        "Gelek çem di nav geliyên kûr re diherikin û digihîjin deryayê.",
    ],
    "ar": [
        "هذا مقال عن تاريخ مدينة بغداد القديمة ومعالمها.",
        "القاهرة مدينة كبيرة جدا.",
    ],
    "fa": ["این مقاله درباره تاریخ شهر اصفهان و بناهای آن است."],
    "tr": ["Bu makale İstanbul şehrinin tarihi hakkında bilgi verir."],
    "en": [
        "This article describes the history of the old city walls. The weather is nice today.",
        "Another article explains how rivers shape mountain valleys over time.",
    ],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parme(root: Path) -> None:
    manifest = load_manifest(SOURCES / "parme.v1.json")
    raw = root / "data/langid/raw/parme"
    raw.mkdir(parents=True)
    archive = raw / f"parme-{manifest.source_version}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for source_file in parme.SOURCE_FILES:
            split = source_file.removeprefix("datasets/SDH-").removesuffix(".tsv")
            lines = ["\t".join(parme.COLUMNS)]
            for row in PARME_ROWS:
                if row[0] == split:
                    english, persian, text, variety, county, translator = row[1:]
                    # Seven named columns plus PARME's trailing empty TSV cell.
                    lines.append(
                        f"{english}\t{persian}\t{text}\t{variety}\t{county}"
                        f"\tSouthern Kurdish\t{translator}\t"
                    )
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            info = tarfile.TarInfo("PARME-fixture/" + source_file)
            info.size = len(payload)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    manifest = replace(manifest, checksum_sha256=_sha(archive))
    save_manifest(manifest, root / "configs/langid/sources/parme.v1.json")
    parme.build(manifest, archive, output_dir=root / "data/langid/processed")


def _tatoeba(root: Path) -> None:
    raw = root / "data/langid/raw/tatoeba"
    raw.mkdir(parents=True)
    sentences, links = _tatoeba_rows()
    members = {
        "sentences_detailed.tar.bz2": ("sentences_detailed.csv", sentences),
        "sentences_CC0.tar.bz2": (
            "sentences_CC0.csv",
            [["130", "eng", "Hello there my friend.", "2026-01-02"]],
        ),
        "links.tar.bz2": (
            "links.csv",
            links + [[right, left] for left, right in links],
        ),
    }
    files = {}
    for archive_name, (member, rows) in members.items():
        encoded = "".join("\t".join(row) + "\n" for row in rows).encode("utf-8")
        with tarfile.open(raw / archive_name, "w:bz2") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        files[archive_name] = {
            "url": tatoeba.EXPORT_ROOT + archive_name,
            "sha256": _sha(raw / archive_name),
        }
    snapshot = hashlib.sha256(
        json.dumps(
            {name: files[name]["sha256"] for name in tatoeba.FILES},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "source_id": "tatoeba",
        "manifest_version": "1",
        "snapshot_id": snapshot,
        "files": files,
    }
    (raw / tatoeba.RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")
    tatoeba.build(
        load_manifest(SOURCES / "tatoeba.v1.json"),
        raw_dir=raw,
        output_dir=root / "data/langid/processed",
    )


def _wikimedia(root: Path) -> None:
    raw = root / "data/langid/raw/wikimedia"
    for index, (label, texts) in enumerate(WIKI_PAGES.items(), 1):
        manifest = wikimedia.manifests_for((label,))[label]
        project = wikimedia.PROJECTS[label][0]
        pages = [
            {
                "page_id": str(1000 * index + offset),
                "namespace": "0",
                "title": f"{label} fixture {offset}",
                "revision_id": str(900_000 + 1000 * index + offset),
                "revision_timestamp": "2026-10-01T00:00:00Z",
                "contributor": f"editor_{label}",
                "redirect": False,
                "wikitext": text,
            }
            for offset, text in enumerate(texts, 1)
        ]
        directory = raw / project
        wikimedia_acquire._write_pages(directory / "pages.jsonl", pages)
        wikimedia_acquire._receipt(
            manifest,
            project,
            directory,
            method="revision-pinned-action-api",
            pages=pages,
            source={
                "url": manifest.source_url,
                "cached_page_bytes": 100,
                "network_transfer_bytes": None,
            },
            sampling={
                "method": "fixture",
                "seed": 42,
                "frame_size": len(pages),
                "target_pages": len(pages),
            },
        )
    wikimedia.build(
        tuple(WIKI_PAGES), raw_dir=raw, output_dir=root / "data/langid/processed"
    )


def _benchmark(root: Path) -> None:
    directory = root / "data/langid/benchmarks/ud/v2.18"
    directory.mkdir(parents=True)
    records, provenance = [], []
    for record_id, (language, text, strict) in UD_TEXTS.items():
        treebank = "sdh_garrusi" if language == "sdh" else "kmr_kurmanji"
        record = ExternalBenchmarkRecord(
            record_id,
            text,
            language,
            "ud-kurdish-v2.18",
            treebank,
            "test",
            "external_test",
            "universal-dependencies",
            "CC-BY-SA-4.0",
            f"{treebank}_fixture",
            strict,
            () if strict else ("legacy_ud_prior_repository_exposure",),
        )
        records.append(record.to_dict())
        provenance.append(
            {
                "evaluation_record_id": record_id,
                "treebank": treebank,
                "original_split": "test",
                "license": "CC-BY-SA-4.0",
                "ud_release": "v2.18",
                "benchmark_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_text_sha256": hashlib.sha256(
                    normalize_text(text).encode()
                ).hexdigest(),
            }
        )
    for name, rows in (
        ("ud_kurdish.jsonl", records),
        ("ud_kurdish.provenance.jsonl", provenance),
    ):
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "benchmark": "ud-kurdish-v2.18",
        "record_count": len(records),
        "output_hashes": {
            name: _sha(directory / name)
            for name in ("ud_kurdish.jsonl", "ud_kurdish.provenance.jsonl")
        },
    }
    (directory / "ud_kurdish.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _ref(root: Path, path: str) -> dict[str, str]:
    return {"path": path, "sha256": _sha(root / path)}


def write_config(root: Path, **overrides: object) -> Path:
    config = json.loads(
        (REPO / "configs/langid/dataset-v1.json").read_text(encoding="utf-8")
    )
    processed = "data/langid/processed"
    for source in config["sources"]:
        family = source["family"]
        canonical = (
            f"{processed}/parme.sdh.jsonl"
            if family == "parme"
            else f"{processed}/{family}.jsonl"
        )
        source["canonical"] = _ref(root, canonical)
        source["provenance"] = _ref(root, f"{processed}/{family}.provenance.jsonl")
        source["importer_audit"] = _ref(root, f"{processed}/{family}.audit.json")
    benchmark = config["benchmark_protection"]["benchmarks"][0]
    base = "data/langid/benchmarks/ud/v2.18"
    benchmark["records"] = _ref(root, f"{base}/ud_kurdish.jsonl")
    benchmark["provenance"] = _ref(root, f"{base}/ud_kurdish.provenance.jsonl")
    benchmark["benchmark_manifest"] = _ref(root, f"{base}/ud_kurdish.manifest.json")
    config["dataset_name"] = "dataset-test"
    config["grouping"]["giant_group_min_records"] = 8
    config["sampling"]["per_label_cap"] = {
        "ckb": None,
        "kmr": None,
        "sdh": None,
        "ar": 8,
        "fa": 20,
        "tr": 20,
        "en": 20,
    }
    config["outputs"] = {
        "directory": "data/langid/builds/dataset-test",
        "work_directory": "data/langid/cache/dataset-test",
        "lock_file": "configs/langid/dataset-test.lock.json",
    }
    for dotted, value in overrides.items():
        target = config
        *parents, leaf = dotted.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    path = root / "configs/langid/dataset-test.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def build_world_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("build-world")
    (root / "configs/langid/sources").mkdir(parents=True)
    for manifest in SOURCES.glob("*.json"):
        if manifest.name != "parme.v1.json":
            shutil.copy(manifest, root / "configs/langid/sources" / manifest.name)
    _parme(root)
    _tatoeba(root)
    _wikimedia(root)
    _benchmark(root)
    write_config(root)
    return root


@pytest.fixture
def build_world(build_world_template: Path, tmp_path: Path) -> Path:
    """A private copy of the fixture world that a test may modify or build into."""

    root = tmp_path / "world"
    shutil.copytree(build_world_template, root)
    return root
