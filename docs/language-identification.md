# Language identification foundation

This package defines the data and runtime boundaries for a future Kurdish-focused
language and dialect detector. It does **not** include a trained production model.
Real, balanced, licensed datasets must be collected and evaluated before its
predictions can be considered production quality.

## Architecture

```text
text
  -> conservative Unicode/whitespace normalization
  -> backend-independent LanguageDetector
  -> backend scores
  -> reliability and abstention policy
  -> DetectionResult
```

`FastTextLanguageDetector` is the first adapter. The rest of the package imports
without fastText installed, and importing it never downloads or loads a model.
The model is loaded once when the adapter is constructed.

Target codes are `ckb`, `kmr`, `sdh`, `ar`, `fa`, `tr`, and `en`. `und` is a
runtime result meaning uncertain, unsupported, empty, or insufficient evidence.
It is currently produced by abstention rules rather than trained as a class.

Raw fastText scores are not assumed to be calibrated probabilities. A result is
reliable only when it passes minimum lexical-content, confidence-score, and
top-1/top-2 margin thresholds.

## Canonical dataset

Language-ID source data uses one JSON object per line:

```json
{"id":"ckb-news-0001","text":"دەقێکی کوردی","label":"ckb","source":"example","domain":"news","document_id":"doc-1","license":"unknown","split":"train"}
```

All fields are required. `split` is one of `train`, `dev`, or `test`. Documents
must not cross splits. Validation fails on invalid records, duplicate IDs,
normalized duplicate text, text leakage, or document leakage; records are never
silently discarded.

Empty directories under `data/langid/` reserve locations for future raw data,
processed data, and manifests. The only examples now committed are explicitly
synthetic test fixtures under `tests/fixtures/langid/`.

## Dataset commands

From an installed editable checkout:

```bash
kurdish-langid-data validate path/to/dataset.jsonl
kurdish-langid-data summarize path/to/dataset.jsonl
kurdish-langid-data convert-fasttext path/to/dataset.jsonl output.train.txt --split train
```

Without installation, use `PYTHONPATH=src python3 -m kurdish_nlp.langid.cli` with
the same subcommands.

The conversion output is standard supervised fastText text:

```text
__label__ckb دەقێکی کوردی
__label__kmr Ev nivîsek e
```

## Python inference

After separately training or obtaining a compatible custom model:

```python
from kurdish_nlp.langid import create_detector

detector = create_detector(
    backend="fasttext",
    model_path="artifacts/langid/model.bin",
    confidence_threshold=0.70,
    margin_threshold=0.20,
    model_version="1.0.0",
)
result = detector.detect("دەقێکی کوردی", top_k=3)
print(result.to_dict())
```

Install fastText support separately with `pip install -e '.[fasttext]'`. No model
binary is included, and this phase does not train, evaluate, or calibrate one.

## Future workflow and limitations

1. Acquire licensed, balanced data for all seven labels.
2. Deduplicate and split by source document, author, and domain.
3. Validate and convert the canonical JSONL.
4. Train a custom fastText baseline.
5. Evaluate per-class and short-text performance on held-out sources.
6. Calibrate scores and tune abstention thresholds.

FastAPI, React, language-to-pipeline routing, and integration with the legacy
Sorani NER and Kurmanji morphosyntax models are intentionally outside this phase.

## Acquisition manifests and provenance

Every future importer should begin with a versioned source manifest. The manifest
records a stable source ID, source release or dump date, URL, acquisition timestamp,
SHA-256 checksum, declared license, license verification URL, provenance status,
and the dataset roles selected by this project. An acquired artifact has both a
timestamp and a checksum. Metadata-only examples can leave both null; they are
not evidence that any data was downloaded or verified. The reviewed source index
is `configs/langid/sources.v1.json`; it now lists the reviewed PARME source and
Tatoeba export configuration. Example manifests in `tests/fixtures/langid/manifests/` are test data,
not approved production sources.

The compact canonical JSONL still has exactly eight fields. A separate provenance
JSONL sidecar is keyed by `canonical_record_id` and can retain the original record
and document IDs, URL, page/revision, contributor/translator, language variety,
script, original license and split, text hash, acquisition manifest version, and
processing steps. `validate_linkage` requires exactly one sidecar per canonical
record ID. Each processing step can record its name, version, optional timestamp,
JSON parameters, and input/output references. Future importers should emit both
files and retain group-level metadata for leakage-safe splitting.

License rights and project roles are separate. The license model records the
engineering interpretation of commercial use, redistribution, derivative works,
attribution, and share-alike conditions for known licenses. The project manifest
then lists selected roles such as training, calibration, benchmark, and external
evaluation. A permissive license does not force the project to train on that source.
The `commercial` profile is the default. It blocks unknown licensing, unverified
provenance, noncommercial terms, and prohibited or unknown derivative permission
for model-building roles. The `research` profile can allow noncommercial material
for research-only use, but still blocks unverified licensing and derivative bans
for training. Both profiles return structured reasons and warnings. Attribution,
share-alike, conditional redistribution, and documented provenance caveats are
warnings that require review before distribution.

Sources with vague rights in upstream scraped pages should be marked `unverified`,
even if a repackaged dataset advertises an open license. A documented caveat with
otherwise verified rights can use `verified_with_caveats`. The no-commercial (`NC`)
and no-derivatives (`ND`) conditions are evaluated independently. `unknown` is not
a shortcut to permission. Listing a source for `provenance_only` records metadata
without admitting its text into a dataset.

The metadata and policy check commands run entirely offline:

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest validate tests/fixtures/langid/manifests/parme.example.v1.json
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest show tests/fixtures/langid/manifests/parme.example.v1.json
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest policy-check tests/fixtures/langid/manifests/parme.example.v1.json --role training --profile commercial
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest policy-check tests/fixtures/langid/manifests/southern-asr.example.v1.json --role training --profile research
```

An allowed policy decision means the source metadata passes this engineering gate;
it does not verify the bytes of an artifact, validate record-level license claims,
or authorize distribution. This subsystem is an engineering safeguard and does not
replace legal review.

## PARME Southern Kurdish importer

PARME is the first approved source in `configs/langid/sources.v1.json`. The
reviewed manifest pins official repository commit
`6df9269acc75377ed6be3bf2f7966c8e238a62bd`, its MIT license, and the SHA-256
of the commit archive. Acquisition is explicit and verifies that checksum before
placing the archive under `data/langid/raw/parme/`. Import is offline and reads
only `datasets/SDH-train.tsv`, `SDH-val.tsv`, and `SDH-test.tsv` from that archive.
It never extracts the archive to the checkout. No network request occurs when the
package is imported or when tests run.

The pinned split files contain seven named TSV columns: `en_sentence`,
`fa_sentence`, `translation`, `variety`, `county`, `orthography`, and `translator`.
Each data row also has a trailing empty TSV cell. The `translation` column is the
Southern Kurdish text. In the reviewed files, `orthography` contains the value
`Southern Kurdish`; it is preserved as supplied and is not treated as a script
code. The files provide no stable row or alignment IDs. The importer therefore
uses a content hash for each canonical ID and connects rows sharing normalized
English or Persian source text into one alignment group. Those source texts are
used for grouping; only the Southern Kurdish text is exported. Group IDs become
canonical `document_id` values, and a seeded hash assigns each group wholly to
train/dev/test. Ratios default to approximately 80/10/10. Upstream train/val/test
labels remain in sidecars and the audit so differences can be inspected.

The five observed source varieties are Pehley, Kirmashani, Kalhori, Garusi, and
Badrei. Their original labels, county, translator ID, orthography field, source
file and row number are preserved in the provenance sidecar. No `hac`, Laki, or
other neighboring variety is relabeled as `sdh`. Empty or malformed rows,
unexpected labels or varieties, duplicate rows, and exact normalized text
duplicates are counted and reported. Very short and long rows are flagged but
retained. Conservative normalization preserves Kurdish-specific characters and
joiners.

From the repository root:

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest policy-check configs/langid/sources/parme.v1.json --role training --profile commercial
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire parme --dry-run
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire parme
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import parme
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli validate data/langid/processed/parme.sdh.jsonl
```

The import writes `parme.sdh.jsonl`, `parme.provenance.jsonl`,
`parme.audit.json`, and `parme.audit.txt` under ignored
`data/langid/processed/`. Run `import parme` again with the same archive, manifest,
seed and ratios to reproduce the exact canonical and provenance bytes. `--seed`,
`--train-ratio`, `--dev-ratio`, and `--test-ratio` permit a different deterministic
split. Use `--archive` and `--output-dir` for an already acquired local archive or
an isolated test build; checksum verification still applies.

At the pinned commit, the three source splits have 9,806 rows. The verified import
accepted 9,795, rejecting nine duplicate rows and two duplicate texts. There
were 8,579 alignment groups, with one group spanning upstream split labels.
Kirmashani accounts for 7,413 accepted records, and translator `Vol_0` for 6,560;
the machine-readable audit has full variety, translator, split, length, and hash
distributions. These numbers describe this pinned source and processing version,
not a balanced or final Language ID dataset.

PARME alone is not sufficient evidence that the final Southern Kurdish detector
generalizes to natural web/conversational text. Its translated-prompt domain,
variety and translator imbalance, and scarce independent evaluation material
remain limitations. No model training is included in this importer.

## Tatoeba multi-language importer

Tatoeba complements PARME with contributed example sentences and translations in
all seven target labels. It is **not** naturally occurring dialogue or social
media, and should not become the sole source for a language when broader-domain
material is available. Wikimedia is planned separately.

The [official downloads page](https://tatoeba.org/en/downloads) documents three
exports used here: [`sentences_detailed.tar.bz2`](https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2)
(sentence ID, language, text, username, creation and modification dates),
[`sentences_CC0.tar.bz2`](https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2)
(CC0 sentence IDs), and [`links.tar.bz2`](https://downloads.tatoeba.org/exports/links.tar.bz2)
(translation edges). The detailed export does not have a license column. Tatoeba
declares these download files CC BY 2.0 FR, with a subset additionally available
under CC0 1.0. The importer assigns `CC0-1.0` only to IDs present in the CC0
export and `CC-BY-2.0-FR` to other official detailed-export rows; an explicit
unrecognized license field (supported for validation fixtures or future exports)
is rejected, not replaced by the default. Commercial builds reject NC, ND and
unknown/missing licenses. CC BY records require contributor attribution before
redistribution. These are engineering gates, not legal advice.

Official export URLs are **rolling**, not immutable releases, and Tatoeba does
not publish SHA-256 files alongside them in the reviewed download directory.
`configs/langid/sources/tatoeba.v1.json` is a reviewed source configuration,
not a claim that a particular weekly export has been acquired. `acquire tatoeba`
explicitly downloads the three files into ignored `data/langid/raw/tatoeba/`
and creates `tatoeba.snapshot.json` there with a SHA-256 for each file and a
content-derived snapshot ID. Keep the three archives and receipt together to
rebuild exactly; `import tatoeba` verifies all three checksums offline before
reading anything. A mid-download upstream rollover cannot be fully excluded;
inspect recorded HTTP Last-Modified values and reacquire a coherent export if
they suggest different weeks. An existing verified receipt is reused, and a
checksum mismatch fails rather than silently refetching.

Tatoeba export codes map explicitly: `ckb→ckb`, `kmr→kmr`, `sdh→sdh`,
`ara→ar`, `pes→fa`, `tur→tr`, `eng→en`. These exact directories/codes are
visible in the [official per-language export index](https://downloads.tatoeba.org/exports/per_language/).
Generic Kurdish or other Kurdish-related codes (`hac`, `lki`, `zza`, `diq`)
are not relabeled. The importer accepts any subset of the seven canonical
labels, but always reads the **global** links file so transitive paths through
non-target languages remain intact. Each translation component gets the stable
ID `tatoeba:translation-group:<minimum sentence ID>`, independent of row order
and requested language subset. A seeded hash of that ID assigns the entire
component to train/dev/test, default 80/10/10. Only the exact same snapshot,
seed and ratios guarantee that a later subset import receives the same group
and split; Tatoeba can edit links in future snapshots.

The real upstream files are literal tab-separated exports: quotation marks in
sentence text are not CSV quoting. The importer parses them accordingly. Large
uncapped exports use a disk-backed candidate/deduplication table and stream the
canonical and provenance files; the complete global translation graph is still
read. Keep several gigabytes of free disk space for the temporary table and
staged outputs. A read-only streaming quality/overlap check is available with
`PYTHONPATH=src python3 scripts/audit_tatoeba_real.py` after PARME and Tatoeba
have both been imported.

The canonical JSONL remains eight fields: `tatoeba:<label>:<sentence ID>` is the
record ID, `tatoeba:<snapshot hash>` is the source, the component ID is
`document_id`, and the domain is `conversational_translation`. A separate
provenance row retains the original Tatoeba ID and code, contributor, original
license resolution, source snapshot, text hashes, and transformations. The
official created/modified timestamps are retained with the license-resolution
transformation metadata. Normalization remains conservative NFC/whitespace;
one-token linguistic text is retained. Punctuation/numeric-only rows are
excluded as content-free, and length/normalization counts are audited.

Sampling with `--max-per-language` happens after license validation and uses a
seeded ordering of **whole translation groups**, never the first N rows. A group
larger than the cap is skipped, so realized counts may be below the cap; no
language is downsampled unless a cap is explicitly given, including `sdh`.
Duplicate normalized text within one label keeps the lowest sentence ID.
Identical normalized text across labels is excluded from canonical training
output and counted, with hashed examples under the default `report` policy;
`exclude` keeps counts but omits examples. This intentionally preserves evidence
of ambiguous strings such as names and acronyms without violating the strict
canonical duplicate validator. Near-duplicate detection is not implemented.

From the repository root:

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest policy-check configs/langid/sources/tatoeba.v1.json --role training --profile commercial
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire tatoeba --dry-run
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire tatoeba
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import tatoeba --languages ckb kmr sdh ar fa tr en
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import tatoeba --languages ckb sdh fa
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import tatoeba --languages ckb kmr sdh ar fa tr en --max-per-language 20000
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli validate data/langid/processed/tatoeba.jsonl
```

Import writes ignored `tatoeba.jsonl`, `tatoeba.provenance.jsonl`,
`tatoeba.audit.json`, and `tatoeba.audit.txt` in `data/langid/processed/`.
The audit includes source checksums, decisions, per-language/license/split
counts, duplicate and rejection reasons, component leakage, contributor
concentration, short-text counts, and normalization changes. Translationese,
uneven language sizes, contributor bias, and inherently ambiguous short strings
remain important evaluation limitations. No real Tatoeba acquisition or model
training is part of the offline test suite.

## Wikimedia encyclopedia importer

Six reviewed manifests map `ckbwiki→ckb`, `kuwiki→kmr`, `arwiki→ar`,
`fawiki→fa`, `trwiki→tr`, and `enwiki→en`. There is no Wikimedia `sdh`
source in this importer. The central commercial policy gate allows the reviewed
CC-BY-SA-4.0 sources with attribution, share-alike, derivative, and
redistribution warnings; this is not a determination that every embedded quote
or third-party excerpt is covered by the same license. Each source manifest
links to that edition's official API `rightsinfo` declaration, observed to
state CC BY-SA 4.0; receipts bind the full manifest SHA-256. The page and revision
URLs in the provenance sidecar support attribution review. The recorded
`contributor_id` is only the cached revision's editor, not the full set of
article authors; consult page history for attribution.

`ckbwiki` uses the completed dated 20260801 pages-articles-multistream dump;
`kuwiki` uses the completed dated 20261001 dump. Acquisition checks the
official `dumpstatus.json` job and file size/MD5/SHA1, then records local
SHA-256. XML is streamed twice: a seeded, order-independent bottom-k sample of
all mainspace page IDs, then extraction of only the selected current revisions.
The archive download can resume from an existing partial file if the server
honors the exact Range offset. These are current-revision dumps, not histories.

The larger four Wikipedias use a bounded, sequential Action API acquisition:
server-random mainspace page IDs are frozen in an ignored frame, their revision
IDs are pinned, and exact wikitext is cached and hash-receipted. The seed
controls splits, and dump sampling; it **does not** control Wikimedia's
server-random selection. Reproducibility of an API selection depends on
retaining its local frame, page cache, and receipt. API responses collected at
different times are not an atomic Wikimedia snapshot. The client uses a
descriptive User-Agent with a configurable contact URL or email, `maxlag=1`,
one connection, at least one second between requests, bounded retries,
exponential backoff, and `Retry-After` for HTTP 429. Ordinary imports and tests
never access the network.

Install the optional `wikimedia` extra for `mwparserfromhell`. The extractor
conservatively drops templates, tables, references, galleries, navigation,
media links, redirects, disambiguation pages, and residual wiki markup. It
selects prose, then emits sentence, paragraph, or bounded multi-sentence
segments (bounded is the default). Script checks are diagnostic flags, **not**
label ground truth. All segments from one project/page ID share a seeded split,
even when revisions differ. Exact normalized duplicates are removed within a
label; identical text across labels is excluded and audited. Near-duplicate
and semantic language-quality review remain future dataset-builder concerns.

From the repository root:

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire wikimedia --languages ckb kmr ar fa tr en --max-pages 20 --dry-run
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire wikimedia --languages ckb kmr --max-pages 3000 --contact 'https://github.com/NuMaCheamaTudor/Kurdish-NLP/issues'
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire wikimedia --languages ar fa tr en --max-pages 200 --contact 'https://github.com/NuMaCheamaTudor/Kurdish-NLP/issues'
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import wikimedia --languages ckb kmr ar fa tr en
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli validate data/langid/processed/wikimedia.jsonl
```

Acquisition writes ignored `data/langid/raw/wikimedia/<project>/` archives or
cached API pages, sampling frames, source metadata, and receipts. Import verifies
these local inputs and writes ignored `wikimedia.jsonl`,
`wikimedia.provenance.jsonl`, `wikimedia.rejections.jsonl`,
`wikimedia.audit.json`, and
`wikimedia.audit.txt` under `data/langid/processed/`. The canonical file has
the existing eight fields; additional page title, revision timestamp, section,
quality flags, parser version, and source checksum live in provenance
transformation parameters. The audit includes page/segment counts, split and
license distributions, text lengths, script flags, exact deduplication, and
read-only normalized overlap checks against local Tatoeba and PARME files.
The rejection sidecar retains page/revision identity and normalized text hashes
for every discarded duplicate or segment-limit candidate, without copying
discarded full text.

## Universal Dependencies v2.18 external benchmark

The reviewed `ud-sdh-garrusi-v2-18` and `ud-kmr-kurmanji-v2-18` manifests pin
the official UD `r2.18` tag archives and CC BY-SA 4.0. Their only allowed
project roles are `external_evaluation` and `benchmark`, even though the
upstream license permits uses subject to its terms. Preserve contributor
attribution, the license notice and share-alike obligations when distributing
text or derivatives. Cite Gökırmak and Tyers (2017) when using Kurmanji.
The release is dated 2026-05-15. As checked 2026-10-08, the CoNLL-U files in
both repository `master` branches were byte-identical to the tagged files;
the mutable branch archive itself had a different hash. Always use the tag.

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire ud --dry-run
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli acquire ud
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli import ud --languages sdh kmr
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli manifest policy-check configs/langid/sources/ud-sdh-garrusi-v2.18.v1.json --role training
```

Acquisition is the only network step. It caches two checksum-verified tarballs
under ignored `data/langid/raw/ud/v2.18/`; an existing corrupt tarball fails
closed. Import is offline. It reads original CoNLL-U train/dev/test files as
*source metadata*, not Language ID development partitions. The source `# text`
is used when present, with Unicode and orthography unchanged; a surface-token
reconstruction handles multiword tokens, empty nodes and `SpaceAfter=No` only
when needed. Lemmas, UPOS, morphology and dependencies are never Language ID
features. The importer has a small strict CoNLL-U parser rather than a runtime
dependency because it needs only sentence metadata and text surfaces; malformed
rows are counted and rejected, not silently repaired.

All outputs are under ignored `data/langid/benchmarks/ud/v2.18/`, separate
from canonical training JSONL. `ud_kurdish.jsonl` uses a strict evaluation
schema with `expected_language` and `evaluation_partition=external_test`, not
canonical `label`/`split`. The matching provenance sidecar stores original
sentence/document/paragraph IDs where supplied, source comments and annotation
hashes, text hashes, source file and archive checksums, reconstruction choices,
script statistics, genre and overlap matches. Fallback group IDs are coarse
source groupings, **not inferred document identities**. The audit JSON/TXT and
benchmark manifest record exact counts and output hashes. A paired
`ud_kurdish.strict.jsonl`/`.strict.provenance.jsonl` view contains only rows
without within-UD duplicates, matching canonical source text, or prior legacy
UD repository exposure. The full output retains excluded rows and reasons.

The audit streams **all** local PARME, Tatoeba and Wikimedia canonical rows for
exact and conservative normalized equality. Near-duplicate candidates require
a common normalized five-word shingle and then at least 0.88 casefolded
sequence similarity at comparable lengths. This bounded-memory method does
not catch shorter paraphrases, texts with no shared five-word span, or semantic
duplicates. The legacy Kurmanji CoNLL-U files are checked by byte hash, then
sentence text if hashes differ. Missing comparison datasets are reported as
unavailable and block strict-clean eligibility, not treated as zero-overlap
evidence. Reimport from the same verified
archives to compare the output hashes in `ud_kurdish.manifest.json`.

Garrusi is an original Latin-based Southern Kurdish orthography sample in the
`sdh_garrusi_original_orthography` slice. It is not a proxy for all Southern
Kurdish varieties or Arabic-script orthographies in PARME and Tatoeba. Kurmanji
contains fiction and Wikipedia-derived material; its historical presence in
this repository means it is **not** an independent benchmark for the existing
POS/dependency models. A future, previously untrained Language ID model can
report a clearly labeled exploratory Kurmanji analysis after excluding source
matches, but must not use any UD records for training, model selection or
abstention calibration. Repeatedly selecting models by UD performance would
compromise the held-out claim. With only `sdh` and `kmr` true labels, this
benchmark cannot estimate seven-class macro-F1; report per-class recall,
accuracy, confusion, script/genre/length slices, abstention and coverage.
The future dataset-v1 builder must explicitly deny-list both UD source IDs and
all benchmark paths, regardless of original CoNLL-U split names.

On the verified 2026-10-08 import, Garrusi contributed 221/221 accepted
sentences (1,551 surface tokens; train/dev/test 152/49/20); Kurmanji
contributed 754/754 (10,188 surface tokens; train/test 20/734). Neither had
missing `# text` or `sent_id` comments, malformed sentences, or text
reconstruction cases. Neither supplied `newdoc`/`newpar` identifiers. Two
Garrusi normalized-text duplicate pairs affect four records, including one
pair across the original dev/test splits. The conservative strict view has
217 Garrusi and zero Kurmanji records. Both local historical Kurmanji CoNLL-U
files are byte-identical to v2.18; a separate, explicitly exploratory
new-Language-ID eligibility count is 969 after excluding only canonical-source
overlaps and within-UD duplicates. It does **not** erase prior repository
exposure. Tatoeba and Wikimedia each matched one Kurmanji sentence exactly,
both in their canonical train splits. No PARME overlap was found with this
exact/normalized/five-word-shingle method. Every Garrusi sentence was Latin
script; all 9,795 local PARME and 1,046 local Tatoeba `sdh` rows were Arabic
script, making any future performance difference strongly confounded by
orthography and domain.

## Dataset V1 builder

`kurdish-langid-data build dataset-v1` combines the verified PARME, Tatoeba and
Wikimedia exports into one versioned, reproducible train/dev/test release. It
never downloads data, never modifies source exports or the UD benchmark, and
trains no model. Every experiment setting lives in the tracked configuration
`configs/langid/dataset-v1.json`; the tracked lock
`configs/langid/dataset-v1.lock.json` freezes the configuration hash and the
SHA-256 of every reproducible output.

```bash
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli build dataset-v1 --dry-run
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli build dataset-v1
# independent reproducibility check against the frozen lock
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli build dataset-v1 --output-dir data/langid/cache/rebuild
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli validate data/langid/builds/dataset-v1/dataset.jsonl
PYTHONPATH=src python3 -m kurdish_nlp.langid.cli convert-fasttext data/langid/builds/dataset-v1/train.jsonl train.txt
```

The dry run performs every verification step and writes nothing. A rebuild
whose configuration differs from the lock is refused: changing sampling,
grouping or split settings requires a new `dataset_version`, configuration and
lock rather than silently replacing a frozen release.

### Data flow

1. **Verify.** Each source manifest must pass the importer's identity checks
   and the central policy for the training, development and test roles under
   the configured profile. Benchmark source IDs, benchmark paths and `ud:`
   record IDs are deny-listed. PARME's pinned archive, the Tatoeba snapshot
   receipt and the six Wikimedia page/revision receipts are re-verified
   offline; canonical, provenance and importer-audit files must match pinned
   SHA-256 values, and each importer audit must attest the same export hashes
   and acquisition snapshot. The full UD benchmark (all 975 rows, not only the
   strict slice) is validated with its provenance and benchmark manifest.
2. **Ingest.** Canonical and provenance files are streamed line by line into a
   SQLite work store under the ignored `data/langid/cache/`. Every pair is
   checked for schema validity, ID/group/source/snapshot linkage, normalized
   text and per-record license; both files are re-hashed while streaming, and
   every provenance row is strictly schema-validated by parallel worker
   processes. Only compact attributes, keys and byte offsets are stored; text
   and full provenance are re-read by offset for exported records.
3. **Benchmark guard.** Every candidate is compared with every UD sentence:
   exact normalized text, loose (NFKC/casefold/punctuation-insensitive) text,
   sentence containment in either direction (shared sentences of at least three
   tokens), character 5-gram Jaccard >= 0.6, and >= 0.8 containment of a UD
   sentence's 5-grams (UD sentences with at least 20 5-grams). Matches are
   quarantined, never exported, and re-checked on the final output.
4. **Deduplicate.** Exact (`normalize_text`) and loose duplicates are clustered
   across all sources. A one-label cluster keeps one canonical record (source
   priority PARME, Wikimedia, Tatoeba; then the most permissive license; then
   record ID) and links every other member in `duplicates.jsonl`. A cluster
   whose members carry different labels is withheld from supervised data and
   preserved in `ambiguous.jsonl` for a future ambiguity/abstention benchmark.
   Records with fewer than three letters (the runtime abstention minimum) are
   excluded.
5. **Group.** A union-find forest connects records sharing a source document
   (PARME alignment group, Tatoeba translation component, Wikipedia page), an
   exact/loose duplicate cluster, or a contained sentence. Roots are the
   smallest record ID, so groups do not depend on input order. Groups with at
   least 1,000 eligible records are *giant* and are forced into train.
6. **Sample.** The unit is a group/label/source slice, so sampling never splits
   a group. `ckb`, `kmr` and `sdh` keep every eligible record. `ar`, `fa`, `tr`
   and `en` are capped at 20,000: the cap is divided across sources by max-min
   fairness (Wikimedia is kept whole), Tatoeba slices are visited in a seeded
   group-hash order, and a soft cap defers slices that would push one Tatoeba
   contributor above 25% of the source quota. A giant group contributes the
   same fraction of its records as its source's overall sampling rate. No
   record is duplicated and no text is synthesized.
7. **Near duplicates.** An exact prefix-filtered character 5-gram Jaccard
   similarity join (threshold 0.8) runs over the sampled pool. Pairs are put in
   the same group rather than deleted, so legitimately distinct dialect
   variants are kept; cross-label pairs are flagged for review.
8. **Split.** A deterministic greedy assignment places each final group in the
   split that best keeps every (label, source, variety) stratum at 80/10/10.
   Source-export splits are reported and reconciled, not reused.
9. **Export and validate.** Outputs are written to a staging directory,
   validated independently (canonical schema, exact and loose duplicates, text,
   global-group and source-document leakage, near-duplicate pairs across
   splits, provenance linkage, benchmark re-check, caps), compared with the
   lock and only then swapped into `data/langid/builds/dataset-v1/`.

### Outputs

All outputs are git-ignored. `train.jsonl`, `dev.jsonl` and `test.jsonl` use
the unchanged eight-field canonical schema; `dataset.jsonl` is their
concatenation, line-aligned with `provenance.jsonl`. `document_id` is the global
leakage group, so `kurdish-langid-data validate dataset.jsonl` checks group
isolation directly. Each provenance row is the original source sidecar plus a
`dataset_build` transformation recording the source export path, hash, split,
document and label, review status, merged duplicates, global group, sampling
method and split. `duplicates.jsonl`, `near_duplicates.jsonl`,
`quarantined.jsonl` (with reasons and benchmark matches), `ambiguous.jsonl` and
`review_queue.jsonl` explain every decision. `audit.json`/`audit.txt` contain
the 20-section research audit, `manifest.json` the configuration, input and
output hashes, counts, policies, license obligations, software versions and
known limitations. Run time, peak memory and disk use are kept in
`build_run.json`, which is excluded from reproducibility hashes.

The test split is for final internal evaluation only; use dev for model
selection, tuning and calibration. UD v2.18 stays a separate external
benchmark and must never be used for training, development or calibration.

### Human review

Human review is postponed. Every exported label is recorded as `unreviewed`.
Future adjudications can be supplied as hash-pinned JSONL sidecars in the
configuration (`review.sidecars`) with a canonical record ID, review version,
reviewer, timestamp, decision (`accept`, `relabel`, `reject`, `quarantine`),
corrected label, quality status, contamination flags and notes. They are
applied during an explicit, versioned rebuild; nothing changes a label without
such a record.

### Measured dataset-v1 release (2026-10-09)

Built from the verified 2026-10-07/08 acquisitions with configuration SHA-256
`0c948dc2853021d96f0177661f439fe1ba544b1cd7c8d01f1e6fc3f00b8b652b`. Of
2,935,725 input records, 124 had fewer than three letters, 25 overlapped the
UD benchmark, 26 formed 12 cross-label ambiguity clusters, and 12,447 were
removed as duplicates of a retained canonical record (12,446 loose variants,
one exact cross-source PARME/Tatoeba `sdh` duplicate).

| Language | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| ckb | 14,346 | 1,792 | 1,792 | 17,930 |
| kmr | 12,800 | 1,599 | 1,599 | 15,998 |
| sdh | 8,665 | 1,081 | 1,080 | 10,826 |
| ar | 16,002 | 1,999 | 1,999 | 20,000 |
| fa | 16,002 | 1,999 | 1,999 | 20,000 |
| tr | 16,002 | 1,999 | 1,999 | 20,000 |
| en | 16,002 | 1,999 | 1,999 | 20,000 |

All 2,935,725 candidates were screened against all 975 UD sentences. The 25
quarantined records (6 Tatoeba, 19 Wikipedia `kmr`) matched 21 UD Kurmanji
sentences and no strict-slice Garrusi sentence: 2 exact, 1 loose, 14 sentence
containments, 4 5-gram containments and 4 short-text Jaccard matches. The
earlier whole-text UD audit found only the two exact matches; most other
matches are UD Wikipedia sentences embedded in longer kuwiki segments. The
short-text Jaccard matches are conservative and include minimal pairs.

The Tatoeba giant translation component (112,659 records in the source export)
grew to 138,394 eligible records through duplicate and containment links and
absorbed one PARME alignment group; it is train-only and contributes 8,740
rate-matched records. The near-duplicate join verified 278,039 candidate pairs
in the 124,754-record pool and found 3,271 pairs at Jaccard >= 0.8 (none across
labels). Every output check reported zero exact, loose, group, source-document
and near-duplicate cross-split leakage and zero benchmark matches. Two
independent builds produced byte-identical reproducible files matching the
lock. A build takes about 5.3 minutes on an 8-core 16 GB laptop with a peak
resident memory of 1.8 GB and a 1.6 GB temporary work store; outputs occupy
about 420 MB.

Known risks recorded in the audit: one contributor provides 96% of Tatoeba
`ckb` (68% of final `ckb`), one translator 67% of PARME (61% of final `sdh`),
and one contributor 50% of Tatoeba `kmr`; `sdh` is 90% translated prompts and
entirely Arabic script; encyclopedic text is 29% of `ckb` and 35% of `kmr` but
3-6% of the capped classes. These confounds must be considered when
interpreting per-class results.
