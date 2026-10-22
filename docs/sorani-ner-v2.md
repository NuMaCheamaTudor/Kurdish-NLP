# Sorani NER V2 foundation

Validated on 2026-10-21. The release is `sorani-ner-v2.0`. Data quality and
reproducibility are validated; full model training and held-out model evaluation
remain outstanding. The real pretrained XLM-R smoke checkpoint is an engineering
artifact, not a usable quality baseline. Studio routing is unchanged.

## Verified source audit

Source: `data/ner/raw/adyan.txt`, 9,860,817 bytes, SHA-256
`31fec7da9e6ec597eaa3977107bf8bece09da0f78fe09da98e06edd1e33f3847`.
The original file was not modified. The auditor uses strict UTF-8 and two-column
BIO parsing, preserves malformed rows for review, and never silently repairs BIO.

| Measurement | Verified result |
|---|---:|
| Nonempty blank-line-delimited segments | 22,800 |
| Parsed token/label rows | 654,404 |
| Rows with valid BIO labels | 654,403 |
| Malformed rows | 0 |
| Raw entity names | 16 |
| Legitimate entity names after typo mapping | 15 |
| Repeated exact-text records beyond the first | 1,092 |
| Exact-text groups with contradictory labels | 41 (88 records) |
| Invalid BIO transitions | 77 |
| Invalid lowercase outside label | 1 (`o`, segment 1,360) |
| Segments longer than 300 tokens | 21 |
| Length: median / 95th / 99th / maximum | 25 / 58 / 134 / 1,992 tokens |

Thus the preliminary 654,403 count refers to **valid-label rows**, not all
parsed rows. The other preliminary counts were confirmed. Raw entity frequencies
are B-tag starts, not silently repaired span counts; both B starts and entity-token
frequencies are in `audit.json`.

The script inventory found a saved NER config and metadata in `kurdish-ner-v1/`;
the only notebook actually present is the POS notebook under
`kurdish-pos-output (1)/`. NER notebooks/conversion scripts advertised in the old
README are absent. `scripts/train.py`, `scripts/evaluate.py`, `project.yml`, and
`configs/base.cfg` concern Kurmanji grammar, not this Sorani rebuild. The V1 NER
config sets both train and dev to `ner.spacy`, limits corpus length to 64, and
permits oversized batches to be discarded. Its historical 88.17 F1 is excluded
from V2 comparisons. The saved V1 metadata also contains a different historical
F1 value; neither is an independent evaluation result.

Arabic-script inspection found 1,719 segments with Arabic yeh/kaf variants:
6,853 occurrences of U+0643, 5,713 of U+0649, and 684 of U+064A. Tatweel occurs
1,793 times. Digits occur in 6,908 segments and Latin characters in 404. No
non-NFC text, format controls, combining marks or Arabic presentation forms were
observed. These observations do not establish linguistic or annotation quality.
The machine audit records character counts and all flagged record IDs.

Repeated five-token prefixes/suffixes flag potential news boilerplate: for
example, `زیاتر لەسەر ئەم بابەتانە بکۆڵەرەوە` occurs as a prefix seven times.
Repeated date openings and sports/political titles also occur. These are review
candidates, not proof of common article identity. They are retained and flagged;
shared named entities or dates alone do not establish a grouping relationship.

There are no verified document/article IDs, publishers, authors, dates, original
whitespace, annotation guidelines, inter-annotator agreement, or confirmed source
license. The 22,800 segments are **not** assumed to be independent documents.

## Cleaning, provenance and frozen splits

The executable policy is `configs/ner/policy-v2.json`:

1. Map `B/I-POLITICAL_PARY` to `B/I-POLITICAL_PARTY` explicitly. One B tag changes.
   Canonicalize the sole lowercase `o` to the conventional outside tag `O`;
   no entity named `o` is inferred. Each change records its token index and line.
2. Quarantine every raw exact-text annotation-conflict group, including all its
   copies. Never choose the majority or first annotation variant.
3. After explicit mappings, quarantine remaining invalid labels/transitions.
   Starting a new entity at an orphan I tag, changing the preceding annotation,
   and merging entities imply different human intentions. None is uniquely
   established by this source, so **zero BIO transition repairs** are inferred.
4. Deduplicate exact original token sequences after quarantine. Keep every source
   record reference on the representative and in the decision ledger.
5. Retain all legitimate labels and all 21 long segments. Flag potential
   concatenation without inventing sentence/document boundaries. Transformer
   windows handle long inputs; no `max_length` filter or oversize discard is used.

Result: **21,599 retained**, **1,042 deduplicated**, **159 quarantined**.
Quarantine reasons include 88 conflict-group records and 74 remaining invalid-BIO
records, with three records belonging to both. Retained BIO tokens total 635,953.

Leakage groups are transitive connected components over **all source records**,
including quarantined ones. They combine exact text, normalized exact text, and
token-trigram set Jaccard similarity at least 0.85 for normalized segments of at
least eight tokens. Normalization uses NFKC/case folding, Arabic yeh/kaf folding,
tatweel/ZWNJ removal, and punctuation/mark removal **for grouping only**. Training
text is preserved. A frequency-ordered prefix join finds all qualifying pairs
under this definition; it uses no random candidate sampling or neighbor caps.
There are 46 near-duplicate links. Normalized-equality edges in this source
overlap the exact-text edges. Short segments use normalized equality only.

Groups are assigned using SHA-256 of seed 1729 and their content-derived IDs,
with 80/10/10 hash intervals. These are approximate ratios without label-based
test-set rearrangement. No exact text or group crosses partitions. This controls
defined lexical duplicates, not paraphrases, shared topics, or unknown articles.
**Document-independent evaluation cannot be claimed.**

| Partition | Segments | BIO tokens | Entity spans |
|---|---:|---:|---:|
| Train | 17,396 | 513,364 | 56,727 |
| Dev | 2,058 | 59,375 | 6,736 |
| Test | 2,145 | 63,214 | 7,024 |

| Entity label | Train spans | Dev spans | Test spans |
|---|---:|---:|---:|
| COUNTRY | 8,521 | 980 | 1,102 |
| CURRENCY | 1,098 | 138 | 133 |
| DAY | 597 | 78 | 82 |
| EQUIPMENT | 44 | 12 | 4 |
| EVENT | 85 | 10 | 9 |
| LOC | 6,717 | 764 | 824 |
| MISC | 3,311 | 412 | 377 |
| MONTH | 386 | 48 | 69 |
| ORG | 3,103 | 351 | 369 |
| PER | 26,194 | 3,136 | 3,289 |
| POLITICAL_PARTY | 336 | 33 | 30 |
| POSITION | 1,866 | 211 | 250 |
| RELIGION | 33 | 5 | 7 |
| SEASON | 9 | 1 | 2 |
| SPORT | 4,427 | 557 | 477 |

Insufficient support means fewer than 50 training entities or fewer than 20
entities in either held-out partition. EQUIPMENT, EVENT, RELIGION and SEASON meet
that criterion and must be reported separately. All are retained in training.
Support counts are dataset audit results, not model performance.

Generated files live under `data/ner/builds/sorani-ner-v2.0/`:

- `audit.json`, `audit-issues.jsonl`, `group-edges.jsonl`: measurements and review IDs.
- `cleaned.jsonl`, `train/dev/test.jsonl`: retained BIO records and provenance.
- `quarantine.jsonl`: original rejected annotations and explicit reasons.
- `provenance.jsonl`: one decision per original segment, including duplicates.
- `split-manifest.jsonl`, `release.json`, `policy.json`: immutable release metadata.
- `spacy/`: converted DocBins and conversion checksums.

`configs/ner/ner-v2.lock.json` freezes all release file hashes before model
selection. Its own SHA-256 is
`9cdda285aee64f1119554c40559be45e41799f8a1a19d4c4f53eff37868a8651`.
A separate complete rebuild reproduced every frozen hash. Raw data, generated
corpora, encoder weights and checkpoints stay outside Git tracking. A fresh
checkout needs an authorized copy of the exact source file; nothing downloads
another dataset automatically.

## Training validation and resource decision

`configs/ner/xlmr-v2.cfg` uses spaCy 3.8.7, spacy-transformers 1.3.9,
transformers 4.49.0 and PyTorch 2.6.0. The complete tested Python 3.12/macOS
dependency versions are locked in `requirements-py312-macos.lock.txt`. Keep the
NER environment separate from the existing Python 3.14 Language ID environment.
The pinned encoder revision and individual file hashes are in `encoder-v2.json`.

BIO entity boundaries become character offsets on single-space reconstructed
text. Strict `char_span` alignment uses spaCy's default `xx` tokenizer, shared by
training and inference. There is no permissive expansion, contraction or dropped
span. Every segment and entity survived a DocBin round trip. Only two train
segments and one dev segment retokenized (one additional token each); all entity
boundaries stayed exact. Test conversion required no token boundary changes.
The original whitespace cannot be recovered from this BIO source.

The configuration uses seed 1729, Adam with 2e-5 warmup/linear decay, development
entity F1 for selection, 32-token windows/24-token stride, padded batches of 256
with gradient accumulation 4, and up to 15 epochs/20,000 updates with patience
1,600. Oversize batches are retained. All 26,660 train/dev windows fit the encoder:
the largest has 171 wordpieces against capacity 512, without truncation.
Seeds support repeatability within the locked environment; identical floating
point results across different hardware/library builds are not promised.

A real pretrained **full-size XLM-R base**, not a random miniature substitute,
completed two optimizer updates on 15 real training records covering all labels,
with eight separate dev records. Save/load, inventory, original-text offsets and
prediction equality after reload passed. No test predictions were made.
The original sandbox attempt failed before training because OpenMP could not
open shared memory; the successful run used normal local process access.

The Apple M1 Pro has eight CPU cores and 16 GiB unified memory. CUDA is absent;
PyTorch reports MPS available, but this validated configuration uses CPU. The
initial successful smoke took 13.0 seconds for initialization/training/checkpoint
saving, with 6.02 GiB peak child RSS. Its pipeline occupies 1,134,752,126 bytes.
Reload timing includes serialization overhead and is not an inference benchmark.
After pinning a dependency-compatible CLI stack, the final replay took 11.9
seconds and 5.92 GiB peak child RSS; the final pipeline occupies 1,134,752,130
bytes. Its checkpoint is `artifacts/ner/smoke-v2-final/checkpoints/model-best`.
The transformer and NER weight files matched the initial successful run exactly.
The existing Studio loader loaded it in an isolated validation process, and the
existing API adapter's real output matched the prepared contract.

A separate CPU benchmark (four threads, batch size one) processed 32 ordinary
24–32-token dev segments, 878 tokens, in 1.35 seconds: **23.8 segments/s,
652 tokens/s**. This measures the smoke checkpoint, not a selected full model.
Two real training updates on 29/31-token training segments yielded roughly
54.7 tokens/s and a naive **39.1-hour 15-epoch compute extrapolation**. This is
not an ETA: batching, the update cap, early stopping, long segments, validation,
checkpoint writes and memory pressure change it substantially. Budget roughly
one to several days on this CPU rather than launching unattended full training.

The smoke/benchmark peaked near 6–6.3 GiB RSS on short examples. Full training can
need roughly 8–16+ GiB with activations and optimizer state; the 1,992-token
segment has not had a worst-case backward memory stress test. A CUDA host with
16–24 GiB VRAM is a more practical target, subject to an initial memory/throughput
check. **Full training was not launched** on this 16 GiB workstation. There is no
full-model V2 micro/per-label F1, rare-label F1, confusion matrix or boundary-error
result yet.
`configs/ner/validation-v2.json` preserves measured smoke/benchmark evidence.

## Reproduce and continue

Run from the repository root. `python3.12` must be available; use a separate
environment. The tested environment also passes `uv pip check`.
The supplied offline tests passed: 212 existing/repository tests (one spaCy test
skipped in the existing environment), all nine NER tests in the isolated spaCy
environment, and all 67 API tests. Lint, formatting and diff whitespace checks
passed. No frontend files changed.

```bash
python3.12 -m venv .venv-ner
.venv-ner/bin/python -m pip install -r configs/ner/requirements-py312-macos.lock.txt
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

# Existing local release: verify without modifying it.
.venv-ner/bin/python -m kurdish_nlp.ner.cli verify

# Fresh checkout: place the original source file, then build once using a new
# candidate lock. The checked-in frozen lock is never overwritten.
.venv-ner/bin/python -m kurdish_nlp.ner.cli build \
  --lock artifacts/ner/rebuilt.lock.json
cmp configs/ner/ner-v2.lock.json artifacts/ner/rebuilt.lock.json
.venv-ner/bin/python -m kurdish_nlp.ner.cli verify
.venv-ner/bin/python -m kurdish_nlp.ner.cli convert \
  --output data/ner/builds/sorani-ner-v2.0/spacy

# Only the requested encoder is downloaded. Subsequent model operations are local.
.venv-ner/bin/python scripts/ner_download_encoder.py
NER_ENCODER_PATH="$(cat data/ner/cache/encoder-path.txt)"
export HF_HUB_OFFLINE=1
.venv-ner/bin/python -m kurdish_nlp.ner.cli smoke \
  --encoder "$NER_ENCODER_PATH" --output artifacts/ner/smoke-replay
.venv-ner/bin/python -m kurdish_nlp.ner.cli benchmark \
  --encoder "$NER_ENCODER_PATH" \
  --model artifacts/ner/smoke-replay/checkpoints/model-best \
  --output artifacts/ner/benchmark-replay.json

# Full training: run only after the resource decision, preferably on CUDA.
# CPU: --gpu-id -1; CUDA device 0: --gpu-id 0 with a matching PyTorch/CUDA runtime.
.venv-ner/bin/python -m kurdish_nlp.ner.cli train \
  --encoder "$NER_ENCODER_PATH" --gpu-id -1 --output artifacts/ner/full-v2

# Select only after finishing all development-only model decisions.
.venv-ner/bin/python -m kurdish_nlp.ner.cli select \
  --model artifacts/ner/full-v2/checkpoints/model-best \
  --output artifacts/ner/full-v2/selection.lock.json
.venv-ner/bin/python -m kurdish_nlp.ner.cli evaluate \
  --model artifacts/ner/full-v2/checkpoints/model-best \
  --selection artifacts/ner/full-v2/selection.lock.json \
  --output artifacts/ner/full-v2/test-evaluation.json

# Offline regression tests (no model/network downloads).
.venv-ner/bin/python -m pytest tests/test_ner_data.py -q
```

Commands refuse existing release/run/output paths. On an existing release, skip
the fresh-checkout build/conversion block. Model training rebuilds its own
train/dev DocBins from hash-verified JSONL and does not read test annotations.
Verification hashes test files for integrity; the benchmark checks train/dev
inputs only. Dataset auditing/conversion inspects test annotations, but no model
selection or smoke predictions use them.

The test evaluator requires a frozen selection of a completed full run, rejects
smoke checkpoints, verifies the model/dataset hashes, and writes a receipt to
prevent accidental repeated evaluation in the same experiment directory. It
reports exact-span entity micro precision/recall/F1, all label scores/support,
insufficient-support labels, type confusions, one-to-one overlapping boundary
errors, model size and timed CPU inference. Undefined precision and unsupported
gold-label scores are `null`. Never use the test report to select another model.

## Studio readiness and remaining gates

The existing `POST /api/v1/analyze` Entities task and React `EntityResult` already
accept this output shape, mirrored by `kurdish_nlp.ner.contract.entities_payload`:

```text
entities: {
  model: { id: string, name: string, version: string | null },
  labels: string[],
  entities: [{ text: string, label: string, start: integer, end: integer }]
}
```

Offsets are Unicode code points into the **unchanged submitted text**, end
exclusive: `text[start:end] == entity.text`. They are not UTF-8 bytes or JavaScript
UTF-16 offsets. React already uses `Array.from` code point slicing, including
astral characters. Empty predictions are an empty array, with no fabricated
entities or uncalibrated confidence score. The benchmark validates this contract
against actual model output, including an emoji-prefixed input.

Do not point Studio at a smoke checkpoint. After full training, development
selection, one frozen test evaluation, and load/offset validation, configure a
V2 `TaskModelSpec` identity and `KNLP_NER_MODEL_PATH` for that validated model.
The current registry still names the legacy model `kurdish-ner-v1`, so merely
changing its path would misidentify V2. API/React behavior, Language ID models,
Dataset V1, and experiment locks are unchanged; protected source/lock hashes
were checked again after this work. No commit or push was made.

Remaining blockers are practical full-training resources, genuine article/source
metadata, human review of quarantined/long records, rare-label support, and
unverified source licensing/annotation methodology. No document identities or
human annotations were invented.

Implementation references: [spaCy DocBin](https://spacy.io/api/docbin),
[spaCy transformer component](https://spacy.io/api/transformer), and the
[pinned XLM-R configuration](https://huggingface.co/FacebookAI/xlm-roberta-base/blob/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/config.json).

## Google Colab Tesla T4 workflow

Open `notebooks/sorani_ner_v2_colab.ipynb` using Colab's **File → Upload notebook**
and choose **Runtime → Change runtime type → T4 GPU**. Run numbered sections
1–10 first. They verify data, the pinned encoder, real CUDA updates/reloads,
worst-case memory and GPU throughput before allowing full training. Then review
the measured budget and explicitly enable section 11. Freeze development
selection in section 13, enable the one-time test evaluation in section 14,
and export the evaluated model in section 15.

Upload exactly these two files to your own Drive:

```text
MyDrive/KurdishNLP/NER-V2/
  input/
    adyan.txt                    # Original source, unchanged
    sorani-ner-v2-code.zip        # Prepared allowlisted code snapshot
  cache/huggingface/             # Created by the notebook
  runs/<sorani-ner-v2-run-id>/    # Validated checkpoint generations / pointers
  exports/<run-id>-sorani-ner-v2.tar.gz
```

The ready code capsule is generated locally under
`artifacts/ner/colab-stress-fix/sorani-ner-v2-code.zip`. It includes the NER changes
that have not been committed; cloning the current remote alone would omit them.
The notebook pins its archive SHA-256, base Git commit and exact working-tree
content identity. It transfers no dataset, credentials, `.git` contents or model
weights. If code changes, generate a fresh matching notebook/capsule pair with:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_ner_colab.py \
  --output artifacts/ner/colab-transfer-v2/sorani-ner-v2-code.zip
```

The capsule builder refuses to overwrite an existing archive and updates the
notebook's expected identity. Use the matching newly sealed notebook, and upload
the newly generated zip under the `input/sorani-ner-v2-code.zip` name. A source
zip/tar with exactly one `adyan.txt` is also accepted if `SOURCE_UPLOAD` is changed
to its filename. Raw-source SHA-256 and the official split lock are mandatory;
no different dataset is downloaded or silently substituted.

For the CuPy 14 / Thinc 8.3.11 DLPack adapter, activation checks and a fresh
official smoke run, see [the DLPack compatibility note](sorani-ner-v2-colab-dlpack.md).
That note distinguishes the reported CuPy 14.2.0 environment from the unchanged
CuPy 13.4.1 dependency lock below.

`configs/ner/requirements-colab-cu124.lock.txt` resolves 77 exact Linux/CUDA
package versions. The macOS lock is unchanged. The model stack stays at spaCy
3.8.7 / spacy-transformers 1.3.9 / transformers 4.49.0 / PyTorch 2.6.0, with
PyTorch's CUDA 12.4 build and CuPy 13.4.1. The complete lock needs Python 3.12
because NetworkX 3.7 rejects 3.11. On another kernel Python, the installer obtains
an isolated Python 3.12 via pinned uv 0.8.22 instead of changing model versions.
Kernel packages are untouched; all ML commands run in fresh subprocesses and
`pip check` must pass. A Colab kernel restart is not needed for those isolated
installs. Driver failures require a compatible/fresh T4 runtime. Linux dependency
resolution was tested here; actual installation and CUDA execution remain Colab
steps, not locally validated GPU results.

Memory preflight uses the eight longest real TRAIN records and the longest DEV
inputs at the effective evaluation batch size. The longest TRAIN record is 819
tokens, DEV is 494, and the 1,992-token record is in TEST. Test content is not
used for tuning. A separate in-memory engineering fixture mechanically repeats
complete annotations from the longest training record to exceed 1,992 tokens;
it adds no labels or records to the frozen dataset and never updates the final
production model. It supplies a conservative long-window memory test.

Only explicit memory controls may vary: gradient accumulation, evaluation batch
size and optional FP16 (with a recorded initial gradient scale of 128). All
effective changes are recorded. Padded effective batch size 256, learning rate,
seed, objective, windows, epochs, update cap, patience and development selection
remain fixed. Accumulation cannot split a single oversized record. An OOM or
non-finite-gradient failure blocks training; retry under a new run ID with
recorded memory settings and rerun every preflight.
The runner records and tests a pinned spaCy 3.8.7 workaround: subdivision count
is capped to current batch record count. Upstream otherwise leaves the whole
batch unsplit when accumulation exceeds its length. This changes only that
memory edge, retains every record and the same effective batch, and preserves
one optimizer update and scheduler step per effective batch. GPU preflight and
full training use the same guard; the existing macOS workflow is unchanged.

Checkpoint backups run synchronously after spaCy has completely saved best/last
pipelines, not in a filesystem polling race. A generation is published only
after structural and full-file checksum verification; restore verifies again.
Two complete generations are retained, each with separate best and last models.
Pointers never advance to a partial copy. Interrupted-copy tests confirm that
the previous good backup survives. Approximately 12 GiB free Drive quota and
30 GiB local disk are recommended; account quota is not always reflected by
Drive's mounted filesystem free-space report. Synchronous Drive writes add
overhead beyond the GPU compute estimate.

The inspected spaCy 3.8.7 loop calls `nlp.to_disk()` for checkpoints and does not
serialize optimizer/scheduler/RNG state. **Exact training resumption is not
supported.** Section 12 restores verified artifacts for inspection or continuing
evaluation of a completed run. An interrupted training run restarts from the
pinned pretrained encoder under a new ID, with a fresh optimizer/scheduler.
No misleading weights-only pseudo-resume is offered. Completed runs are guarded
against replacement; automatic retention only removes older backup generations
within the same active run.

Section 14 uses the established CPU strict-span evaluator, reports mean latency
alongside precision/recall/F1, per-label support, rare-label scores, boundary/type
errors and model size, and persists a Drive test-start marker before evaluation.
It prevents accidental retesting after a disconnect. If a report/receipt reached
Drive before the large backup, recovery verifies their selection hashes and
restores them. If a test-start marker exists without a report, inspect the
interruption rather than blindly rerunning or tuning on test results.

The final private archive contains `sorani-ner-v2/model/`, the exact environment
and run/config identities, frozen selection/evaluation, logs, preflight reports,
dataset release lock and an integrity manifest. Export verifies a real CPU model
load and source-text offsets before publishing. No raw corpus is included.
Check the sidecar archive SHA-256, extract the private archive, and point a new
V2 Studio `TaskModelSpec(id="sorani-ner-v2", ...)` at its `model/` directory via
`KNLP_NER_MODEL_PATH`. The current V1 registry identity must be updated as a
separate reviewed integration; this notebook does not activate any Studio model.

No Colab T4 run was executed during preparation, so GPU memory, speed, training
duration and full-model metrics remain unverified until the user runs those
cells. Local checks cover JSON/cell order/Python syntax, import/CLI paths, real
source transfer/frozen-release verification, backup corruption/interruption and
recovery, config-bound preflight gates, and offline regression tests. Full
training was not launched on the Mac, and no commit or push was performed.
Final checks passed: 224 repository tests (three spaCy-dependent tests skipped
there), all 23 NER/Colab tests in the isolated spaCy environment, and 67 API tests.
The transferred code was restored and its CLI imported in a separate workspace;
the Python 3.12 source rebuild matched every frozen release checksum. Notebook
and code lint/format checks passed. The original BIO, macOS lock, dataset/experiment
locks and both trained Language ID artifacts still match their protected hashes.

Environment references: [PyTorch 2.6 CUDA builds](https://docs.pytorch.org/get-started/previous-versions/),
[CuPy 13.4.1 installation](https://docs.cupy.dev/en/v13.4.1/install.html),
[spaCy GPU activation](https://spacy.io/api/top-level#require_gpu), and
[Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).


## T4 mixed-precision stress correction

The notebook now defaults to FP16 (initial scale 128), accumulation 8, evaluation
batch 8 and non-reentrant XLM-R gradient checkpointing. Frozen config, release,
labels and 32/24 windows are unchanged. Runtime controls live in a separately
hashed `runtime-policy.json`. All preflights and full training use the same
numerical adapter. See [the failure analysis](sorani-ner-v2-colab-stress-fix.md).
No T4 success is claimed until the regenerated notebook passes in Colab.
