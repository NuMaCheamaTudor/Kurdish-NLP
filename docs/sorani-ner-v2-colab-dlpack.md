# Sorani NER v2: CuPy 14 / Thinc 8.3.11 DLPack compatibility

The reported CuPy 14.2.0 / Thinc 8.3.11 CUDA failure occurs when Thinc's
`xp2torch` selects the deprecated `array.toDlpack()` capsule API before the
modern `__dlpack__` protocol. The reported isolated conversion and diagnostic
smoke succeed when PyTorch consumes the CuPy array directly. That diagnostic
is user-supplied evidence; it is not a local GPU test of this implementation.

`src/kurdish_nlp/ner/colab_dlpack.py` installs a process-local adapter only for
Thinc 8.3.11 with CuPy 14.x. Only `cupy.ndarray` inputs take the modern
`torch.utils.dlpack.from_dlpack(array)` path. PyTorch negotiates the producer's
stream and owns the resulting tensor's shared-memory lifetime. Thinc's default
device resolution, subsequent `.to(device)`, and conditional
`.requires_grad_()` are preserved. Other inputs call the original function with
the original arguments. Installation is idempotent; importing the module alone
does nothing. Installed dependency files are never modified.

The pinned installed source was audited: spaCy Transformers 1.3.9 captures
`xp2torch` in `data_classes` and `layers.transformer_model`; Thinc also
re-exports it through `thinc.api`. No other production consumers were found in
spaCy Transformers, Thinc, or spaCy. Installation loads spaCy Transformers and
replaces references by identity across loaded Thinc and spaCy Transformers
modules, including aliases. Subsequent imports use the patched Thinc API.
The regression suite independently scans the installed spaCy Transformers
Python sources for imports and verifies every discovered consumer.

`colab.gpu_environment()` installs the adapter after CUDA validation and before
any pipeline initialization/load. Every official command reaches this shared
entry point: environment, prepare, smoke, stress, benchmark, gate, train and
export. The CLI's existing `cuda_runtime.bootstrap()` still runs first and
may replace Python with `os.execve()`. Installation therefore occurs in the
final process, including when the caller configured CUDA before launching
Python and no restart is needed. New environment/preflight reports include
`dlpack_compatibility` with activation status, versions and patched references.

No dataset, lock, encoder revision, hyperparameter, numerical policy,
accumulation, checkpointing, FP16, split isolation or training gate is changed.
The original `sorani-ner-v2-t4-20261022-a` and its artifacts must remain intact.
Changed source requires a new capsule and a new run; old manifests/preflights
cannot certify it.

## Rebuild and repeat on Tesla T4

The checked-in CUDA lock pins **CuPy 13.4.1**, while the reported diagnostic
environment uses **14.2.0**. The lock remains unchanged. A fresh notebook
installation from that lock should report `active: false` / `unaffected_cupy`.
That validates the locked stack, not the CuPy 14 adapter. To repeat the reported
CuPy 14 smoke, use the existing isolated Python 3.12 interpreter with CuPy
14.2.0, torch 2.6.0+cu124 and the other reported versions, retaining its original
dependency freeze/installation receipt. Do not run the CuPy 13 environment
installer over it. No dependency upgrade is needed for the adapter. A clean
CuPy 14 installation is not specified by the repository's CuPy 13 lock.

1. From the local repository, choose an unused output directory:

   ```bash
   PYTHONPATH=src .venv/bin/python scripts/prepare_ner_colab.py \
     --output artifacts/ner/colab-dlpack-fix-20261022/sorani-ner-v2-code.zip
   ```

   This refuses archive overwrite, hashes the new source and tests, and seals
   both notebook copies to the new archive/content hashes and base Git commit.
   Preserve previous archives and diagnostics. Upload the new ZIP and matching
   generated notebook together; restore into a fresh Colab workspace. The
   capsule includes `colab_dlpack.py` and `test_ner_colab_dlpack.py`.

2. Select Tesla T4. Use the notebook's existing CUDA environment discovery and
   `run_python` helper with the chosen isolated interpreter (`PYTHON`) and new
   checkout (`REPO`). Keep `LD_LIBRARY_PATH` and `CUDA_PATH` discovery intact.
   Restore/verify the unchanged dataset and cached encoder using the notebook.
   In the CuPy 14 environment, check `environment` output for CuPy 14.2.0 and
   `dlpack_compatibility.active: true`, including `thinc.util`, `thinc.api`,
   `spacy_transformers.data_classes` and
   `spacy_transformers.layers.transformer_model` references. Run:

   ```python
   run_python("-m", "kurdish_nlp.ner.colab", "environment")
   run_python("-m", "pytest", "tests/test_ner_colab_dlpack.py", "-q", "-ra")
   run_python("-m", "pytest", "tests/test_ner_colab_numerics.py", "-q", "-k", "cuda")
   ```

   The six real CuPy conversion cases must execute, rather than skip, when
   claiming CuPy 14 coverage. Two additional tiny-XLM-R CUDA tests cover FP16,
   checkpointed backward/reload and overflow replay. CPU simulations of CUDA
   probes and restart wiring do not count as GPU validation.

3. Create a new official run with the same operational settings. With the
   notebook's `datetime`, `UTC`, `Path`, `DRIVE_ROOT` and `ENCODER` variables:

   ```python
   RUN_ID = "sorani-ner-v2-t4-dlpack-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
   RUN = Path("/content/ner-runs") / RUN_ID
   DRIVE_RUN = DRIVE_ROOT / "runs" / RUN_ID
   run_python(
       "-m", "kurdish_nlp.ner.colab", "prepare",
       "--run", RUN, "--drive", DRIVE_RUN, "--encoder", ENCODER,
       "--accumulate", "8", "--eval-batch", "8",
       "--mixed-precision", "--gradient-checkpointing",
   )
   run_python(
       "-m", "kurdish_nlp.ner.colab", "preflight",
       "--run", RUN, "--stage", "smoke", log=RUN / "smoke.log",
   )
   ```

   Inspect the new `dependencies.txt`, `environment.json`, `code-manifest.json`,
   `numerics.jsonl` and `preflight-smoke.json`. Require three applied finite
   optimizer updates, expected encoder gradients, changed transformer weights,
   all 15 labels, FP16/checkpointing, equal predictions after reload, and no OOM.
   Record overflow retries explicitly. A passing diagnostic does not replace
   this official smoke. Leave full training and frozen test evaluation disabled.
   Stress and benchmark use the same adapter and remain required gates for any
   later training decision.

## Local verification and limits

Run the main suite with `.venv/bin/python -m pytest -q`, and the pinned NER suite
with `.venv-ner/bin/python -m pytest tests/test_ner_colab_dlpack.py
tests/test_ner_colab.py tests/test_ner_colab_numerics.py tests/test_ner_data.py -q`.
GPU tests skip cleanly without NVIDIA CUDA. CPU tests cover signature/device/
gradient semantics, fallback, version gating, idempotence, package consumers,
all command paths and an actual bootstrap `execve()` with simulated probes.
Local verification on 2026-10-22: main suite **249 passed, 15 skipped**
(17.29 seconds); pinned NER suite **65 passed, 8 skipped** (4.04 seconds).
The eight pinned-suite skips are six CuPy conversion cases and two tiny-XLM-R
CUDA tests. Ruff checks and formatting of the two new Python files passed.
The frozen dataset verify command passed checksum, BIO and split-isolation
checks. Temporary code capsules were generated twice with identical bytes;
restoration verified all 76 manifest members, including the adapter/tests.
Existing config/lock/notebook hashes were checked unchanged. Existing archives,
run manifests, checkpoints and diagnostic outputs were not edited or rebuilt.
Local CPU results cannot establish T4 memory/stream/AMP behavior; the official
CuPy 14 smoke, stress and benchmark remain necessary. No full training is run
by this fix.

Protocol reference: [PyTorch DLPack documentation](https://docs.pytorch.org/docs/stable/generated/torch.from_dlpack.html).
