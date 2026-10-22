# Sorani NER V2: T4 stress failure and correction

The reported FP32 run exhausted roughly 14.5 GiB. The subsequent FP16 smoke
passed but long-example stress failed at the old generic gradient assertion.
The historical log contains no tensor names, devices, values or scaler state.
Consequently **the exact tensors and numerical cause in that historical run
cannot be recovered**. The assertions below distinguish reproduced defects from
GPU observations that still require a rerun.

## Findings from the exact pinned implementations

Inspected local source: spaCy 3.8.7 `training/loop.py`, Thinc 8.3.11
`shims/pytorch.py` and `shims/pytorch_grad_scaler.py`, spacy-transformers 1.3.9
`layers/hf_shim.py`, `layers/transformer_model.py` and `pipeline_component.py`,
and transformers 4.49.0 XLM-R encoder checkpointing.

1. The old `_update()` collected only non-None parameter gradients. It could
   not detect an individual unexpectedly missing gradient. It collapsed empty,
   wrong-device and non-finite cases into one error, discarding tensor names.
   Since the CUDA smoke passed, a scale-dependent backward overflow on longer
   inputs is plausible, but the old message cannot establish it.
2. Thinc scales the incoming transformer output gradients, runs backward,
   then unscales **parameter gradients on every backward callback**. Thus the
   old validation ran **after** per-backward unscaling and **before**
   `finish_update()` could skip and back off its scale. It incorrectly aborted
   recoverable overflows before that handling occurred.
3. With accumulation and scale S, the pinned implementation computes
   `G1/S + G2` after two microbatches instead of `G1 + G2`. Earlier gradients
   had already been unscaled when the next scaled gradients were added.
   A regression reproduces this arithmetic at S=128, then verifies the fix.
   This is a separate training-correctness defect, not proof that it caused
   the historical overflow. PyTorch requires constant scaling across the
   effective batch and one parameter unscale after accumulation.
4. Stock spaCy finishes both components even if Thinc skips the transformer
   due to overflow, potentially applying only the NER head's update and
   advancing schedules. The new step validates before either update.
5. XLM-R's `pooler.dense.weight` and `pooler.dense.bias` legitimately have no
   gradient: the spaCy listener backpropagates `last_hidden_state`, not the
   unused pooler output. Those exact two names are reported as expected-unused;
   other missing trainable parameters fail. Frozen parameters are separately
   listed. Zero-valued gradients remain valid (including initial NER behavior),
   with nonzero tensor counts logged; smoke separately requires changed encoder
   weights across three finite updates.

## Permanent changes

- `colab_numerics.py`: a version-guarded scaler proxy defers parameter unscale
  until all microbatches finish. Differentiable input gradients retain immediate
  unscaling. The original scaler is restored before `finish_update()` and on
  errors. Dependency files and installed package sources are not patched.
- Before optimizing: check finite transformer forward windows, finite losses,
  FP32 Thinc/NER gradients, every active encoder gradient's presence, device and
  finite values. Log named bad tensors, shape/dtype/device, separate NaN/+Inf/-Inf
  counts, scaled/unscaled phases, scaler flags/scales, microbatch sizes, forward
  window shapes, update index and stage to `numerics.jsonl`. Check parameters
  after applying the update. Large embedding checks use bounded temporary masks.
- Only backward non-finites detected by an enabled scaler, with finite forward,
  losses and NER gradients, enter bounded backoff/replay. Discard **both** heads'
  gradients, lower the scale, restore RNG/dropout choices and replay the **same
  complete effective batch**. At most eight retries. No optimizer or scheduler
  advances until a finite update applies. Persistent Inf/NaN, missing/device
  failures and forward/loss/head instability stop the run; a skipped batch can
  never count as a successful preflight. No training records are omitted.
- `colab.py`: use that same checked step in smoke, stress, benchmark and full
  training. The full-training generator retains pinned development evaluation,
  checkpoint ties, patience, epoch iteration and step-cap behavior. Runtime
  policy is separately hashed; matching policy/config and applied updates are
  required by the training gate. Failure JSON includes traceback, numerical
  details and memory when available. Previous successful reports cannot certify
  a changed runtime policy.
- Reject any oversize wordpiece window before the pinned built-in helper could
  truncate it; require the observed transformer forward window count to equal
  the complete strided-span count on every update.
- Bound record microbatches even when accumulation exceeds batch size or the
  pinned subdivision leaves a large uneven remainder. Keep all records, complete
  examples and one applied optimizer/scheduler step per effective batch.
- Enable supported non-reentrant XLM-R gradient checkpointing via
  `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`.
  Reapply after initialization **and each checkpoint reload**, disable unused
  training cache, and verify activation. This reduces stored encoder activations
  for long single records without deleting transformer windows. Record-level
  accumulation alone cannot reduce a singleton's window footprint.
- `cuda_runtime.py`: discover isolated environment NVIDIA-wheel `lib` directories
  and `torch/lib` using stdlib only. Put them in `LD_LIBRARY_PATH`, and select
  matching runtime headers for `CUDA_PATH`, **before launching ML processes**.
  The CLI re-execs itself before imports when necessary. The notebook computes
  the environment with the isolated interpreter before the first CUDA probe.
- `ner_download_encoder.py`: explicitly create `data/ner/cache` before writing
  `encoder-path.txt`, including when `HF_HOME` points to an external Drive cache.
- Include runtime policy and numerical logs in backup/export. Ship the targeted
  tests in the capsule; the notebook runs real tiny-XLM-R CUDA regressions before
  dataset preflights. Both notebook copies are sealed to the same archive/content SHA-256 identities and base Git commit.

The frozen dataset release, source, split files, base config, label inventory,
encoder identity and dependency locks are unchanged. Window=32, stride=24 and
all long records remain intact. `max_batch_items` is not presented as a hard
bound on training windows. No unsupported checkpointing config key or custom
truncation is introduced.

## Verification and T4 rerun

Local tests use small randomly initialized XLM-R models and local synthetic
vocabulary; no pretrained downloads or full training. They cover actual
spaCy/Thinc backward, accumulation math, named failure classification,
checkpointed long-window updates, reload, backward Inf and NaN replay,
persistent failure, missing gradients, non-finite forward/head, FP32 failure,
full-loop selection/patience behavior, uneven batching, loader discovery,
cache creation, notebook safety and policy-bound preflight gating. CPU replay
fixtures emulate only the scaler's CUDA-only multiplication; they do not claim
CUDA AMP validation.

Verification: the isolated pinned NER suite passed **41 tests**, with **2 CUDA
tests skipped**. The main environment suite passed **226 tests** (ML tests skip
there because the main environment intentionally has no spaCy/PyTorch). Ruff
checks passed. Source, frozen configs/locks and model identities are checked
separately against their protected hashes.

Two separate tests exercise real CUDA autocast/scaling, checkpointed backward,
checkpoint reload and overflow replay. They skip when CUDA is absent. This Mac
has no CUDA GPU, so there are **no local T4 memory, throughput or full-training
results**. No full training, frozen test prediction, commit or push was performed.

Upload the new notebook and matching zip from
`artifacts/ner/colab-stress-fix/sorani-ner-v2-code.zip`; retain the failed run.
Use a fresh T4 runtime/workspace and run ID. The notebook proposes FP16,
init_scale=128, accumulation=8, eval_batch=8, allocator=pytorch and checkpointing.
It runs the real CUDA regressions, three-update smoke, the eight longest actual
TRAIN records, the complete 2,457-token engineering envelope, longest-DEV
inference and throughput/full-DEV benchmark. All must pass on this exact code,
config and policy before full training can be enabled.

**Whether this fits and can sustain training on a 15 GiB T4 remains to be
verified in Colab.** Checkpointing plus FP16 addresses the activation-memory
pressure but is not a capacity guarantee. On failure, share `preflight-stress.json`,
`numerics.jsonl` and `stress.log`; these now preserve the missing evidence.
If finite singleton stress still OOMs, more VRAM is required rather than
truncating data or claiming success. Full training remains disabled by default.

References: [PyTorch AMP accumulation rules](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[Transformers 4.49 checkpointing API](https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable),
[CuPy CUDA library/header setup](https://docs.cupy.dev/en/v13.5.0/install.html).


Sealed delivery identities:

- Code zip SHA-256: `fb88a0c0f63b38b28e384755e940ecf690f1a1aeba19e65b211c345df973b4ee`
- Source content SHA-256: `9a32a23ecd00e6aedbe1394d76660d4c56a03c2b6d033dc6242afc3f8285a42f`
- Both notebook copies SHA-256: `20b520713162f644d9a4e362e5fb07d7e1a4d0a4f4aaa679c0d89b468640da01`
- Base Git commit: `756d8942b416fe01e56725723d944730316f09c3` (74 exact allowlisted source files).

The capsule was restored to a fresh temporary checkout, every member checked
against its manifest and the live source, and independently regenerated with an
identical archive hash. Notebook cells compile, have no saved execution outputs,
and retain disabled full-training, frozen-test and export switches. The frozen
release verify command passed checksum/BIO/split-isolation validation; all 11
protected source/config/lock hashes and both Language ID model hashes match.
