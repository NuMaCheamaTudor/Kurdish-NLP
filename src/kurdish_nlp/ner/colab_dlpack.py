"""Process-local CuPy 14 / Thinc 8.3.11 compatibility for Colab CUDA only.

Importing this module has no side effects. Install after CUDA bootstrap in the
final interpreter, before initializing or loading a transformer pipeline.
"""

from __future__ import annotations

import sys
from functools import wraps
from importlib import import_module


def install_cupy_dlpack_compatibility() -> dict:
    """Prefer CuPy's stream-aware DLPack protocol on the affected stack only.

    Thinc's API re-exports xp2torch, and spaCy Transformers imports it by value.
    Load its production modules, then replace references by identity (including
    aliases) within those two packages. Future imports use the patched API.
    Repeated installation never wraps the adapter again and repairs any stale
    references captured since the previous call. No installed files are edited.
    """
    thinc = import_module("thinc")
    if thinc.__version__ != "8.3.11":
        return {"active": False, "reason": "unaffected_thinc"}
    try:
        cupy = import_module("cupy")
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise
        return {"active": False, "reason": "cupy_absent"}
    if cupy.__version__.split(".")[0] != "14":
        return {"active": False, "reason": "unaffected_cupy"}

    util = import_module("thinc.util")
    import_module("thinc.api")
    # __init__ imports all production consumers in pinned spacy-transformers
    # 1.3.9: data_classes and layers.transformer_model. Scan the loaded package
    # as well so another consumer/alias cannot silently keep the old function.
    import_module("spacy_transformers")
    torch = import_module("torch")
    current = util.xp2torch
    if getattr(current, "_sorani_ner_cupy_dlpack", False):
        adapter, original = current, current.__wrapped__
    else:
        original = current

        @wraps(original)
        def adapter(xp_tensor, requires_grad=False, device=None):
            if not isinstance(xp_tensor, cupy.ndarray):
                return original(xp_tensor, requires_grad=requires_grad, device=device)
            if device is None:
                device = util.get_torch_default_device()
            tensor = torch.utils.dlpack.from_dlpack(xp_tensor)
            tensor = tensor.to(device)
            if requires_grad:
                tensor.requires_grad_()
            return tensor

        adapter._sorani_ner_cupy_dlpack = True

    references = []
    for name, module in list(sys.modules.items()):
        if module is None or not (
            name in {"thinc", "spacy_transformers"}
            or name.startswith(("thinc.", "spacy_transformers."))
        ):
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                setattr(module, attribute, adapter)
            if value is original or value is adapter:
                references.append(f"{name}.{attribute}")
    return {
        "active": True,
        "adapter": "cupy14_thinc8311_dlpack_v1",
        "cupy": cupy.__version__,
        "thinc": thinc.__version__,
        "references": sorted(references),
    }
