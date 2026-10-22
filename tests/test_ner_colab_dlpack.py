"""DLPack contract, captured imports and real bootstrap restart regressions."""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from kurdish_nlp.ner.colab_dlpack import install_cupy_dlpack_compatibility

REPO = Path(__file__).resolve().parents[1]


class FakeArray:
    def toDlpack(self):
        raise AssertionError("Deprecated capsule API must never be called")

    def __dlpack__(self):
        return "modern-protocol"


class FakeTensor:
    def __init__(self):
        self.device = "cuda:0"
        self.requires_grad = False

    def to(self, device):
        self.device = device
        return self

    def requires_grad_(self):
        self.requires_grad = True
        return self


def fake_stack(monkeypatch):
    """Install isolated modules so contract tests also run without ML packages."""
    for name in list(sys.modules):
        if name.split(".")[0] in {"thinc", "spacy_transformers", "cupy", "torch"}:
            monkeypatch.delitem(sys.modules, name)
    modules = {}
    for name in (
        "thinc",
        "thinc.util",
        "thinc.api",
        "cupy",
        "torch",
        "spacy_transformers",
        "spacy_transformers.data_classes",
        "spacy_transformers.layers",
        "spacy_transformers.layers.transformer_model",
        "spacy_transformers.another_consumer",
    ):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    calls = []

    def original(xp_tensor: object, requires_grad: bool = False, device=None):
        calls.append((xp_tensor, requires_grad, device))
        return "original-result"

    def from_dlpack(array):
        calls.append(array)
        assert array.__dlpack__() == "modern-protocol"
        return FakeTensor()

    modules["thinc"].__version__ = "8.3.11"
    modules["cupy"].__version__ = "14.2.0"
    modules["cupy"].ndarray = FakeArray
    modules["torch"].utils = SimpleNamespace(
        dlpack=SimpleNamespace(from_dlpack=from_dlpack)
    )
    for name in (
        "thinc.util",
        "thinc.api",
        "spacy_transformers.data_classes",
        "spacy_transformers.layers.transformer_model",
    ):
        modules[name].xp2torch = original
    modules["thinc.util"].get_torch_default_device = lambda: "cpu"
    modules["spacy_transformers.another_consumer"].conversion_alias = original
    return SimpleNamespace(modules=modules, original=original, calls=calls)


@pytest.fixture
def stack(monkeypatch):
    return fake_stack(monkeypatch)


@pytest.mark.parametrize(
    "device, expected", [(None, "cpu"), ("cpu", "cpu"), ("cuda:3", "cuda:3")]
)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_protocol_device_and_requires_grad_contract(
    stack, device, expected, requires_grad
):
    install_cupy_dlpack_compatibility()
    adapter = stack.modules["thinc.util"].xp2torch
    array = FakeArray()
    # Positional arguments remain supported, as in the original signature.
    result = adapter(array, requires_grad, device)
    assert result.device == expected
    assert result.requires_grad is requires_grad
    assert stack.calls == [array]
    assert inspect.signature(adapter) == inspect.signature(stack.original)
    assert adapter.__wrapped__ is stack.original


def test_non_cupy_inputs_are_delegated_unchanged(stack):
    install_cupy_dlpack_compatibility()
    adapter = stack.modules["thinc.util"].xp2torch
    for value in (object(), [1, 2], None):
        assert adapter(value, True, "cpu") == "original-result"
        assert stack.calls[-1] == (value, True, "cpu")


def test_idempotence_all_captured_references_aliases_and_future_imports(
    stack, monkeypatch
):
    outsider = ModuleType("unrelated_consumer")
    outsider.xp2torch = stack.original
    monkeypatch.setitem(sys.modules, outsider.__name__, outsider)
    first = install_cupy_dlpack_compatibility()
    adapter = stack.modules["thinc.util"].xp2torch
    assert install_cupy_dlpack_compatibility() == first
    assert stack.modules["thinc.util"].xp2torch is adapter
    assert outsider.xp2torch is stack.original
    for name in (
        "thinc.api",
        "spacy_transformers.data_classes",
        "spacy_transformers.layers.transformer_model",
    ):
        assert stack.modules[name].xp2torch is adapter
    assert (
        stack.modules["spacy_transformers.another_consumer"].conversion_alias is adapter
    )
    from thinc.api import xp2torch

    assert xp2torch is adapter
    # A late-loaded module with a stale alias is repaired without nesting wrappers.
    late = ModuleType("spacy_transformers.late")
    late.old_alias = stack.original
    monkeypatch.setitem(sys.modules, late.__name__, late)
    assert (
        "spacy_transformers.late.old_alias"
        in install_cupy_dlpack_compatibility()["references"]
    )
    assert late.old_alias is adapter
    assert adapter.__wrapped__ is stack.original


@pytest.mark.parametrize(
    "package, version",
    [("cupy", "13.4.1"), ("cupy", "15.0.0"), ("thinc", "8.3.10"), ("thinc", "8.3.12")],
)
def test_unaffected_versions_do_not_patch(stack, package, version):
    stack.modules[package].__version__ = version
    assert not install_cupy_dlpack_compatibility()["active"]
    assert stack.modules["thinc.util"].xp2torch is stack.original
    assert stack.modules["thinc.api"].xp2torch is stack.original


def test_missing_cupy_is_a_noop_but_broken_installation_is_not_hidden(
    stack, monkeypatch
):
    import kurdish_nlp.ner.colab_dlpack as compat

    def missing(name):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'", name="cupy")
        return importlib.import_module(name)

    monkeypatch.setattr(compat, "import_module", missing)
    assert install_cupy_dlpack_compatibility() == {
        "active": False,
        "reason": "cupy_absent",
    }

    def broken(name):
        if name == "cupy":
            raise ModuleNotFoundError("Broken CUDA installation", name="cupy_backend")
        return importlib.import_module(name)

    monkeypatch.setattr(compat, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="Broken CUDA"):
        install_cupy_dlpack_compatibility()


def test_importing_adapter_does_not_import_or_patch_ml_packages():
    code = """
import sys
from kurdish_nlp.ner import colab_dlpack
assert not any(n.split('.')[0] in {'torch', 'cupy', 'thinc', 'spacy_transformers'} for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=_subprocess_env(), check=True
    )


def _subprocess_env(*paths):
    return dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, (*paths, REPO / "src")))
    )


def fake_gpu_runtime(monkeypatch):
    """CPU simulation of environment probes; never represents a CUDA test pass."""
    from kurdish_nlp.ner import colab, cuda_runtime

    stack = fake_stack(monkeypatch)
    cupy = stack.modules["cupy"]
    cupy.arange = lambda n: n
    cupy.sum = lambda value: SimpleNamespace(get=lambda: 6)
    torch = stack.modules["torch"]
    torch.__version__ = "2.6.0+cu124"
    torch.version = SimpleNamespace(cuda="12.4")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: SimpleNamespace(
            name="Tesla T4", total_memory=15 * 1024**3
        ),
    )
    api = stack.modules["thinc.api"]
    api.get_current_ops = lambda: SimpleNamespace(xp=cupy)
    api.set_gpu_allocator = lambda name: None
    spacy = ModuleType("spacy")
    spacy.__version__ = "3.8.7"
    spacy.require_gpu = lambda i: None
    monkeypatch.setitem(sys.modules, "spacy", spacy)
    psutil = ModuleType("psutil")
    psutil.virtual_memory = lambda: SimpleNamespace(available=1024)
    psutil.Process = lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=512)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monkeypatch.setattr(colab.platform, "system", lambda: "Linux")
    monkeypatch.setattr(colab.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(colab.platform, "platform", lambda: "simulated Linux")
    monkeypatch.setattr(colab.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        colab.subprocess, "check_output", lambda *a, **k: "Tesla T4, simulated"
    )
    return stack, cuda_runtime


def test_cli_installs_in_final_process_after_real_bootstrap_execve(tmp_path):
    # sitecustomize recreates the simulated dependencies independently on BOTH
    # interpreter starts. The real bootstrap and real CLI dispatch run unchanged.
    # No mock of os.execve: the initial interpreter is actually replaced.
    libraries = tmp_path / "site/nvidia/cuda_runtime/lib"
    libraries.mkdir(parents=True)
    include = libraries.parent / "include"
    include.mkdir()
    (include / "cuda_runtime.h").touch()
    trace = tmp_path / "starts.jsonl"
    startup = tmp_path / "sitecustomize.py"
    startup.write_text(f"""
import json, os, runpy
from pathlib import Path
from pytest import MonkeyPatch
helpers = runpy.run_path({str(Path(__file__).resolve())!r})
patches = MonkeyPatch()
stack, runtime = helpers["fake_gpu_runtime"](patches)
assert stack.modules["thinc.util"].xp2torch is stack.original
with open({str(trace)!r}, "a") as stream:
    stream.write(json.dumps({{"libraries": os.environ.get("LD_LIBRARY_PATH"), "active_at_start": False}}) + "\\n")
original_environment = runtime.runtime_environment
runtime.runtime_environment = lambda: original_environment(roots=[Path({str(tmp_path / "site")!r})])
""")
    env = _subprocess_env(tmp_path)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("CUDA_PATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "kurdish_nlp.ner.colab", "environment"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    starts = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(starts) == 2, result.stderr
    assert starts[0]["libraries"] is None
    assert str(libraries) in starts[1]["libraries"]
    report = json.loads(result.stdout)
    assert report["dlpack_compatibility"]["active"] is True
    assert (
        "spacy_transformers.data_classes.xp2torch"
        in report["dlpack_compatibility"]["references"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["environment"],
        [
            "prepare",
            "--run",
            "unused",
            "--drive",
            "unused-drive",
            "--encoder",
            "unused-encoder",
        ],
        ["preflight", "--run", "unused", "--stage", "smoke"],
        ["preflight", "--run", "unused", "--stage", "stress"],
        ["preflight", "--run", "unused", "--stage", "benchmark"],
        ["gate", "--run", "unused"],
        ["train", "--run", "unused"],
        ["export", "--run", "unused", "--output", "unused-export"],
    ],
)
def test_all_cuda_commands_install_before_run_or_pipeline_work(monkeypatch, arguments):
    from kurdish_nlp.ner import colab

    stack, runtime = fake_gpu_runtime(monkeypatch)

    # Supply imports only; fail if a command tries to initialize/train a model.
    def forbidden(*args, **kwargs):
        raise AssertionError("Pipeline work before environment setup")

    for name, attributes in {
        "spacy.util": ["load_config", "registry"],
        "spacy.tokens": ["DocBin"],
        "spacy.training.initialize": ["init_nlp"],
        "spacy.training.loop": ["train"],
    }.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, forbidden)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runtime, "runtime_environment", lambda: dict(os.environ))
    monkeypatch.setattr(sys, "argv", ["colab", *arguments])
    real_environment = colab.gpu_environment

    class ReachedEnvironment(Exception):
        pass

    def checked_environment():
        report = real_environment()
        assert report["dlpack_compatibility"]["active"]
        assert (
            stack.modules["spacy_transformers.data_classes"].xp2torch
            is not stack.original
        )
        raise ReachedEnvironment

    monkeypatch.setattr(colab, "gpu_environment", checked_environment)
    with pytest.raises(ReachedEnvironment):
        colab.main()


def test_real_spacy_transformers_consumers_and_numpy_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    thinc = pytest.importorskip("thinc")
    transformers = pytest.importorskip("spacy_transformers")
    if thinc.__version__ != "8.3.11":
        pytest.skip("Regression targets Thinc 8.3.11")
    util = importlib.import_module("thinc.util")
    original = util.xp2torch
    # Audit every source file in the installed pinned package, rather than
    # assuming the two known consumers are exhaustive. Import any consumer.
    consumers = set()
    for path in Path(transformers.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "thinc.api",
                "thinc.util",
            }:
                for alias in node.names:
                    if alias.name == "xp2torch":
                        relative = path.relative_to(
                            Path(transformers.__file__).parent
                        ).with_suffix("")
                        name = "spacy_transformers." + ".".join(relative.parts)
                        consumers.add((name, alias.asname or alias.name))
                        importlib.import_module(name)
    assert consumers
    # Register restoration of every affected real reference before installation.
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith(("thinc.", "spacy_transformers.")):
            for key, value in list(vars(module).items()):
                if value is original:
                    monkeypatch.setattr(module, key, original)
    cupy = ModuleType("cupy")
    cupy.__version__, cupy.ndarray = "14.2.0", FakeArray
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(util, "get_torch_default_device", lambda: torch.device("cpu"))
    assert install_cupy_dlpack_compatibility()["active"]
    for name, attribute in consumers:
        assert getattr(sys.modules[name], attribute) is util.xp2torch
    array = numpy.arange(4, dtype="float32")
    tensor = util.xp2torch(array, requires_grad=True)
    assert tensor.device == torch.device("cpu")
    assert tensor.requires_grad
    assert tensor.data_ptr() == array.ctypes.data
    (tensor * 2).sum().backward()
    assert torch.equal(tensor.grad, torch.full((4,), 2.0))

    # Exercise the actual torch consumer on CPU, with a protocol provider whose
    # legacy API deliberately fails. This is a stand-in, not a real CuPy test.
    class ProtocolArray:
        def toDlpack(self):
            raise AssertionError("legacy API")

        def __dlpack__(self, *args, **kwargs):
            return array.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return array.__dlpack_device__()

    cupy.ndarray = ProtocolArray
    converted = util.xp2torch(
        ProtocolArray(), requires_grad=True, device=torch.device("cpu")
    )
    assert converted.requires_grad
    assert converted.data_ptr() == array.ctypes.data
    assert torch.equal(converted, tensor)


@pytest.fixture
def cuda_conversion():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA GPU unavailable; real CuPy DLPack test not run")
    cupy = pytest.importorskip("cupy")
    thinc = pytest.importorskip("thinc")
    if thinc.__version__ != "8.3.11" or cupy.__version__.split(".")[0] != "14":
        pytest.skip("Adapter regression requires CuPy 14 and Thinc 8.3.11")
    assert install_cupy_dlpack_compatibility()["active"]
    return cupy, torch, importlib.import_module("thinc.util")


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cuda_cupy_dlpack_conversion_device_and_grad(
    cuda_conversion, requires_grad, device
):
    cupy, torch, util = cuda_conversion
    with cupy.cuda.Device(0):
        array = cupy.arange(12, dtype=cupy.float32).reshape(3, 4)[:, ::2]
        tensor = util.xp2torch(
            array, requires_grad=requires_grad, device=torch.device(device)
        )
        assert tensor.device == torch.device(device)
        assert tensor.requires_grad is requires_grad
        expected = torch.tensor([[0, 2], [4, 6], [8, 10]], dtype=torch.float32)
        assert torch.equal(tensor.detach().cpu(), expected)
        if device == "cuda:0":
            assert tensor.data_ptr() == array.data.ptr
        if requires_grad:
            tensor.sum().backward()
            assert torch.equal(tensor.grad, torch.ones_like(tensor))


def test_cuda_default_device_is_thinc_default(cuda_conversion, monkeypatch):
    cupy, torch, util = cuda_conversion
    monkeypatch.setattr(util, "get_torch_default_device", lambda: torch.device("cpu"))
    assert util.xp2torch(cupy.ones(2, dtype=cupy.float32)).device.type == "cpu"


def test_cuda_protocol_coordinates_nondefault_streams(cuda_conversion):
    cupy, torch, util = cuda_conversion
    producer = cupy.cuda.Stream(non_blocking=True)
    consumer = torch.cuda.Stream(device=0)
    with cupy.cuda.Device(0), producer:
        array = cupy.arange(4096, dtype=cupy.float32) * 3
        with torch.cuda.stream(consumer):
            tensor = util.xp2torch(array, device=torch.device("cuda:0"))
            result = tensor.clone()
    consumer.synchronize()
    assert torch.equal(result.cpu(), torch.arange(4096, dtype=torch.float32) * 3)
