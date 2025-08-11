# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import types
from typing import OrderedDict
import pytest
import sys

from nnll.model_detect.tasks import AutoPkg


class DummyDiffusersTaskMap(OrderedDict):
    """Mimic a SUPPORTED_TASKS_MAPPINGS entry."""

    pass


def make_dummy_diffusers_modules(monkeypatch):
    """Create minimal diffusers package structure required by AutoPkg.
    ie diffusers.pipelines.auto_pipeline"""
    auto_pipeline = types.SimpleNamespace()
    task_map_norm = DummyDiffusersTaskMap()
    task_map_i2i = DummyDiffusersTaskMap()

    #
    class CoronaPipeline:
        """Fake model code mappped to fake pipe class"""

        __name__ = "CoronaPipeline"

    class CoronaImg2ImgPipeline:
        __name__ = "CoronaImg2ImgPipeline"

    task_map_norm["corona-model"] = CoronaPipeline
    task_map_i2i["corona-model"] = CoronaImg2ImgPipeline
    auto_pipeline.SUPPORTED_TASKS_MAPPINGS = [
        task_map_norm,
        task_map_i2i,
    ]

    def _get_task_class(task_map, class_name, _):
        """Return a dummy class if class_name matches"""

        return task_map.get("corona-model")
        # return None

    auto_pipeline._get_task_class = _get_task_class
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.auto_pipeline", auto_pipeline)


def make_dummy_transformers_modules(monkeypatch):
    """Create minimal transformers package structure required by AutoPkg."""
    utils_fx = types.SimpleNamespace()

    def _generate_supported_model_class_names(code_name):
        """Return a list based on the code_name"""
        return [f"{code_name}_TaskA", f"{code_name}_TaskB"]

    utils_fx._generate_supported_model_class_names = _generate_supported_model_class_names
    monkeypatch.setitem(sys.modules, "transformers.utils.fx", utils_fx)

    # nnll.metadata.helpers.make_callable stub
    helpers = types.SimpleNamespace()

    def make_callable(name, pkg):
        # Return a dummy class with __module__ and __all__
        class Dummy:
            __module__ = f"{pkg}.dummy_module"

        Dummy.__all__ = ["DummyClass"]
        return Dummy

    helpers.make_callable = make_callable
    monkeypatch.setitem(sys.modules, "nnll.metadata.helpers", helpers)


def make_dummy_nnll_modules(monkeypatch):
    """Create minimal nnll package structure required by AutoPkg."""
    # nnll.tensor_pipe.deconstructors.get_code_names
    deconstructors = types.SimpleNamespace()

    def get_code_names(class_name, package_name):
        """Return a deterministic code name"""
        return f"{class_name}_code"

    deconstructors.get_code_names = get_code_names
    monkeypatch.setitem(sys.modules, "nnll.tensor_pipe.deconstructors", deconstructors)

    # nnll.mir.tag.make_scheduler_tag
    mir_tag = types.SimpleNamespace()

    def make_scheduler_tag(class_name):
        """Return dummy series and component"""
        return ("scheduler_series", "scheduler_component")

    mir_tag.make_scheduler_tag = make_scheduler_tag
    monkeypatch.setitem(sys.modules, "nnll.mir.tag", mir_tag)


class DummyMIRDatabase:
    """A very small in‑memory stand‑in for the real MIRDatabase."""

    def __init__(self):
        """# DB Structure: {series: {compatibility: {field_name: {"0": pkg:{  : ...}}}}}"""
        self.database = {}

    def add_entry(self, series, compatibility, field_name, pkg_tree):
        self.database.setdefault(series, {})
        self.database[series].setdefault(compatibility, {})
        self.database[series][compatibility][field_name] = {"0": pkg_tree}

    def find_tag(self, *, field, target, sub_field=None, domain=None):
        """Simplified: return a fake tag if target contains "Known"""
        tree = {
            "IPNDMScheduler": ["ops.scheduler.dummy", "ipndmscheduler"],
            "EQvae": ["info.vae.dummy", "AutoencoderKL"],
            "DummyOther": ["info.dummy.OtherClass", "*"],
            "CLIPTokenizer": [
                "info.encoder.tokenizer",
                "CLIPDummy",
            ],
        }
        return tree.get(target)


@pytest.fixture(autouse=True)
def stub_external_modules(monkeypatch):
    """Patch all external imports used by AutoPkg."""
    from nnll.monitor import file

    make_dummy_diffusers_modules(monkeypatch)
    make_dummy_transformers_modules(monkeypatch)
    make_dummy_nnll_modules(monkeypatch)

    monkeypatch.setattr(file, "dbuq", lambda *args, **kwargs: None)  # Stub dbuq (debug print) to avoid noisy output


def test_show_diffusers_tasks():
    tasks = AutoPkg.show_diffusers_tasks(
        code_name="corona-model",
        class_name="CoronaModel",
    )
    assert "CoronaPipeline" in tasks
    assert "CoronaImg2ImgPipeline" in tasks


def test_show_transformers_tasks_by_class():
    """When code_name is None, make_callable returns a dummy with __all__"""
    tasks = AutoPkg.show_transformers_tasks(class_name="AnyClass")
    assert tasks == ["DummyClass"]  # from Dummy.__all__


def test_show_transformers_tasks_by_code():
    tasks = AutoPkg.show_transformers_tasks(code_name="bert")
    assert tasks == ["bert_TaskA", "bert_TaskB"]


@pytest.mark.asyncio
async def test_trace_tasks_filters_and_sorts():
    """Package entry should be processed (not in `skip_auto` list)
    show_transformers_tasks should return ["DummyClass"]; no snip words, so unchanged"""
    ap = AutoPkg()

    pkg_tree = {"transformers": "SomeModel"}
    tasks = await ap.trace_tasks(pkg_tree)

    assert tasks == ["DummyClass"]


async def test_trace_finds_map_with_code_name():
    ap = AutoPkg()
    pkg_tree = {"diffusers": "CoronaPipeline"}
    tasks = await ap.trace_tasks(pkg_tree)
    assert tasks == [
        "CoronaImg2ImgPipeline",
        "CoronaPipeline",
    ]


@pytest.mark.asyncio
async def test_mflux_path_returns_static_list():
    ap = AutoPkg()
    pkg_tree = {"mflux": "any"}
    tasks = await ap.trace_tasks(pkg_tree)
    assert tasks == ap.mflux_tasks


@pytest.mark.asyncio
async def test_skip_automode_return_none():
    ap = AutoPkg()
    pkg_tree = {"transformers": "AutoModel"}
    tasks = await ap.trace_tasks(pkg_tree)
    assert tasks is None


@pytest.mark.asyncio
async def test_hyperlink_and_tag_class():
    """Populate a known tag for a scheduler class\n"""
    ap = AutoPkg()
    mir_db = DummyMIRDatabase()

    mir_db.add_entry(
        series="ops.scheduler.scheduler_series",
        compatibility="any",
        field_name="pkg",
        pkg_tree={"diffusers": "IPNDMScheduler"},
    )

    class IPNDMScheduler:
        __name__ = "IPNDM"
        __module__ = "schedulers.ipndm.IPNDMScheduler"

    class EQvae:
        __name__ = "EQ-VAE"
        __module__ = "autoencoders.AutoencoderKL"

    class DummyOther:
        __name__ = "OtherClass"
        __module__ = "other_pkg.OtherClass"

    class CLIPTokenizer:
        __name__ = "CLIPTokenizer"
        __module__ = "tokenizers.CLIPTokenizer"

    pipe_args = {
        "scheduler": IPNDMScheduler,
        "vae": EQvae,
        "unrelated": DummyOther,
        "tokenizer": CLIPTokenizer,  # should be mapped to encoder tokenizers
    }

    links = await ap.hyperlink_to_mir(pipe_args, "info.test_series", mir_db)

    assert "scheduler" in links["pipe_names"]  # Scheduler should be resolved via make_scheduler_tag -> find_tag fallback\n
    scheduler_tag = links["pipe_names"]["scheduler"]
    assert scheduler_tag == ["ops.scheduler.dummy", "ipndmscheduler"]

    assert "vae" in links["pipe_names"]  # VAE should be resolved via find_tag (since not in dummy DB)
    assert links["pipe_names"]["vae"] == ["info.vae.dummy", "AutoencoderKL"]

    assert links["pipe_names"]["unrelated"] == ["info.dummy.OtherClass", "*"]  # Unrelated should just return the class name

    assert links["pipe_names"]["tokenizer"] == ["info.encoder.tokenizer", "test_series"]  # Tokenizer role is *special‑cased*


@pytest.mark.asyncio
async def test_detect_tasks_and_pipes():
    ap = AutoPkg()
    mir_db = DummyMIRDatabase()

    mir_db.add_entry(
        series="info.art.modelA",  # Add a series that passes the skip filters
        compatibility="compat1",
        field_name="pkg",
        pkg_tree={"transformers": "SomeModel"},
    )

    mir_db.add_entry(
        series="info.lora.modelB",  # Add a series (".lora") that should be ignored (skip_series)
        compatibility="compat2",
        field_name="pkg",
        pkg_tree={"transformers": "SomeModel"},
    )

    async def fake_trace_tasks(pkg_tree):
        """Patch trace_tasks to return a predictable list"""
        return ["TaskX", "TaskY"]

    ap.trace_tasks = fake_trace_tasks

    tasks = await ap.detect_tasks(mir_db)
    print(tasks)
    assert any("modelA" in series for prefix, series, _ in tasks)
    assert not any("lora" in prefix for prefix, series, _ in tasks)

    class DummyPipe:
        """diffusers entry with a pipe class for detect_pipes"""

        def __init__(arg1: int, arg2: str):
            """Exists purely for annotation reading!"""
            pass

    def fake_make_callable(name, pkg):
        """Stub make_callable to return DummyPipe for the module name"""
        return DummyPipe

    # Monkeypatch the helper used inside detect_pipes
    from nnll.metadata import helpers

    helpers.make_callable = fake_make_callable

    mir_db.add_entry(
        series="info.vit.modelC",
        compatibility="compat3",
        field_name="pkg",
        pkg_tree={"diffusers": "DummyPipe"},
    )

    async def fake_hyperlink(pipe_args, series, db):
        """Patch hyperlink_to_mir to return a simple marker"""
        return {"pipe_names": {"dummy": ["OK"]}}

    ap.hyperlink_to_mir = fake_hyperlink

    pipes = await ap.detect_pipes(mir_db)  # Should contain the non‑skipped diffusers entry
    assert any("modelC" in series for prefix, series, _ in pipes)
    for _, _, data in pipes:  # Ensure the returned structure matches the fake hyperlink output
        assert data["compat3"]["pipe_names"]["dummy"] == ["OK"]
