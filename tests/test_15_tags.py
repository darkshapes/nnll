import pytest
from datetime import datetime
from nnll_15.constants import LibType
from nnll_15 import RegistryEntry


# Test cases
@pytest.fixture
def registry_entry_ollama():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=["mllama", "llava", "text"],
        library=LibType.OLLAMA,
        timestamp=int(datetime.now().timestamp()),
    )


@pytest.fixture
def registry_entry_hub():
    return RegistryEntry(
        model="🤡",
        size=512,
        tags=["text-generation", "image-to-text", "text-to-speech", "text"],
        library=LibType.HUB,
        timestamp=int(datetime.now().timestamp()),
    )


def test_ollama_available_tasks(registry_entry_ollama: RegistryEntry):
    assert registry_entry_ollama.available_tasks == [("text", "text")]


def test_hub_available_tasks(registry_entry_hub: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [("text", "text"), ("text", "text"), ("image", "text"), ("text", "speech")]
    assert set(registry_entry_hub.available_tasks) == set(expected_tasks)
    assert registry_entry_hub.available_tasks.count(("text", "text")) == 1


@pytest.fixture
def registry_entry_hub_extra():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=["speech-translation", "speech-summarization", "automatic-speech-recognition", "text-to-speech", "video generation"],
        library=LibType.HUB,
        timestamp=int(datetime.now().timestamp()),
    )


def test_hub_extra_tasks(registry_entry_hub_extra: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [
        ("speech", "text"),
        ("text", "speech"),
        ("text", "video"),
    ]
    assert set(registry_entry_hub_extra.available_tasks) == set(expected_tasks)


@pytest.fixture
def registry_entry_cortex():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=["speech-translation", "speech-summarization", "automatic-speech-recognition", "text-to-speech", "video generation"],
        library=LibType.CORTEX,
        timestamp=int(datetime.now().timestamp()),
    )


def test_cortex_tasks(registry_entry_cortex: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [
        ("text", "text"),
    ]
    assert set(registry_entry_cortex.available_tasks) == set(expected_tasks)


@pytest.fixture
def registry_entry_vllm():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=["text", "vision"],
        library=LibType.VLLM,
        timestamp=int(datetime.now().timestamp()),
    )


def test_vllm_tasks(registry_entry_vllm: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [
        ("text", "text"),
        ("image", "text"),
    ]
    assert set(registry_entry_vllm.available_tasks) == set(expected_tasks)


@pytest.fixture
def registry_entry_llamafile():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=[],
        library=LibType.LLAMAFILE,
        timestamp=int(datetime.now().timestamp()),
    )


def test_llamafile_tasks(registry_entry_llamafile: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [
        ("text", "text"),
    ]
    assert set(registry_entry_llamafile.available_tasks) == set(expected_tasks)


@pytest.fixture
def registry_entry_lm_studio():
    return RegistryEntry(
        model="🤡",
        size=1024,
        tags=["text", "llm"],
        library=LibType.LM_STUDIO,
        timestamp=int(datetime.now().timestamp()),
    )


def test_lm_studio_tasks(registry_entry_lm_studio: RegistryEntry):  # ensure that the system does not duplicate entries
    expected_tasks = [
        ("text", "text"),
    ]
    assert set(registry_entry_lm_studio.available_tasks) == set(expected_tasks)
    assert registry_entry_lm_studio.available_tasks.count(("text", "text")) == 1
