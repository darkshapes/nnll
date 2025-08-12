# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call
from hashlib import sha256
import pytest

EXAMPLE_DB = {
    ("pkg", "CLIPTextModelWithProjection"): {"repo": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"},  # info.vit.clip-vit-g-14-laion-s-b
    ("layer_b3", "d754db276f2d89d2808abb7086b3b8eccee43ac521c128d21a071f3a631474a8"): "layer_b3_tag_1",
    ("file_256", sha256(b"dummy").hexdigest()): "file_256_tag_1",
    ("pkg", "AutoencoderKL"): {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5"},  # info.unet.stable-diffusion-v1-5
    ("layer_b3", "82e2dc440a23d78bb91df8c9fce069a8512da51f8f54ea29e3431f545808171e"): "sd_layer_b3_tag",
    ("file_256", sha256(b"sd1dummy").hexdigest()): "sd_file_256_tag",
}


def fake_find_tag(field: str, target: str, sub_field: str | None = None):
    """Simple stand‑in for ``MIRDatabase.find_tag``."""
    key = (field, target)
    return EXAMPLE_DB.get(key)


@pytest.fixture
def mock_nnll_modules(monkeypatch):
    """Replace heavy nnll imports with lightweight fakes.
    The fixture returns a dict with the objects that were injected,
    allowing tests to introspect call history."""

    class DummyReader:
        pass

    monkeypatch.setattr(
        "nnll.metadata.model_tags.ReadModelTags",
        lambda: DummyReader(),
        raising=False,
    )

    class DummyMIRDatabase:
        def __init__(self):
            self.database = {}
            self.find_tag = MagicMock(side_effect=fake_find_tag)

    monkeypatch.setattr(
        "nnll.mir.maid.MIRDatabase",
        DummyMIRDatabase,
        raising=False,
    )

    async def dummy_hash_layers_or_files(*, path_named, layer, b3, unsafe):
        """Ddeterministic mapping mimicry of hash_layers_or_files.
        ``path_named`` key returns fake hashes. Smol."""

        if "clip-vit-g-14" in path_named:
            return {
                ("file1.bin", "ca18e0c67c1ef1e64cac22926266765b60688f692307ecc06283d987c5768134"): None,
                ("file2.bin", "d754db276f2d89d2808abb7086b3b8eccee43ac521c128d21a071f3a631474a8"): None,
            }
        if "stable-diffusion-v1-5" in path_named:
            return {
                ("sd_file.bin", "0b204ad0cae549e0a7e298d803d57e36363760dec71c63109c1da3e1147ec520"): None,
                ("sd_layer.bin", "82e2dc440a23d78bb91df8c9fce069a8512da51f8f54ea29e3431f545808171e"): None,
            }
        return {}

    monkeypatch.setattr(
        "nnll.integrity.hash_256.hash_layers_or_files",
        dummy_hash_layers_or_files,
        raising=False,
    )

    async def dummy_get_hub_path(file_name: str):
        """ "Return a temporary directory that pretends to be the hub cache."""
        return str(Path("/tmp/hub_cache") / file_name)

    monkeypatch.setattr(
        "nnll.download.hub_cache.get_hub_path",
        dummy_get_hub_path,
        raising=False,
    )

    def dummy_class_to_mir_tag(mir_db, name):
        """Very small mapping – just echo the name for visibility."""
        return f"class_tag:{name}" if name else None

    monkeypatch.setattr(
        "nnll.mir.tag.class_to_mir_tag",
        dummy_class_to_mir_tag,
        raising=False,
    )

    monkeypatch.setattr(
        "nnll.monitor.file.dbuq",
        lambda *_: None,
        raising=False,
    )
    monkeypatch.setattr(
        "nnll.monitor.file.dbug",
        lambda *_: None,
        raising=False,
    )

    return {
        "MIRDatabase": DummyMIRDatabase,
        "hash_layers_or_files": dummy_hash_layers_or_files,
        "get_hub_path": dummy_get_hub_path,
    }


@pytest.fixture
def model_identity(mock_nnll_modules):
    """Construct a fresh ``ModelIdentity`` instance for each test.
    The heavy imports have already been monkey‑patched.
    Import inside the fixture so that the monkeypatches are already active."""
    from nnll.model_detect.identity import ModelIdentity

    return ModelIdentity()


@pytest.mark.asyncio
async def test_label_model_class_returns_none_when_no_match(model_identity):
    """If ``find_tag`` never matches, the method should return ``None``.
    we verify it was invoked."""
    result = await model_identity.label_model_class("nonexistent-model")
    assert result is None
    assert model_identity.find_tag.called


@pytest.mark.asyncio
async def test_label_model_hub_ordering(model_identity):
    """
    For ``cue_type='HUB'`` the order of attempts is:
    1. label_model_layers
    2. find_tag(repo)
    3. class_to_mir_tag(base_model)
    4. class_to_mir_tag(repo_folder)
    5. label_model_class
    The test forces the first step to succeed.
    """

    async def fake_layers(*_, **__):
        """Force ``label_model_layers`` to return a known tag."""
        return ["layer_tag_hub"]

    model_identity.label_model_layers = AsyncMock(side_effect=fake_layers)

    result = await model_identity.label_model(
        repo_id="any-repo",
        base_model="info.vit.clip-vit-g-14-laion-s-b",
        cue_type="HUB",
    )
    assert result == ["layer_tag_hub"]

    # with pytest.raises(AssertionError) as exc_info: # To ensure later strategies were never called (higher precision identity check on)
    model_identity.find_tag.assert_called_once()  # This gets called first now
    # assert type(exc_info.value) is AssertionError

    # model_identity.label_model_class.__wrapped__
    # print(pytest.CallInfo(model_identity.label_model_class.__wrapped__))


@pytest.mark.asyncio
async def test_label_model_ollama_falls_back_to_class_tag(model_identity):
    """
    For ``cue_type='OLLAMA'`` the order is different.
    This test makes the first two strategies return ``None`` and checks that
    label_model = None
    find_ tag = None
    ``class_to_mir_tag`` is used.
    """
    model_identity.label_model_class = AsyncMock(return_value=None)
    result = await model_identity.label_model(
        repo_id="some-repo",
        base_model="info.vit.clip-vit-g-14-laion-s-b",
        cue_type="OLLAMA",
    )
    assert result == ["class_tag:info.vit.clip-vit-g-14-laion-s-b"]


@pytest.mark.asyncio
async def test_get_cache_path_hub(monkeypatch, model_identity):
    """
    When ``repo_obj`` is ``None`` the method should delegate to
    ``nnll.download.hub_cache.get_hub_path``.
    as dummy ``get_hub_path`` returns a predictable path.
    """

    path = await model_identity.get_cache_path("my-model")
    assert path == "/tmp/hub_cache/my-model"


# @pytest.mark.asyncio
# async def test_label_model_class_finds_pkg_tag(model_identity):
#     """
#     ``label_model_class`` should return the first tag found via ``find_tag``.
#     The fake ``find_tag`` knows about the ``CLIPTextModelWithProjection`` entry.
#     Subsequent calls hit the LRU cache – ``find_tag`` should have been called only once
#     """
#     tag = await model_identity.label_model_class("info.vit.clip-vit-g-14-laion-s-b")

#     assert isinstance(tag, dict)
#     assert tag["repo"] == "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"

#     await model_identity.label_model_class("info.vit.clip-vit-g-14-laion-s-b")
#     assert model_identity.find_tag.call_count == 1


# @pytest.mark.asyncio
# async def test_label_model_layers_directory_path(tmp_path, model_identity):
#     """
#     When the cache path points to a directory, ``label_model_layers`` should
#     walk the directory, hash each file and collect matching tags.
#     """
#     model_dir = tmp_path / "clip-vit-g-14"
#     model_dir.mkdir()
#     (model_dir / "file1.bin").write_bytes(b"dummy")
#     (model_dir / "file2.bin").write_bytes(b"dummy")

#     async def fake_get_cache_path(file_name, repo_obj=None):
#         """Monkey‑patch ``get_cache_path`` to return our temporary directory.\n"""
#         return str(model_dir)

#     model_identity.get_cache_path = AsyncMock(side_effect=fake_get_cache_path)

#     tags = await model_identity.label_model_layers(
#         repo_id="info.vit.clip-vit-g-14-laion-s-b",
#         cue_type="HUB",
#     )

#     assert isinstance(tags, list)
#     assert "file_256_tag_1" in tags
#     assert "layer_b3_tag_1" in tags


# @pytest.mark.asyncio
# async def test_label_model_layers_file_path(tmp_path, model_identity):
#     """When the cache path points to a single file, the function should hash the file
#     and return the matching tag (or ``None`` if not found)."""

#     file_path = tmp_path / "sd_file.bin"
#     file_path.write_bytes(b"sd1dummy")

#     async def fake_get_cache_path(file_name, repo_obj=None):
#         """Create a single dummy file.
#         returns a hash that maps to ``sd_file_256_tag``."""
#         return str(file_path)

#     model_identity.get_cache_path = AsyncMock(side_effect=fake_get_cache_path)

#     tags = await model_identity.label_model_layers(
#         repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
#         cue_type="HUB",
#     )
#     assert tags == ["sd_file_256_tag"]


# @pytest.mark.asyncio
# async def test_get_cache_path_tuple_repo_obj(model_identity):
#     """
#     When ``repo_obj`` is a tuple the function extracts the path from the tuple.
#     Simulate the tuple format used by the real code and returns the stripped path.
#     """
#     dummy_tuple = (types.SimpleNamespace(partition=lambda _: ("", "prefix/file.txt", "")),)
#     path = await model_identity.get_cache_path("file.txt", repo_obj=dummy_tuple)
#     assert path == "prefix/file.txt"


# @pytest.mark.asyncio
# async def test_get_cache_path_with_revisions(monkeypatch, model_identity):
#     """
#     When ``repo_obj`` has a ``revisions`` attribute the function should scan
#     the first revision's files for a matching ``file_path``.
#     """

#     class DummyFileInfo:
#         def __init__(self, path):
#             self.file_path = path

#     class DummyRevision:
#         def __init__(self, files):
#             self.files = files

#     dummy_repo = types.SimpleNamespace(revisions=[DummyRevision([DummyFileInfo("some/dir/target-model.bin")])])

#     path = await model_identity.get_cache_path("target-model.bin", repo_obj=dummy_repo)
#     assert path == "some/dir/target-model.bin"

# {
#     "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
#     "repo_type": "model",
#     "repo_path": PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5"),
#     "size_on_disk": 33802192125,
#     "nb_files": 30,
#     "revisions": frozenset({CachedRevisionInfo(commit_hash="451f4fe16113bff5a5d2269ed5ad43b0592e9a14", snapshot_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"), size_on_disk=33802192125, files=frozenset({CachedFileInfo(file_name="config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/55d78924fee13e4220f24320127c5f16284e13b9"), size_on_disk=547, blob_last_accessed=1754506055.8349757, blob_last_modified=1754506056.0292048), CachedFileInfo(file_name="pytorch_model.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/safety_checker/pytorch_model.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/193490b58ef62739077262e833bf091c66c29488058681ac25cf7df3d8190974"), size_on_disk=1216061799, blob_last_accessed=1754505068.6162932, blob_last_modified=1754505169.0597456), CachedFileInfo(file_name="v1-5-pruned.safetensors", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned.safetensors"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/1a189f0be69d6106a48548e7626207dddd7042a418dbf372cefd05e0cdba61b6"), size_on_disk=7703324286, blob_last_accessed=1754505785.3746443, blob_last_modified=1754506210.5227818), CachedFileInfo(file_name="special_tokens_map.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer/special_tokens_map.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/2c2130b544c0c5a72d5d00da071ba130a9800fb2"), size_on_disk=472, blob_last_accessed=1754505140.786112, blob_last_modified=1754505140.9721138), CachedFileInfo(file_name="pytorch_model.fp16.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/text_encoder/pytorch_model.fp16.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/05eee911f195625deeab86f0b22b115d7d8bc3adbfc1404f03557f7e4e6a8fd7"), size_on_disk=246187076, blob_last_accessed=1754505069.0568192, blob_last_modified=1754505157.902884), CachedFileInfo(file_name="diffusion_pytorch_model.fp16.safetensors", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/diffusion_pytorch_model.fp16.safetensors"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c"), size_on_disk=167335342, blob_last_accessed=1754506210.8638008, blob_last_modified=1754506233.4594111), CachedFileInfo(file_name=".gitattributes", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/.gitattributes"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/55d2855c5be698e0572b9f42af95f06bfd5fb002"), size_on_disk=1548, blob_last_accessed=1754505068.6659923, blob_last_modified=1754505068.7288482), CachedFileInfo(file_name="model.fp16.safetensors", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/text_encoder/model.fp16.safetensors"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e"), size_on_disk=246144864, blob_last_accessed=1754505068.821573, blob_last_modified=1754505140.2267406), CachedFileInfo(file_name="tokenizer_config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer/tokenizer_config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/5ba7bf706515bc60487ad0e1816b4929b82542d6"), size_on_disk=806, blob_last_accessed=1754505141.2071974, blob_last_modified=1754505141.3560214), CachedFileInfo(file_name="config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/text_encoder/config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/4d3e873ab5086ad989f407abd50fdce66db8d657"), size_on_disk=617, blob_last_accessed=1754505068.8321602, blob_last_modified=1754505068.8767262), CachedFileInfo(file_name="scheduler_config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/scheduler/scheduler_config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/82d05b0e688d7ea94675678646c427907419346e"), size_on_disk=308, blob_last_accessed=1754505068.9570565, blob_last_modified=1754505069.0050142), CachedFileInfo(file_name="preprocessor_config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/feature_extractor/preprocessor_config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/5294955ff7801083f720b34b55d0f1f51313c5c5"), size_on_disk=342, blob_last_accessed=1754505068.6680932, blob_last_modified=1754505068.7331107), CachedFileInfo(file_name="vocab.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer/vocab.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/469be27c5c010538f845f518c4f5e8574c78f7c8"), size_on_disk=1059962, blob_last_accessed=1754505141.535469, blob_last_modified=1754505141.887503), CachedFileInfo(file_name="v1-5-pruned.ckpt", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned.ckpt"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053"), size_on_disk=7703807346, blob_last_accessed=1754505607.6253524, blob_last_modified=1754506054.1177294), CachedFileInfo(file_name="pytorch_model.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/text_encoder/pytorch_model.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/770a47a9ffdcfda0b05506a7888ed714d06131d60267e6cf52765d61cf59fd67"), size_on_disk=492305335, blob_last_accessed=1754505068.9252634, blob_last_modified=1754505159.761987), CachedFileInfo(file_name="diffusion_pytorch_model.fp16.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/diffusion_pytorch_model.fp16.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/30eb3dc47c90e4a55476332b284b2331774c530edbbb83b70cacdd9e7b91af92"), size_on_disk=1719327893, blob_last_accessed=1754505142.3512044, blob_last_modified=1754505517.734286), CachedFileInfo(file_name="model_index.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/model_index.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/daf7e2e2dfc64fb437a2b44525667111b00cb9fc"), size_on_disk=541, blob_last_accessed=1754505068.667997, blob_last_modified=1754505068.7639651), CachedFileInfo(file_name="config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/safety_checker/config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/5dbd88952e7e521aa665e5052e6db7def3641d03"), size_on_disk=4723, blob_last_accessed=1754505068.6561334, blob_last_modified=1754505068.721496), CachedFileInfo(file_name="diffusion_pytorch_model.safetensors", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/diffusion_pytorch_model.safetensors"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c"), size_on_disk=334643276, blob_last_accessed=1754506214.6440477, blob_last_modified=1754506239.6400182), CachedFileInfo(file_name="diffusion_pytorch_model.fp16.safetensors", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/diffusion_pytorch_model.fp16.safetensors"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21"), size_on_disk=1719125304, blob_last_accessed=1754505158.165412, blob_last_modified=1754505536.8238177), CachedFileInfo(file_name="v1-5-pruned-emaonly.ckpt", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.ckpt"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516"), size_on_disk=4265380512, blob_last_accessed=1754505517.9641025, blob_last_modified=1754505784.8397872), CachedFileInfo(file_name="diffusion_pytorch_model.fp16.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/diffusion_pytorch_model.fp16.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/b7643b3e40b9f128eda5fe174fea73c3ef3903562651fb344a79439709c2e503"), size_on_disk=167405651, blob_last_accessed=1754506145.597145, blob_last_modified=1754506214.478245), CachedFileInfo(file_name="v1-inference.yaml", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-inference.yaml"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/d4effe569e897369918625f9d8be5603a0e6a0d6"), size_on_disk=1873, blob_last_accessed=1754506055.2589412, blob_last_modified=1754506055.432063), CachedFileInfo(file_name="diffusion_pytorch_model.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/diffusion_pytorch_model.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/c7da0e21ba7ea50637bee26e81c220844defdf01aafca02b2c42ecdadb813de4"), size_on_disk=3438354725, blob_last_accessed=1754505142.1722383, blob_last_modified=1754505504.3635552), CachedFileInfo(file_name="diffusion_pytorch_model.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/vae/diffusion_pytorch_model.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5"), size_on_disk=334707217, blob_last_accessed=1754506056.1635058, blob_last_modified=1754506145.3955717), CachedFileInfo(file_name="merges.txt", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer/merges.txt"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/76e821f1b6f0a9709293c3b6b51ed90980b3166b"), size_on_disk=524619, blob_last_accessed=1754505140.3510892, blob_last_modified=1754505140.544956), CachedFileInfo(file_name="README.md", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/README.md"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/2a5ba19664aab2052c4ea484e737dff9aca1aa69"), size_on_disk=14461, blob_last_accessed=1754587331.0938323, blob_last_modified=1754505068.7306516), CachedFileInfo(file_name="config.json", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/config.json"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/1a02ee8abc93e840ffbcb2d68b66ccbcb74b3ab3"), size_on_disk=743, blob_last_accessed=1754505142.0918443, blob_last_modified=1754505142.185155), CachedFileInfo(file_name="diffusion_pytorch_model.non_ema.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/diffusion_pytorch_model.non_ema.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/42bc8b8f3af32866db3c7bb5bcf591ab04438296c2712246d7a640bde5a5ddc1"), size_on_disk=3438366373, blob_last_accessed=1754505159.935079, blob_last_modified=1754505607.008827), CachedFileInfo(file_name="pytorch_model.fp16.bin", file_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/safety_checker/pytorch_model.fp16.bin"), blob_path=PosixPath("/Users/unauthorized/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/blobs/22ba87205445ad5def13e54919b038dcfb7321ec1c3f4b12487d4fba6036125f"), size_on_disk=608103564, blob_last_accessed=1754505068.7760322, blob_last_modified=1754505142.0268857)
#         }), refs=frozenset({
#             "main"
#         }), last_modified=1754506239.6400182)
#     }),
#     "last_accessed": 1754587331.0938323,
#     "last_modified": 1754506239.6400182
# }
