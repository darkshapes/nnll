import pytest
from unittest.mock import patch
from nnll.mir.indexers import create_pipe_entry


def root_class(pipe_data):
    """Mock function to simulate sub_classes retrieval."""
    if "unet" in pipe_data:
        return ["unet", "encoder"]
    elif "transformer" in pipe_data:
        return ["transformer", "decoder"]
    else:
        return []


@pytest.fixture
def mock_diffusers():
    with patch("diffusers", autospec=True) as mocked:
        return mocked


def test_create_unet():
    repo_path = "standard_org/standard_repo-prior"
    from diffusers import StableDiffusionPipeline

    class_name = StableDiffusionPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    assert len(result) == 2
    mir_series, prefixed_data = result

    assert mir_series == "info.unet.standard-repo"
    assert "repo" in prefixed_data.get("prior")
    assert prefixed_data["prior"]["repo"] == repo_path
    assert "pkg" in prefixed_data.get("prior")
    assert prefixed_data["prior"]["pkg"][0]["diffusers"] == class_name


def test_create_transformer():
    repo_path = "default_series/default_repo"
    from diffusers import HunyuanDiTPipeline

    class_name = HunyuanDiTPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.dit.default-repo"
    assert prefixed_data["*"]["repo"] == repo_path
    assert prefixed_data["*"]["pkg"][0]["diffusers"] == class_name


def test_create_kandinsky():
    repo_path = "kandinsky_series/kandinsky_repo-v1"
    from diffusers import KandinskyPipeline

    class_name = KandinskyPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.kandinsky-repo-v1"
    assert prefixed_data["*"]["repo"] == repo_path
    assert prefixed_data["*"]["pkg"][0]["diffusers"] == class_name


def test_create_shap_e():
    repo_path = "openai/shap-e_40496"
    from diffusers import ShapEPipeline

    class_name = ShapEPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.shap-e-40496"
    assert prefixed_data["*"]["repo"] == repo_path
    assert prefixed_data["*"]["pkg"][0]["diffusers"] == class_name


def test_create_flux():
    repo_path = "cocktailpeanut/xulf-schnell"
    from diffusers import FluxPipeline

    class_name = FluxPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.dit.xulf-schnell"
    assert prefixed_data["*"]["repo"] == repo_path
    assert prefixed_data["*"]["pkg"][0]["diffusers"] == class_name
    assert 1 in prefixed_data["*"]["pkg"]
    assert prefixed_data["*"]["pkg"][1]["mflux"] == "Flux1"


def test_create_empty():
    repo_path = ""
    from diffusers import StableDiffusionPipeline

    with pytest.raises(TypeError) as exc_info:
        create_pipe_entry(repo_path, StableDiffusionPipeline.__name__)

    assert isinstance(exc_info.value, TypeError)
    assert exc_info.value.args == ("'repo_path'  or 'pipe_class' StableDiffusionPipeline unset",)


def test_create_prior():
    repo_path = "babalityai/finish_him_.kascade_prior"
    from diffusers import StableCascadePriorPipeline

    class_name = StableCascadePriorPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.finish-him--kascade"
    assert prefixed_data["prior"]["repo"] == repo_path
    assert prefixed_data["prior"]["pkg"][0]["diffusers"] == class_name


def test_create_decoder():
    repo_path = "babalityai/finish_him_.kascade"
    from diffusers import StableCascadeDecoderPipeline

    class_name = StableCascadeDecoderPipeline.__name__
    result = create_pipe_entry(repo_path, class_name)

    mir_series, prefixed_data = result

    assert mir_series == "info.unet.finish-him--kascade"
    assert prefixed_data["decoder"]["repo"] == repo_path
    assert prefixed_data["decoder"]["pkg"][0]["diffusers"] == class_name
