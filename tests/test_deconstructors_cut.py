### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->
import pytest
from unittest import mock
from unittest.mock import Mock, patch
from nnll.tensor_pipe.deconstructors import cut_docs
from unittest.mock import call


def test_list_diffusers_models():
    cut_docs()


@pytest.fixture
def mock_import_module(mocker):
    """Fixture to mock import_module and simulate different module scenarios."""
    return mocker.patch("nnll.tensor_pipe.deconstructors.import_module")


@pytest.fixture
def mock_pkgutil_iter_modules(mocker):
    """Fixture to mock pkgutil.iter_modules for controlled testing."""
    from importlib.machinery import FileFinder

    module_finder = Mock()
    return mocker.patch(
        "pkgutil.iter_modules",
        return_value=[
            (Mock(), "allegro", True),
            (Mock(), "amused", True),
            (Mock(), "animatediff", True),
            (Mock(), "audioldm", True),
            (Mock(), "cogvideo", True),
            (Mock(), "deepfloyd_if", True),
        ],
    )


def test_cut_docs_excluded(mock_import_module, mock_pkgutil_iter_modules):
    """Test that excluded modules are not processed."""
    # Define the list of excluded modules for testing
    excluded_modules = ["ddpm"]

    # Mock import_module to raise ImportError for excluded modules
    def side_effect(import_name, *args, **kwargs):
        if any(exc in import_name for exc in excluded_modules):
            raise ImportError(f"Module {import_name} is excluded.")
        return Mock()

    mock_import_module.side_effect = side_effect

    results = list(cut_docs())

    # Ensure no attempts to import excluded modules like ddpm
    assert not any("ddpm" in call[0][0] for call in mock_import_module.call_args_list)


def test_cut_docs_non_standard(mock_import_module, mock_pkgutil_iter_modules):
    """Test that non-standard module names are correctly mapped."""
    docstrings = ["mock_docstring1", "mock_docstring2", "mock_docstring3", "mock_dock_string4", "mock_dock_floyd"]
    call_count = 0

    def side_effect(module_name):
        nonlocal call_count
        result = type("MockModule", (object,), {"EXAMPLE_DOC_STRING": docstrings[call_count]})
        call_count += 1
        return result

    mock_import_module.side_effect = side_effect

    results = list(cut_docs())

    # Ensure correct import path for non-standard name mapping
    mock_import_module.assert_any_call("diffusers.pipelines.cogvideo.pipeline_cogvideox")
    mock_import_module.assert_any_call("diffusers.pipelines.deepfloyd_if.pipeline_if")

    # Check if the docstrings are returned correctly

    assert "mock_docstring1" in results
    assert "mock_dock_string4" in results
    assert "mock_dock_floyd" in results


def test_cut_docs_module_not_found(mock_import_module, mock_pkgutil_iter_modules, capsys):
    """Test handling of ModuleNotFoundError."""
    mock_import_module.side_effect = ModuleNotFoundError("Simulated module not found")

    list(cut_docs())

    captured = capsys.readouterr()

    # Ensure the error is logged correctly
    assert "Module Not Found for allegro" in captured.err


def test_cut_docs_docstring_not_found(mock_import_module, mock_pkgutil_iter_modules, capsys):
    """Test handling of missing docstrings."""
    mock_import_module.return_value = type("MockModule", (object,), {})

    list(cut_docs())

    captured = capsys.readouterr()

    # Ensure the error is logged correctly
    assert "Doc String Not Found for cogvideo" in captured.err


def test_cut_docs_yield_docstrings(mock_import_module, mock_pkgutil_iter_modules):
    """Test that docstrings are yielded as expected."""
    mock_import_module.return_value = type("MockModule", (object,), {"EXAMPLE_DOC_STRING": "mock_docstring"})

    results = list(cut_docs())

    # Ensure the correct docstring is returned
    assert "mock_docstring" in results


# pipe_doc = next(line.partition(prefix)[2].split('",')[0] for line in doc_string.splitlines() if prefix in line)

# Print only the class and cleaned repo path
# [2].replace("...", "").strip()
# repo_path_unbracket =repo_path_full.split('",')[0]
# repo_path = repo_path_full.split('",')[0]  # Only keep the part before '",'

# repo_path = pipe_doc.partition(pretrained_prefix)[2].partition(")")


#    repo_prefixes = [
#         ">>> repo_id = ",
#         ">>> model_id_or_path = ",
#         ">>> model_id = ",
#     ]
# init_pipe.setdefault(name, {"pipe": pipe_class})  # "repo": repo_path,

#     cuda_suffix = 'pipe.to("cuda")'
# pipe_doc = [line.strip().replace(pipe_prefix, "") for line in doc_string.splitlines() if pipe_prefix in line]
# repo_path = [line.strip().replace(prefix, "").strip('"') for line in d for prefix in repo_prefixes if prefix in doc_string]
# if pipe_doc:
#     pipe_doc = next(iter(pipe_doc)).split(pretrained_prefix)
#     if isinstance(pipe_doc, list) and len(pipe_doc) > 1 and not repo_path:
#         repo_path = pipe_doc[1].split('",')[0].strip('"')
#         # print(f"no repo {pipe_doc}")
#             if len(next(iter(pipe_command))) > 1:
# pipe_repo_id = next(iter(pipe_command))[1]
# else:
#     temp_repo_id = [line.strip().replace(repo_prefix, "") for line in doc_string.splitlines() if repo_prefix in line])
#     pattern = r'.from_pretrained\("([^"]+)"'
#     pretrained_pattern = re.compile(pattern)
#     pipe_repo_id = re.findall(pretrained_pattern, temp_repo_id)
# print(init_pipe)
# print([y.get("repo") for x, y in init_pipe.items()])
# repo_id =
# if not repo_id:
#     repo_id = [str(line.replace(pretrained_prefix, "")).split(pretrained_prefix) for line in doc_string.splitlines() if pretrained_prefix in line]
#     if repo_id and len(repo_id) > 1:
#         repo_id = repo_id[1]
# # if "/" not in repo_id:

# pattern = r'.from_pretrained\("([^"]+)"'
# matches = re.findall(pattern, text)

#
# # repo_id = [line.strip().replace(repo_prefix, "") for line in doc_string.splitlines() if ")" in line]

if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
