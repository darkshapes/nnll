# ### <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
# ### <!-- // /*  d a r k s h a p e s */ -->

import pytest
from unittest import mock
from nnll.tensor_pipe.deconstructors import process_docs


def test_list_diffusers_models():
    process_docs()


# @pytest.fixture
# def mock_modules(mocker):
#     # Mock pkgutil.iter_modules to return controlled values
#     mock_iter_modules = mocker.patch("pkgutil.iter_modules")
#     mock_iter_modules.return_value = [mock.Mock(name="example_module_1", is_pkg=True), mock.Mock(name="example_module_2", is_pkg=True)]

#     # Mock import_module for dynamic imports
#     mock_import_module = mocker.patch("importlib.import_module")

#     def side_effect(module_name):
#         if module_name == "diffusers.pipelines.example_module_1.pipeline_example_module_1":
#             return mock.Mock()  # Return a mock object representing the module
#         raise ModuleNotFoundError(f"Module {module_name} not found")

#     mock_import_module.side_effect = side_effect

#     yield mock_iter_modules, mock_import_module


# @pytest.fixture
# def mock_kandinsky():
#     with mock.patch("diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2", autocast=True) as mock_data:
#         mock_data.return_value = """
#     Examples:
#         ```py
#         >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
#         >>> import torch

#         >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
#         >>> pipe_prior.to("cuda")
#         >>> prompt = "red cat, 4k photo"
#         >>> out = pipe_prior(prompt)
#         >>> image_emb = out.image_embeds
#         >>> zero_image_emb = out.negative_image_embeds
#         >>> pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
#         >>> pipe.to("cuda")
#         >>> image = pipe(
#         ...     image_embeds=image_emb,
#         ...     negative_image_embeds=zero_image_emb,
#         ...     height=768,
#         ...     width=768,
#         ...     num_inference_steps=50,
#         ... ).images
#         >>> image[0].save("cat.png")
#         ```
# """
#         return mock_data


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
