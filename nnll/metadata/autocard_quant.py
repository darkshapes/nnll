### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nnll.integrity import ensure_path
from nnll.monitor.file import nfo


def index_model_card(repo_path) -> Optional[Dict[str, Any]]:
    """Fetch repo modelcard\n
    :param repo_path: Path to repo for modelcard
    :return: The model card tags as dictionary
    """
    from huggingface_hub import repocard, constants

    constants.HF_HUB_OFFLINE = 0
    constants.HF_XET_HIGH_PERFORMANCE = 1
    constants.HF_HUB_ENABLE_HF_TRANSFER = 1

    model_metadata = repocard.RepoCard.load(repo_path).data
    nfo(f"Metadata acquired from {repo_path}")
    return model_metadata


def convert_repo(conditions: tuple) -> str:
    """card"""
    from huggingface_hub import constants, snapshot_download
    import subprocess

    constants.HF_HUB_OFFLINE = 0
    constants.HF_XET_HIGH_PERFORMANCE = 1
    constants.HF_HUB_ENABLE_HF_TRANSFER = 1
    ensure_path(conditions[5])
    if conditions[2]:
        folder_path_named = snapshot_download(repo_id=conditions[0], local_dir=conditions[5])
        command = ["convert_hf_to_gguf.py", folder_path_named]
        if conditions[4]:
            command.extend([" --outtype", f"{conditions[4]}"])

    else:
        command = ["mlx_lm.convert", "--hf-path", conditions[0], "--mlx-path", conditions[5]]
        if conditions[4]:
            command.extend(["-q", "--q-bits", f"{conditions[4]}"])

    output = subprocess.run(command, check=True)
    nfo(f"status: {output}")

    return conditions[5]


def write_card(model_path, model_card):
    """card"""

    readme = Path(os.path.join(model_path, "README.md"))
    if os.path.exists(readme):
        os.remove(readme)
    readme.touch(exist_ok=False)
    with open(readme, "w", encoding="UTF-8") as md_file:
        md_file.write(model_card)


def generate_model_card(conditions: tuple, full_tags: dict, model_name: str, dry_run: bool) -> None:
    """Assemble the gathered metadata for the card\n
    :param conditions: Generation details
    :param full_tags: The repocard with user-augmented fields
    :param model_name: Name for the model
    :param dry_run: Whether or not to generate model card only or run conversion
    """
    tag_block = ["---"]
    model_field = [f"name: {model_name}"]
    link_block = ["> [!IMPORTANT]", f"> Original Model Link : [https://huggingface.co/{conditions[0]}](https://huggingface.co/{conditions[0]})", "> \n"]
    code_block = ["```"]
    content = []
    for tag, value in full_tags.items():
        if value is not None:
            if isinstance(value, list) and len(value) > 1:
                new_value = []
                for each in value:
                    new_value.append(f" - {each}")
                new_value.insert(0, f"{tag} :")
                value = new_value
            elif isinstance(value, str):
                value = [f"{tag}: {value}"]
            else:
                value = [f"{tag}: {value[0]}"]
            content.extend(value)
    mlx_blurb = f"""

    # {model_name}
    {model_name} <Headline>

    > [!WARNING]
    > MLX is a framework for METAL graphics supported by Apple computers with ARM M-series processors (M1/M2/M3/M4)

    > [!NOTE]
    > Generation using uv https://docs.astral.sh/uv/**:
    > ```
    > uvx --from mlx-lm mlx_lm.generate -model \"darkshapes/{model_name}\" --prompt 'Test prompt'
    >```

    > [!NOTE]
    > Generation using pip:
    > ```
    > pipx --from mlx-lm mlx_lm.generate -model \"darkshapes/{model_name}\" --prompt 'Test prompt'
    > ```
    """
    model_card_elements = tag_block + model_field + content + tag_block + link_block + code_block + model_field + content + code_block + mlx_blurb
    model_card = "\n".join(model_card_elements)
    validate = input(f"""Generated_card:\n{model_card}\n\n to {conditions[5]} Correct? """)

    if "y" in validate.lower():
        if not dry_run:
            model_path = convert_repo(conditions)
        else:
            model_path = ensure_path(conditions[5])
        write_card(model_path, model_card)
    else:
        return


def autocard(conditions: tuple, dry_run: bool) -> None:  # pylint:disable=redefined-outer-name
    """Generate a model card then convert the downloaded repo\n
    :param conditions: Generation details
    :param dry_run: Run card generation only\n
    """

    def ask_multi_input(tag, polite_msg: str = "Please provide", preposition: str = "metadata for", required=True) -> List[str]:
        """card"""
        input_store = []
        for prompt in [polite_msg, preposition]:
            prompt = prompt.strip()
        user_input = input(f"{polite_msg} {preposition} {tag}: ")
        if not user_input and not required:
            return None
        input_store.append(user_input)
        while True:
            if user_input and input_store:
                metadata = f"additional {preposition}"
                user_input = input(f"{polite_msg} {metadata} {tag} (leave blank to skip): ")
                if user_input:
                    input_store.append(user_input)
                else:
                    return input_store

    test_prompt = "Test prompt"
    required_tags = {
        "base_model": None,
        "license": None,
        "pipeline_tag": None,
        "tasks": None,
        "language": None,
        "datasets": None,
    }
    optional_tags = {
        "funded_by": None,
        "hardware_type": None,
        "cloud_region": None,
        "cloud_provider": None,
        "hours_used": None,
        "co2_emitted": None,
        "thumbnail": None,
        "short headline description": None,
    }
    model_name = f"{os.path.basename(conditions[5])}"
    if not conditions[2]:
        optional_tags.setdefault(
            "get_started_code",
            f"uvx --from mlx-lm mlx_lm.generate --model \"darkshapes/{model_name}\" --prompt '{test_prompt}'",
        )
    model_metadata = index_model_card(conditions[0])
    full_tags = required_tags | optional_tags
    for tag, _ in full_tags.items():
        if hasattr(model_metadata, tag):
            full_tags[tag] = getattr(model_metadata, tag)
        if not full_tags[tag]:
            full_tags[tag] = ask_multi_input(tag, required=tag in required_tags)
    generate_model_card(conditions, full_tags=full_tags, model_name=model_name, dry_run=dry_run)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Process repo path and conversion type\n\n **Valid pipeline tags**:\n\n text-classification, token-classification, table-question-answering, question-answering, zero-shot-classification, translation, summarization, feature-extraction, text-generation, text2text-generation, fill-mask, sentence-similarity, text-to-speech, text-to-audio, automatic-speech-recognition, audio-to-audio, audio-classification, audio-text-to-text, voice-activity-detection, depth-estimation, image-classification, object-detection, image-segmentation, text-to-image, image-to-text, image-to-image, image-to-video, unconditional-image-generation, video-classification, reinforcement-learning, robotics, tabular-classification, tabular-regression, tabular-to-text, table-to-text, multiple-choice, text-ranking, text-retrieval, time-series-forecasting, text-to-video, image-text-to-text, visual-question-answering, document-question-answering, zero-shot-image-classification, graph-ml, mask-generation, zero-shot-object-detection, text-to-3d, image-to-3d, image-feature-extraction, video-text-to-text, keypoint-detection, visual-document-retrieval, any-to-any, other"""
    )

    parser.add_argument("repo", type=str, help="Relative path to HF repository")
    parser.add_argument("-g", "--gguf", action="store_true", help="Boolean to switch from MLX to GGUF format")
    parser.add_argument("-d", "--dry_run", action="store_true", help="Perform a dry run without downloading")
    parser.add_argument("-f", "--folder", type=str, default=os.path.join(os.path.expanduser("~"), "Downloads"), help="Optional folder path for downloading")
    parser.add_argument("-q", "--quantization", type=Union[int, str], required=False, help="Set quantization level")

    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    conditions = (
        args.repo,
        args.folder,
        args.gguf,
        "GGUF" if args.gguf else "MLX",
        args.quantization,
        os.path.join(args.folder, os.path.basename(f"{args.repo}-{'GGUF' if args.gguf else 'MLX'}")),
    )
    nfo(f"Processing repo: {conditions[0]} to {conditions[5]} : extra arguments {conditions[4]}")
    autocard(conditions=conditions, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
