# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
from nnll.integrity.hash_256 import write_to_file
from nnll.monitor.console import nfo
from nnll.monitor.file import dbuq


async def generate_modality_data(mir_db):
    from requests.exceptions import HTTPError
    from huggingface_hub import repocard, model_info
    from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError, HFValidationError

    for series, compatibility in mir_db.database.items():
        if "info." in series:
            for comp, field in compatibility.items():
                if repo := field.get("repo"):
                    try:
                        if model_tags := model_info(repo_id=repo, expand=["pipeline_tag", "library_name", "tags"]):
                            if model_tags.pipeline_tag:
                                yield (
                                    f"{series}.{comp}",
                                    {
                                        "pipeline": model_tags.pipeline_tag,
                                        "library": model_tags.library_name,
                                        "tags": model_tags.tags,
                                    },
                                )
                        else:
                            yield (
                                f"{series}.{comp}",
                                {
                                    "pipeline": repocard.RepoCard.load(repo).data.get("pipeline_tag"),
                                    "library": repocard.RepoCard.load(repo).data.get("tasks"),
                                    "tags": repocard.RepoCard.load(repo).data.get("tags"),
                                },
                            )
                    except RepositoryNotFoundError as error_log:
                        nfo(f"Repository not found: {repo}")
                        dbuq(error_log)
                        continue
                    except (HTTPError, EntryNotFoundError) as error_log:
                        nfo(f"No model card for {repo}")
                        dbuq(error_log)
                        continue
                    except HFValidationError as error_log:
                        nfo(f"Invalid repo name {repo}")
                        dbuq(error_log)
                        continue


async def modality_data():
    import os
    from huggingface_hub import constants
    from nnll.mir import config
    from nnll.mir.maid import MIRDatabase
    from nnll.metadata.json_io import write_json_file

    constants.HF_HUB_OFFLINE = 0
    os.environ["HF_HUB_OFFLINE"] = "0"
    mir_db = MIRDatabase()
    repo_db = {}
    async for key_name, value_data in generate_modality_data(mir_db):
        repo_db.setdefault(key_name, value_data)
    constants.HF_HUB_OFFLINE = 1
    os.environ["HF_HUB_OFFLINE"] = "1"
    write_json_file(folder_path_named=os.path.dirname(config.__file__), file_name="modes.json", data=repo_db)
    return repo_db


def main():
    import asyncio

    nfo(asyncio.run(modality_data()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Add generative tasks available to the currently installed system environment to MIR database.\Online function.",
        usage="mir-autopipe",
        epilog="""""",
    )
    parser.parse_args()
    main()
