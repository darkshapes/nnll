### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel,unused-variable, assignment-from-no-return, redefined-outer-name
# ruff: noqa: F841

import os
from typing import Any, Callable, Tuple

from nnll.monitor.file import dbug, debug_monitor
from nnll.monitor.file import nfo


@debug_monitor
async def retry(max_retries: int, delay_seconds: int, operation: Callable, exception_type: Exception) -> Callable:
    """
    Loop a routine from 0 to `max_retries` and catch exceptions in advance\n
    :param max_retries: The amount of times to attempt the operation
    :param delay_seconds: The delay between attempts
    :param operation: The function/method to attempt
    :param exception_type: The most likely exception that will be encountered
    :return: The awaited result of operation()
    """
    import asyncio

    for retries in range(max_retries + 1):
        try:
            return await operation()
        except exception_type as error_log:
            if retries < max_retries:
                dbug(f"Operation failed (retry {retries + 1}/{max_retries}", tb=error_log.__traceback__)
                await asyncio.sleep(delay_seconds)
                retries += 1
            else:
                dbug(f"Operation failed after {max_retries}) retries.", tb=error_log.__traceback__)
                raise


@debug_monitor
async def async_remote_transfer(session, remote_file_path: str) -> Any:
    """Request a file from a server.  Ensure empty download location exists in advance\n
    :param session: Current asynchronous session
    :param remote_file_path: Full URL to the file
    :return: The downloaded file from `response`"""

    # import json
    import aiohttp

    nfo(f"Downloading PDB file from: {remote_file_path}")

    try:
        async with await session.get(remote_file_path) as response:
            response.raise_for_status()
            return await response.read()
    except aiohttp.client_exceptions.ClientConnectionError as error_log:
        dbug(f"Connection error.{error_log}", tb=error_log.__traceback__)

    except RuntimeError as error_log:
        dbug(f"Failed to download, Session Error. {error_log}", tb=error_log.__traceback__)


@debug_monitor
async def async_save_file(save_file_path_absolute: str, file_content: Any, mode=None) -> None:
    """
    Write a file to disk while keeping asynchronicity\n
    :param save_file_path_absolute: Location to save the file
    :param file_content: The data to store in the file
    :param mode: File write method (default: `w` or `wb` by `content type`)
    :return: None
    """
    from aiofiles import open as async_open

    nfo(f"Saving file: {save_file_path_absolute}")
    if mode is None:
        mode = "wb" if isinstance(file_content, bytes) else "w"
    file_obj = await async_open(save_file_path_absolute, mode)
    await file_obj.write(file_content)
    await file_obj.close()


@debug_monitor
async def prepare_download(file_prefix: str, file_suffix: str, remote_url: str, local_download_folder: str) -> Tuple[str]:
    """
    Construct paths and URL locations\n
    :param file_prefix: The head of the file name
    :param file_suffix: The extension of the file name
    :param remote_url: Remote location of file
    :param local_download_folder: Relative or absolute target folder
    :return: The absolute paths of local and remote endpoints
    """

    file_name = f"{file_prefix}{file_suffix}"
    remote_file_name = f"{remote_url}{file_name}"
    os.makedirs(local_download_folder, exist_ok=True)
    save_file_path_absolute = os.path.join(local_download_folder, file_name)
    return remote_file_name, save_file_path_absolute


@debug_monitor
async def gather_text_lines_from(file_path_absolute: str) -> list:
    """
    Asynchronously read lines from a text file using aiofiles as async_ope\n
    :param gather_text_lines_from: A file with a linebreak-separated list of file basenames
    :return: A iterator object from the text lines
    """
    from aiofiles import open as async_open

    async with async_open(file_path_absolute, "r") as file_contents:
        text_lines = [line.strip() async for line in file_contents if line.strip()]
    return text_lines


@debug_monitor
async def async_download_session(remote_url: str, save_file_path_absolute: str) -> None:
    """
    Create an async task for heavy downloading procedures
    Await and use lambda to give retry something callable\n
    :param remote_url: Remote location of file_content
    :param save_file_path_absolute: Local path to save file atan
    :return: None
    """
    import asyncio
    import aiohttp
    import requests

    tasks = []

    async with aiohttp.ClientSession() as session:
        try:
            file_content = await retry(3, 10, lambda: async_remote_transfer(session, remote_url), requests.HTTPError)

            save_task = asyncio.create_task(retry(3, 1, lambda: async_save_file(save_file_path_absolute, file_content), OSError))
            tasks.append(save_task)

        except aiohttp.ClientError as error_log:
            dbug(f"Error occurred during request. {error_log}", tb=error_log.__traceback__)
        else:
            await asyncio.gather(*tasks)


@debug_monitor
async def bulk_download(
    remote_file_segments: str,
    remote_url: str,
    file_suffix: str = ".pdb",
    local_download_folder: str = ".",
) -> None:
    """
    Download a large quantity of files to a single location\n
    :param remote_file_segments: Plain text, newline-separated list of files to download
    :param remote_url: Server location for the files, including training /
    :param file_suffix: Extension for the files
    :return: None
    """
    from tqdm.auto import tqdm

    from nnll.monitor.console import pretty_tabled_output

    url_segments = await gather_text_lines_from(remote_file_segments)
    for file_prefix in tqdm(url_segments, total=len(url_segments), position=0, leave=True):
        remote_path, save_file_path_absolute = await prepare_download(file_prefix, file_suffix, remote_url, local_download_folder)
        pretty_tabled_output({"title": file_prefix}, {"url": remote_path}, 30)
        await async_download_session(remote_path, save_file_path_absolute)


### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


if __name__ == "__main__":
    import asyncio

    remote_file_segments = "test_dl.txt"
    remote_url = "https://alphafold.ebi.ac.uk/files/"
    file_suffix = ".pdb"
    local_download_folder = "/home/maxtretikov/Archive/Code/genie2/data/afdbreps_l-256_plddt_80/pdbs"  # this should all probably be argparse statements
    print("\n\n\n\n")
    asyncio.run(
        bulk_download(
            remote_file_segments,
            remote_url,
            file_suffix,
            local_download_folder,
        )
    )
