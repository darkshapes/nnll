### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel

from typing import Callable, Tuple

error_dict = "No Errors"


async def retry(max_retries: int, delay_seconds: int, operation: Callable, exception_type: Exception) -> Callable:
    """
    Loop from 0 to `max_retries`\n
    :param max_retries: The amount of times to attempt the operation
    :param delay_seconds: The delay between attempts
    :param operation: The function/method to attempt
    :param exception_type: The most likely exception that will be encountered
    :return: The awaited result of operation()
    """
    import asyncio
    import json

    for retries in range(max_retries + 1):
        try:
            return await operation()
        except exception_type as error_log:
            if retries < max_retries:
                # error_dict = f"Operation failed (retry {retries + 1}/{max_retries})"
                json_log = json.dumps(str(error_log))
                await async_save_file(f"error_log{operation}.json", json_log)
                await asyncio.sleep(delay_seconds)
                retries += 1
            else:
                # error_dict = f"Operation failed after {max_retries} retries"
                json_log = json.dumps(str(error_log))
                await async_save_file(f"error_log{operation}.json", json_log)
                raise


async def async_remote_transfer(session, remote_file_path: str):
    """Request .pdb file from AlphaFold server.
    Ensure the download folder exists (will be rewritten)."""
    import json
    import aiohttp

    # error_dict = f"Downloading PDB file from: {remote_file_path}"

    try:
        async with await session.get(remote_file_path) as response:
            response.raise_for_status()
            return await response.read()
    except aiohttp.client_exceptions.ClientConnectionError as error_log:
        # error_dict = f"Connection error"
        json_log = json.dumps(str(error_log))
        await async_save_file(f"error_log{remote_file_path}.json", json_log)
    except RuntimeError as error_log:
        # error_dict = f"Failed to download, Session Error"
        json_log = json.dumps(str(error_log))
        async_save_file(f"error_log{remote_file_path}.json", json_log)


async def async_save_file(save_file_path_absolute, file_content, mode=None):
    from aiofiles import open as async_open

    # error_dict = f"Saving file: {save_file_path_absolute}"
    mode = "wb" if isinstance(file_content, bytes) else "w"
    file_obj = await async_open(save_file_path_absolute, mode)
    await file_obj.write(file_content)
    await file_obj.close()


async def prepare_download(file_prefix: str, file_suffix: str, remote_url: str, local_download_folder: str) -> Tuple[str]:
    """Prepare download paths for retrieval"""
    import os

    file_name = f"{file_prefix}{file_suffix}"
    remote_file_name = f"{remote_url}{file_name}"
    os.makedirs(local_download_folder, exist_ok=True)
    save_file_path_absolute = os.path.join(local_download_folder, file_name)
    return remote_file_name, save_file_path_absolute


async def gather_text_lines_from(file_path_absolute: str) -> list:
    """synchronously read lines from a text file using aiofiles as async_open"""
    from aiofiles import open as async_open

    async with async_open(file_path_absolute, "r") as file_contents:
        text_lines = [line.strip() async for line in file_contents if line.strip()]
    return text_lines


async def async_download_session(remote_url, save_file_path_absolute):
    """
    Create an async task for heavy downloading procedures
    Await and use lambda to give retry something callable
    """
    tasks = []

    import asyncio
    import aiohttp
    import json
    import requests

    async with aiohttp.ClientSession() as session:
        try:
            file_content = await retry(3, 10, lambda: async_remote_transfer(session, remote_url), requests.HTTPError)

            save_task = asyncio.create_task(retry(3, 1, lambda: async_save_file(save_file_path_absolute, file_content), OSError))
            tasks.append(save_task)

        except aiohttp.ClientError as error_log:
            # error_dict = "Error occurred during request"
            json_log = json.dumps(str(error_log))
            async_save_file(f"error_log{remote_url}.json", json_log)
        else:
            await asyncio.gather(*tasks)


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
    from nnll_27 import pretty_tabled_output
    from tqdm.auto import tqdm

    url_segments = await gather_text_lines_from(remote_file_segments)
    for file_prefix in tqdm(url_segments, total=len(url_segments), position=0, leave=True):
        # error_dict = "No Errors"
        remote_path, save_file_path_absolute = await prepare_download(file_prefix, file_suffix, remote_url, local_download_folder)
        pretty_tabled_output({"title": file_prefix}, {"url": remote_path}, 30)
        await async_download_session(remote_path, save_file_path_absolute)
