import asyncio
import os
from typing import Callable, Tuple

import aiohttp
import requests
from aiofiles import open as async_open


async def retry(max_retries: int, delay_seconds: int, operation: Callable, exception_type: Exception) -> Callable:
    retries = 0
    while retries <= max_retries:
        try:
            return await operation()
        except exception_type as error_log:
            if retries < max_retries:
                print(f"Operation failed (retry {retries + 1}/{max_retries}): {error_log}")
                await asyncio.sleep(delay_seconds)
                retries += 1
            else:
                print(f"Operation failed after {max_retries} retries: {error_log}")
                raise


async def concurrent_download(session, remote_file_path: str):
    """Request .pdb file from AlphaFold server.
    Ensure the download folder exists (will be rewritten)."""

    print(f"Downloading PDB file from: {remote_file_path}")

    try:
        async with await session.get(remote_file_path) as response:
            response.raise_for_status()
            return await response.read()
    except aiohttp.client_exceptions.ClientConnectionError as error_log:
        print(f"Connection error: {error_log}")
    except RuntimeError as error_log:
        print(f"Failed to download, Session Error: {error_log}")


async def save_file(save_file_path_absolute, file_content):
    print(f"Saving PDB file to: {save_file_path_absolute}")
    mode = "wb" if isinstance(file_content, bytes) else "w"
    async with await async_open(save_file_path_absolute, mode) as open_file:
        await open_file.write(file_content)


async def prepare_download(file_prefix: str, file_suffix: str, remote_url: str, local_download_folder: str) -> Tuple[str]:
    """Prepare download paths for retrieval"""
    file_name = f"{file_prefix}{file_suffix}"
    remote_file_name = f"{remote_url}/{file_name}"
    os.makedirs(local_download_folder, exist_ok=True)
    save_file_path_absolute = os.path.join(local_download_folder, file_name)
    return remote_file_name, save_file_path_absolute


async def gather_text_lines_from(file_path_absolute: str) -> list:
    """synchronously read lines from a text file using aiofiles as async_open"""
    async with async_open(file_path_absolute, "r") as file_contents:
        text_lines = [line.strip() async for line in file_contents if line.strip()]
    return text_lines


async def async_download(remote_url, save_file_path_absolute):
    """
    Create an async task for heavy downloading procedures
    """
    tasks = []

    async with aiohttp.ClientSession() as session:
        try:
            download_task = asyncio.create_task(retry(3, 10, concurrent_download(session, remote_url), requests.HTTPError))
            tasks.append(download_task)
        except aiohttp.ClientError as error_log:
            print(f"Error occurred during request: {error_log}")
        else:

            async def save_task(file_content):
                await retry(3, 1, save_file(save_file_path_absolute, file_content), OSError)

            tasks.append(asyncio.create_task(download_task.then(save_task)))

        await asyncio.gather(*tasks)


async def bulk_download(
    remote_files: str = "test_dl.txt",
    remote_url: str = "https://alphafold.ebi.ac.uk/files/",
    file_suffix: str = ".pdb",
    local_download_folder: str = "/home/maxtretikov/Archive/Code/genie2/data/afdbreps_l-256_plddt_80/pdbs",
) -> None:
    """
    Download a large quantity of files to a single location\n
    :param remote_files: Plain text, newline-separated list of files to download
    :param remote_url: Server location for the files
    :param file_suffix: Extension for the files
    :return: None
    """
    url_segments = gather_text_lines_from(remote_files)
    for file_prefix in url_segments:
        remote_url, save_file_path_absolute = await prepare_download(file_prefix, file_suffix, remote_url, local_download_folder)
        await async_download(remote_url, save_file_path_absolute)


if __name__ == "__main__":
    asyncio.run(bulk_download())
