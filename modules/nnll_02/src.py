import os
import aiofiles
import aiohttp
import asyncio
import requests
from typing import Callable, Tuple


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


async def save_file(save_file_path_absolute, file_content):
    print(f"Saving PDB file to: {save_file_path_absolute}")
    mode = "wb" if isinstance(file_content, bytes) else "w"
    async with await aiofiles.open(save_file_path_absolute, mode) as open_file:
        await open_file.write(file_content)


async def prepare_storage(file_prefix: str, file_suffix: str, remote_url: str, local_download_folder: str) -> Tuple[str]:
    """Prepare download paths for retrieval"""
    file_name = f"{file_prefix}{file_suffix}"
    remote_file_name = f"{remote_url}/{file_name}"
    os.makedirs(local_download_folder, exist_ok=True)
    save_file_path_absolute = os.path.join(local_download_folder, file_name)
    return remote_file_name, save_file_path_absolute


async def main(
    remote_file_collection: str = "test_dl.txt",
    remote_url: str = "https://alphafold.ebi.ac.uk/files/",
    local_download_folder: str = "/home/maxtretikov/Projects/Code/darkshapes/rna/genie2/data/afdbreps_l-256_plddt_80/pdbs",
    file_suffix=".pdb",
):
    """Main function, specify filename and trigger download
    Open and read the file line by line (each line is part of a URL)
    Trim whitespace, then run attempts to download.
    file_suffix must include "." !
    """

    with open(remote_file_collection, "r") as file_prefixes:
        url_segments = [line.strip() for line in file_prefixes if line.strip()]
    tasks = []
    async with await aiohttp.ClientSession() as session:
        for file_prefix in url_segments:
            try:
                remote_file_name, save_file_path_absolute = await prepare_storage(
                    file_prefix,
                    file_suffix,
                    remote_url,
                    local_download_folder,
                )
                file_content = await retry(3, 10, await concurrent_download(session, remote_file_name), requests.HTTPError)
                tasks.append(file_content)
            except RuntimeError as error_log:
                print(f"Failed to download, Session Error: {error_log}")
            except BaseException as error_log:
                print(f"Download failed. Unexpected Error: {error_log}")
                raise
            else:
                tasks.append(await retry(3, 1, await save_file(save_file_path_absolute, file_content), OSError))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
