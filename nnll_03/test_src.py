import os
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import asyncio

from aioresponses import aioresponses

from nnll_03 import async_remote_transfer, prepare_download, retry, async_save_file

local_folder = os.path.join(os.path.dirname(__file__), "test", "download")
file_name = "test_dl.txt"


@pytest.mark.asyncio(loop_scope="session")
async def test_prepare_download():
    # Arrange: Set up mock return values
    expected_filename = "prefix.suffix"
    expected_return_remote = f"http://example.com/file/{expected_filename}"
    expected_return_local = os.path.join(local_folder, expected_filename)
    # mock_prepare_storage.return_value = (expected_return_remote, expected_return_local)

    # Act: Call the code that uses prepare_storage
    file_prefix = "prefix"
    file_suffix = ".suffix"
    remote_url = "http://example.com/file"
    local_download_folder = local_folder

    remote_file_name, save_file_path_absolute = await prepare_download(
        file_prefix,
        file_suffix,
        remote_url,
        local_download_folder,
    )

    # Assert: Check if the mocked values were returned
    assert remote_file_name == expected_return_remote
    assert save_file_path_absolute == expected_return_local


@pytest.mark.asyncio(loop_scope="session")
async def test_save_file_async():
    save_file_path_absolute = "/fake/path/to/file.pdb"
    file_content = b"mock content"

    with patch("nnll_03.async_open", new_callable=AsyncMock) as mock_open:
        mock_file = AsyncMock()
        mock_open.return_value.__aenter__.return_value = mock_file

        await async_save_file(save_file_path_absolute, file_content)

        # Assert file was opened with the correct mode
        mock_open.assert_awaited_once_with(save_file_path_absolute, "wb")

        # Assert file content was written
        mock_file.write.assert_called_once_with(file_content)
        with pytest.raises(AssertionError):
            mock_file.write.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_async_remote_transfer_success():
    """simulate aiohttp.ClientSession
    Use aioresponses to intercept the HTTP request
    Mock GET request using aioresponse
    respond with status 200 or content
    """
    remote_file_path = "https://example.com/file.pdb"
    mock_content = b"mocked pdb content"

    with aioresponses() as mock_response:
        # Mocking GET request with status 200 and predefined content
        mock_response.get(remote_file_path, status=200, body=mock_content)

        # Create an actual aiohttp.ClientSession (aioresponses will intercept it)
        async with aiohttp.ClientSession() as session:
            result = await async_remote_transfer(session, remote_file_path)

        # Ensure we got back the expected content
        assert result == mock_content


@pytest.mark.asyncio(loop_scope="session")
async def test_async_remote_transfer_failure():
    # intercept HTTP request, Mock 404
    # build session, simulate failure

    remote_file_path = "https://example.com/nonexistent.pdb"
    with aioresponses() as mock_response:
        mock_response.get(remote_file_path, status=404)
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientResponseError):
                await async_remote_transfer(session, remote_file_path)


@pytest.mark.asyncio(loop_scope="session")
async def test_async_remote_transfer_read_awaits():
    """Mock get"""
    remote_file_path = "https://example.com/file.pdb"
    mock_content = b"mocked pdb content"

    with aioresponses() as mock_response:
        mock_response.get(remote_file_path, status=200, body=mock_content)
        async with aiohttp.ClientSession() as session:
            result = await async_remote_transfer(session, remote_file_path)

            assert result == mock_content


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_success_after_three_attempts():
    """Mock operation that fails twice and then succeeds
    Patch asyncio.sleep to avoid delay"""
    mock_operation = AsyncMock(side_effect=[Exception("Error 1"), Exception("Error 2"), "Success"])

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await retry(max_retries=3, delay_seconds=1, operation=mock_operation, exception_type=Exception)

        assert result == "Success"

        # Verify 3 calls (2 failures + 1 success)
        assert mock_operation.call_count == 3


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_fails_after_max_retries():
    """Mock 100% failure"""
    mock_operation = AsyncMock(side_effect=[Exception("Error")] * 4)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(Exception):  # Expect the exception to be raised after max retries
            await retry(max_retries=3, delay_seconds=1, operation=mock_operation, exception_type=Exception)

        # Verify 4 cals (max_retries + 1)
        assert mock_operation.call_count == 4


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_first_attempt_success():
    """Mock success on the first try"""
    mock_operation = AsyncMock(return_value="Success")

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await retry(max_retries=3, delay_seconds=1, operation=mock_operation, exception_type=Exception)

        assert result == "Success"

        mock_operation.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_custom_exception():
    """Mock custom exception (e.g., ValueError)"""
    mock_operation = AsyncMock(side_effect=[ValueError("Custom Error")] * 3)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ValueError):
            await retry(max_retries=2, delay_seconds=1, operation=mock_operation, exception_type=ValueError)

        assert mock_operation.call_count == 3  # Max retries + 1 attempt


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_no_retries():
    # Always fails, 0 Retries (PERMADEATH MODE)
    mock_operation = AsyncMock(side_effect=Exception("No retries allowed"))

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(Exception):  # death
            await retry(max_retries=0, delay_seconds=1, operation=mock_operation, exception_type=Exception)

        mock_operation.assert_called_once()  # dead?
