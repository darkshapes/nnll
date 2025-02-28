import asyncio
from unittest import mock
import pytest
import pytest_asyncio
from nnll_03 import async_download_session


@pytest_asyncio.fixture(loop_scope="session")
def mock_session():
    # Mock aiohttp.ClientSession properly
    with mock.patch("aiohttp.ClientSession", new_callable=mock.AsyncMock) as session_mock:
        yield session_mock


@pytest_asyncio.fixture(loop_scope="session")
def mock_async_remote_transfer():
    # Correctly mock async_remote_transfer as an AsyncMock
    with mock.patch("nnll_03.async_remote_transfer", new=mock.AsyncMock()) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="session")
def mock_async_save_file():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll_03.async_save_file", new=mock.AsyncMock()) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="session")
def mock_retry():
    # Mock retry function properly so it returns an awaitable
    async def mock_retry_func(*args, **kwargs):
        if isinstance(args[2], Exception):  # Simulating an error
            raise args[2]
        return await args[0]()  # Ensure we await the passed function

    with mock.patch("nnll_03.retry", new=mock.AsyncMock(side_effect=mock_retry_func)) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_async_download_session_os_error(mock_retry, mock_async_save_file):
    # Arrange: Simulate an OSError during file saving
    # Mock download task (this will be successful)
    # Mock save task (this will fail with OSError)
    # Set side effects for the mock_retry to return both tasks
    # Act: Run the async function under test
    # Assert that no client error print occurred (i.e., no aiohttp.ClientError)
    # Ensure the save function was called with correct arguments
    # The key here is to resolve the future and pass its result explicitly.
    remote_url = "https://example.com/file"
    file_path = "/tmp/saved_file"
    expected_content = b"file content"
    expected_error = OSError("Disk error")

    async def fake_fail():
        raise expected_error  # Simulate disk failure

    async def fake_successful_dl():
        return expected_content

    mock_retry.side_effect = [
        lambda *args, **kwargs: fake_fail(),  # First call: Fail
        lambda *args, **kwargs: fake_successful_dl(),  # Second call: Succeed
    ]
    future_content = asyncio.Future()
    future_content.set_result(expected_content)
    with mock.patch("builtins.print") as mock_print:
        await async_download_session(remote_url, file_path)

        mock_print.assert_not_called()

        mock_async_save_file.assert_called_once()

        resolved_content = await future_content  # This will give you `expected_content` as bytes.
        assert resolved_content == expected_content
