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

class AsyncContextManager:
    def __init__(self, obj):
        self._obj = obj

    @pytest_asyncio.fixture(loop_scope="session")
    async def __aenter__(self):
        return self._obj

    @pytest_asyncio.fixture(loop_scope="session")
    async def __aexit__(self, *args):
        pass

@pytest.mark.asyncio(loop_scope="session")
async def test_async_download_session_os_error(mock_retry, mock_async_save_file):
    # Arrange: Simulate an OSError during file saving

    remote_url = "https://example.com/file"
    local_path = "/tmp/saved_file"
    expected_content = b"file content"
    simulated_error = OSError("Disk error")

    async def fake_fail():
        print("Simulating fail")  # This will now actually print to console
        raise simulated_error

    async def fake_successful_dl():
        print("Simulating successful download")  # This will now print to console
        return expected_content

    # Set up side effects for mock_retry: first fail, then succeed
    mock_retry.side_effect = [fake_fail(), fake_successful_dl()]

    with mock.patch("nnll_03.async_open", new_callable=mock.AsyncMock) as mock_open:
        mock_file = mock.AsyncMock()
        mock_open.return_value = AsyncContextManager(mock_file)

        # Act: Perform the async download session
        await async_download_session(remote_url, local_path)

        # Assert that file was opened in binary write mode
        mock_open.assert_awaited_once_with(local_path, "wb")

        # Assert that content was written to the file
        mock_file.write.assert_called_once_with(expected_content)


