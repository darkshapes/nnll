# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from unittest import mock
import pytest
import pytest_asyncio

from nnll.download.async_download import bulk_download


@pytest_asyncio.fixture(loop_scope="session")
def mock_prepare_download():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll.download.async_download.prepare_download", new=mock.AsyncMock()) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="session")
def mock_async_download_session():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll.download.async_download.async_download_session", new=mock.AsyncMock()) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="session")
def mock_gather_text_lines_from():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll.download.async_download.gather_text_lines_from", new=mock.AsyncMock(return_value=["file1", "file2"])) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="session")
def mock_gather_empty():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll.download.async_download.gather_text_lines_from", new=mock.AsyncMock(return_value=[])) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_bulk_download_success(mock_async_download_session, mock_prepare_download, mock_gather_text_lines_from):
    # Arrange: Mock the prepare_download to return dummy remote_url and save_file_path_absolute
    mock_prepare_download.side_effect = [("https://example.com/file1.pdb", "/local/path/file1.pdb"), ("https://example.com/file2.pdb", "/local/path/file2.pdb")]

    # Act: Call the bulk_download function
    await bulk_download(remote_file_segments="test_dl.txt", remote_url="https://alphafold.ebi.ac.uk/files/")

    # Assert: Ensure gather_text_lines_from was called with correct parameter
    mock_gather_text_lines_from.assert_called_once_with("test_dl.txt")

    # Assert: Ensure prepare_download was called twice (for each file)
    assert mock_prepare_download.call_count == 2

    # Assert: Ensure async_download_session was awaited for both files
    assert mock_async_download_session.call_count == 2
    mock_async_download_session.assert_any_await("https://example.com/file1.pdb", "/local/path/file1.pdb")
    mock_async_download_session.assert_any_await("https://example.com/file2.pdb", "/local/path/file2.pdb")


@pytest.mark.asyncio(loop_scope="session")
async def test_bulk_download_no_files(mock_async_download_session, mock_prepare_download, mock_gather_empty):
    # Act: Call the bulk_download function with no files
    await bulk_download(remote_file_segments="test_dl.txt", remote_url="https://alphafold.ebi.ac.uk/files/")

    # Assert: Ensure gather_text_lines_from was called
    mock_gather_empty.assert_called_once_with("test_dl.txt")

    # Assert: No download sessions should be initiated if there are no files
    mock_prepare_download.assert_not_awaited()
    mock_async_download_session.assert_not_awaited()


@pytest_asyncio.fixture(loop_scope="session")
async def mock_gather_exception():
    # Ensure async_save_file is correctly mocked as an AsyncMock
    with mock.patch("nnll.download.async_download.gather_text_lines_from", new=mock.AsyncMock(side_effect=FileNotFoundError("File not found"))) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_bulk_download_file_error(mock_async_download_session, mock_prepare_download, mock_gather_exception):
    """Simulate a failure when reading the remote_files.
    Note that there is no global retry in the tested function"""

    with pytest.raises(Exception):
        await bulk_download(remote_file_segments="test_dl.txt", remote_url="https://alphafold.ebi.ac.uk/files/")

    mock_prepare_download.assert_not_awaited()
    mock_async_download_session.assert_not_awaited()


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
