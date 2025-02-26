from unittest.mock import patch, mock_open, AsyncMock
import aiohttp
import asyncio
from aioresponses import aioresponses
from dataset_dl import main, save_file


@patch("builtins.open", new_callable=AsyncMock, read_data="file1\nfile2\n")
def test_main_reads_file(mock_file):
    loop = asyncio.get_event_loop()
    with patch("aiofiles.open", new_callable=AsyncMock) as mock_open_file:
        loop.run_until_complete(main())

        # Ensure the file was opened and read correctly
        mock_file.assert_called_once_with("test_dl.txt", "r")

    # We also need to assert that url_segments are generated correctly from this content
    # Since this is async, we check in a broader test later.


@patch("aiohttp.ClientSession", new_callable=AsyncMock)
async def test_main_downloads_files(mock_session):
    # Simulate a successful response from the server
    mock_response = AsyncMock()
    mock_response.content.read.return_value = b"mocked_content"
    mock_session.get.return_value.__aenter__.return_value = mock_response

    # Mock prepare_storage to return expected values without actual file system interaction
    with patch("dataset_dl.prepare_storage", new_callable=AsyncMock) as mock_prepare:
        mock_prepare.return_value = ("file1.pdb", "/mocked/path/file1.pdb")

        await main()

        # Ensure that the session.get method was called with correct URL
        mock_session.get.assert_called_with("https://alphafold.ebi.ac.uk/files/file1.pdb")

        # Ensure download function is invoked correctly (concurrent_download)
        assert mock_prepare.call_count == 2  # Two files ("file1", "file2") based on mocked file content


@patch("aiohttp.ClientSession", new_callable=AsyncMock)
async def test_main_retries_on_download_failures(mock_session):
    # Simulate a failure (HTTPError) when trying to download content
    mock_response = AsyncMock()
    mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(None, None, status=404)
    mock_session.get.return_value.__aenter__.return_value = mock_response

    with patch("dataset_dl.prepare_storage", new_callable=AsyncMock) as mock_prepare:
        # Mock prepare_storage to avoid real file operations
        mock_prepare.return_value = ("file1.pdb", "/mocked/path/file1.pdb")

        # We expect retries on download failure (HTTPError)
        await main()

        # Check that the retry mechanism was invoked 3 times for a failed download
        assert mock_session.get.call_count == 3


@patch("builtins.open", new_callable=AsyncMock)
async def test_main_retries_on_save_failures(mock_file):
    # Simulate an OSError when saving content to disk
    with patch("dataset_dl.save_file", side_effect=[OSError, b"content"]):
        with patch("aiohttp.ClientSession") as mock_session:
            with aioresponses() as mock_response:
                async with aiohttp.ClientSession() as session:
                    session = AsyncMock()
                    mock_response = session
                    mock_response.content.read.return_value = b"mocked_content"

                    mock_session.get.return_value.__aenter__.return_value = mock_response

            await main()

            # Check that the retry mechanism for file saving was invoked
            assert save_file.call_count == 3  # Retry limit is set to 3


@patch("aiohttp.ClientSession", new_callable=AsyncMock)
async def test_main_executes_all_tasks(mock_session):
    # Simulate successful responses for multiple files (file1.pdb & file2.pdb)
    mock_response = AsyncMock()
    mock_response.content.read.return_value = b"mocked_content"
    mock_session.get.return_value.__aenter__.return_value = mock_response

    with patch("dataset_dl.prepare_storage", new_callable=AsyncMock) as mock_prepare:
        # Mock prepare_storage to avoid real file system operations
        mock_prepare.side_effect = [("file1.pdb", "/mocked/path/file1.pdb"), ("file2.pdb", "/mocked/path/file2.pdb")]

        await main()

        # Ensure that both download and save tasks were gathered correctly
        assert mock_session.get.call_count == 2  # One for each file

        # Check if asyncio.gather was invoked with the correct number of tasks (file downloads + saves)


@patch("aiohttp.ClientSession", new_callable=AsyncMock)
@patch("builtins.open", new_callable=mock_open, read_data="file1\nfile2\n")
async def test_main_full_integration(mock_file, mock_session):
    # Simulate successful file download and content saving
    mock_response = AsyncMock()
    mock_response.content.read.return_value = b"mocked_content"
    mock_session.get.return_value.__aenter__.return_value = mock_response

    with patch("dataset_dl.prepare_storage", new_callable=AsyncMock) as mock_prepare:
        # Mock prepare_storage to avoid real file system operations
        mock_prepare.side_effect = [("file1.pdb", "/mocked/path/file1.pdb"), ("file2.pdb", "/mocked/path/file2.pdb")]

        await main()

        # Ensure that all necessary steps were executed correctly (file reading, downloading, saving)
        assert mock_file.call_count == 1
        assert mock_session.get.call_count == 2  # Two files to download
