# import os
# from datetime import datetime
# from logging import DEBUG
# from unittest.mock import Mock, patch

# import nnll_01
# from nnll_01 import info_monitor as nfo


# def test_assign_logging_to():
#     """Mock logging and rich, then test log output"""
#     mock_get_logger = Mock()
#     mock_file_handler = Mock()
#     mock_formatter = Mock()
#     mock_stream_handler = Mock()
#     mock_rich_handler = Mock()
#     mock_console = Mock()

#     with (
#         patch("logging.getLogger", mock_get_logger),
#         patch("logging.FileHandler", mock_file_handler),
#         patch("logging.Formatter", mock_formatter),
#         patch("logging.StreamHandler", mock_stream_handler),
#         patch("rich.logging.RichHandler", mock_rich_handler),
#         patch("rich.console.Console", mock_console),
#     ):
#         # Set up test inputs
#         file_name = "test_log"
#         folder_path_named = "logs"

#         # Call the function
#         logger = nnll_01.assign_logging_to(file_name, folder_path_named)

#         # Assert getLogger was called with correct name
#         mock_get_logger.assert_called_with(nnll_01.__name__)

#         # Check if FileHandler is created with expected path
#         expected_file_name = f"{file_name}{datetime.now().strftime('%Y%m%d')}"
#         expected_path = os.path.join(folder_path_named, expected_file_name)
#         mock_file_handler.assert_called_with(expected_path, "a+", encoding="utf-8")

#         # Ensure handlers were added to the root logger
#         assert mock_get_logger.return_value.addHandler.call_count == 0

#         # Check if RichHandler was configured correctly
#         mock_rich_handler.assert_called_with(
#             level=DEBUG,
#             rich_tracebacks=True,
#             tracebacks_show_locals=True,
#             console=mock_console.return_value,
#             show_time=True,
#             log_time_format="%H:%M:%S",
#         )

#         # Ensure the logger instance is returned
#         assert logger == mock_get_logger.return_value
#         assert mock_get_logger.return_value.addHandler.call_count == 0


# def test_default_parameters():
#     """Call and ensire file exists"""
#     nnll_01.assign_logging_to()
#     # Check if file name and folder path use defaults ".nnll" and "log"
#     expected_file_name = f".nnll{datetime.now().strftime('%Y%m%d')}"
#     expected_path = os.path.join("log", expected_file_name)
#     assert os.path.exists(expected_path)


# def test_output():
#     """This test will fail if mock is running
#     so, ensure override for pytest circumstances"""
#     nnll_01.assign_logging_to()
#     nfo("output")
