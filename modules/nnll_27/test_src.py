#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
import unittest
from unittest.mock import patch, MagicMock
import gc
import os

from modules.nnll_27.src import wipe_printer, pretty_tabled_output


class TestPrinterMethods(unittest.TestCase):
    @patch('sys.stdout', new_callable=MagicMock)
    def test_wipe_printer(self, mock_stdout):
        # Sample formatted data
        formatted_data = [
            {"line1": "This is line 1"},
            {"line2": "This is line 2"},
            {"line3": "This is line 3"}
        ]

        # Call the method under test
        self.wipe_printer(*formatted_data)

        # Expected calls to sys.stdout.write
        expected_calls = [
            "\033[F\033[F\033[F",  # Move cursor up three lines
            " " * 175 + "\x1b[1K\r",  # Clear line
            f"{formatted_data[0]}\r\n",
            " " * 175 + "\x1b[1K\r",  # Clear line
            f"{formatted_data[1]}\r\n",
            " " * 175 + "\x1b[1K\r",  # Clear line
            f"{formatted_data[2]}\r\n"
        ]

        # Verify the calls to sys.stdout.write
        actual_calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        self.assertEqual(actual_calls, expected_calls)

    def wipe_printer(self, *formatted_data: dict) -> None:
        import sys
        sys.stdout.write("\033[F" * len(formatted_data))  # ANSI escape codes to move the cursor up 3 lines
        for line_data in formatted_data:
            sys.stdout.write(" " * 175 + "\x1b[1K\r")
            sys.stdout.write(f"{line_data}\r\n")  # Print the lines
        sys.stdout.flush()              # Empty output buffer to ensure the changes are shown


class TestPrinterMethods(unittest.TestCase):
    def test_pretty_tabled_output(self):
        # Sample input data
        title = "Sample Table"
        aggregate_data = {
            'A': 1,
            'B': 2,
            'C': 3,
            'D': 4
        }

        # Expected output format
        expected_cols = '    category     |        A        |        B        |        C        |        D        |'
        expected_horizontal_bar = '  -------------------------------------------------------------------------------------'
        expected_data = '  Sample Table   |        1        |        2        |        3        |        4        |'

        # Call the method under test
        with patch('modules.nnll_27.src.wipe_printer') as mock_wipe_printer:
            result = pretty_tabled_output(title, aggregate_data)

        # Verify the calls to wipe_printer
        mock_wipe_printer.assert_called_once_with(title, expected_cols, expected_horizontal_bar, expected_data)
        self.assertIsNone(result)  # The method should return None
