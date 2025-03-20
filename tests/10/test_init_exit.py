#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=redefined-outer-name

from unittest import mock

import pytest
import pytest_asyncio

from package.main_screen import MainScreen

from package.__main__ import Combo


@pytest.mark.asyncio(loop_scope="session")
async def test_initial_state(app=Combo()):
    """Test that the initial state of the app is correct."""

    async with app.run_test() as pilot:
        ui_elements = list(pilot.app.query("*"))
        assert isinstance(ui_elements[0], MainScreen)  # root


@pytest_asyncio.fixture(loop_scope="session")
async def mock_app():
    """Create an instance of the app"""
    app = Combo()
    yield app


@pytest_asyncio.fixture(loop_scope="session")
async def mock_exit(mock_app):
    """Create a decoy app exit"""

    with mock.patch.object(mock_app, "exit", autospec=True) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_exit(mock_app, mock_exit):
    """Test that the app exits correctly."""

    async with mock_app.run_test() as pilot:
        # ensure no accidental triggers
        await pilot.press("escape", "tab", "escape")
        mock_exit.assert_not_called()

        await pilot.press("escape", "k", "escape", "k", "k", "escape")
        mock_exit.assert_not_called()

        # ensure exit
        await pilot.press("escape", "escape")
        mock_exit.assert_called_once()
