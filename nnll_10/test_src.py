from unittest import mock
import pytest
import pytest_asyncio
# from textual import events


from .__main__ import Combo

# @pytest.mark.asyncio
# async def test_initial_state():
#     """Test that the initial state of the app is correct."""
#     app = Combo()

# Focus validation check
# panel_focus = self.query_one("#response_panel")  # check response panel focus
# display_focus = self.query_one("#responsive_display").has_focus_within  # check container focus


# @pytest.mark.asyncio(loop_scope="session")
# async def test_reactive_screen(mock_exit):
#     """Screen rotation function"""
#     app = Combo()
#     async with app.run_test() as pilot:
#         # this should make the screen be WideScreen
#         await pilot.resize_terminal(40, 20)
#         assert app.screen == app.get_screen("widescreen")

#         # this should make the screen be TallScreen
#         await pilot.resize_terminal(39, 20)
#         assert app.screen == app.get_screen("tallscreen")


app = Combo()


@pytest_asyncio.fixture(loop_scope="session")
def mock_exit():
    """Create an app session"""
    # Mock aiohttp.ClientSession properly
    with mock.patch.object(app, "exit", autospec=True) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_exit(mock_exit):  # pylint: disable=redefined-outer-name
    """Test that the app exits correctly."""

    async with app.run_test() as pilot:
        # ensure no accidental triggers
        await pilot.press("escape", "tab", "escape")
        mock_exit.assert_not_called()

        await pilot.press("escape", "k", "escape", "k", "k", "escape")
        mock_exit.assert_not_called()

        # ensure exit
        await pilot.press("escape", "escape")
        mock_exit.assert_called_once()

        # print(app.screen.get_widget_by_id("Layout"))
        # assert app.screen == app.screen.get_widget_by_id("footer") == Footer(id="footer")


#         assert
#         assert "Submit" in app.screen.get_widget_by_id("submit_button").label
#         assert app.screen.get_widget_by_id("greeting_output").renderable == ""


# @pytest.mark.asyncio
# async def test_exit():
#     """Test pressing exit has the desired result."""
#     app = Combo()
#     async with app.run_test() as pilot:
#         await pilot.press("escape", "tab")
#         assert app.exit

#         await pilot.press("escape", "escape")
#         assert app.exit


# # async with app.run_test(size=(100, 50)) as pilot:
