import pytest

from package.__main__ import Combo


@pytest.mark.asyncio(loop_scope="session")
async def test_responsive_layout(app=Combo()):
    """Screen rotation function"""
    async with app.run_test() as pilot:
        await pilot.resize_terminal(40, 20)
        expected = "app-grid-horizontal"
        assert app.query_one("#app-grid").classes == frozenset({expected})

        await pilot.resize_terminal(39, 20)
        expected = "app-grid-vertical"
        assert app.query_one("#app-grid").classes == frozenset({expected})
