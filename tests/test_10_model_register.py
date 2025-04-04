#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=redefined-outer-name, unused-argument

from unittest import mock
import pytest
import pytest_asyncio
import datetime

from nnll_10.package.model_register import from_ollama_cache
from nnll_10.package import __main__


@pytest.fixture(scope="session")
def test_app_instance():
    app = __main__.Combo()
    return app


class Model:
    """Mock ollama Model class"""

    def __init__(self, model=None, modified_at=None, digest=None, size=None, details=None):
        self.model = model
        self.modified_at = modified_at
        self.digest = digest
        self.size = size
        self.details = details


class ModelDetails:
    """Mock ollama ModelDetails class"""

    def __init__(self, parent_model=None, format=None, family=None, families=None, parameter_size=None, quantization_level=None):
        self.parent_model = parent_model
        self.format = format
        self.family = family
        self.families = families
        self.parameter_size = parameter_size
        self.quantization_level = quantization_level


class ListResponse:
    """Mock ollama ListResponse class"""

    def __init__(self, models=None):
        self.models = models


@pytest.fixture(scope="session")
def mock_ollama_data():
    """Mock ollama response"""
    with mock.patch("ollama.list", new_callable=mock.MagicMock()) as mock_get_registry_data:
        data = ListResponse(
            models=[
                Model(
                    model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q8_0",
                    modified_at=datetime.datetime(2025, 3, 19, 12, 21, 19, 112890, tzinfo=None),
                    digest="965289b1e3e63c66bfc018051b6a907b2f0b18620d5721dd1cdfad759b679a2c",
                    size=29565711760,
                    details=ModelDetails(parent_model="", format="gguf", family="gemma3", families=["gemma3"], parameter_size="27B", quantization_level="unknown"),
                ),
                Model(
                    model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q5_K_M",
                    modified_at=datetime.datetime(2025, 3, 18, 12, 13, 57, 294851, tzinfo=None),
                    digest="82c7d241b764d0346f382a9059a7b08056075c7bc2d81ac21dfa20d525556b16",
                    size=20129415184,
                    details=ModelDetails(parent_model="", format="gguf", family="gemma3", families=["gemma3"], parameter_size="27B", quantization_level="unknown"),
                ),
                Model(
                    model="hf.co/bartowski/RekaAI_reka-flash-3-GGUF:Q5_K_M",
                    modified_at=datetime.datetime(2025, 3, 13, 18, 28, 57, 859962, tzinfo=None),
                    digest="43d35cd4e25e90f9cbb33585f60823450bd1f279c4703a1b2831a9cba73e60e4",
                    size=15635474582,
                    details=ModelDetails(parent_model="", format="gguf", family="llama", families=["llama"], parameter_size="20.9B", quantization_level="unknown"),
                ),
            ]
        )
        mock_get_registry_data.return_value = data
        yield mock_get_registry_data


@pytest.mark.asyncio(loop_scope="session")
async def test_model_selected_on_init(mock_ollama_data, test_app_instance):
    """Test that a model is available"""

    models = from_ollama_cache()
    expected = models[next(iter(models))]
    async with test_app_instance.run_test() as pilot:
        screen = pilot.app.screen.query("#centre-frame")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert node.current_model == expected


@pytest_asyncio.fixture(loop_scope="session")
def mock_scribe_response():
    """Create a decoy chat machine"""
    with mock.patch("nnll_10.package.fold.ResponsePanel.scribe_response", mock.MagicMock()) as mocked:
        yield mocked


@pytest.mark.asyncio(loop_scope="session")
async def test_status_color_remains(mock_ollama_data, app=__main__.Combo()):
    """Control test for status color reflected in text line"""
    async with app.run_test() as pilot:
        expected = frozenset({"tag_line"})
        screen = pilot.app.screen.query("#tag_line")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert node.classes == expected


@pytest.mark.asyncio(loop_scope="session")
async def test_status_color_continues_to_remain(mock_ollama_data, mock_scribe_response, app=__main__.Combo()):
    """Ensure cannot accidentally trigger"""
    async with app.run_test() as pilot:
        # ensure no accidental triggers
        await pilot.press("grave_accent", "tab")
        expected = frozenset({"tag_line"})
        screen = pilot.app.screen.query("#tag_line")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert node.classes == expected
        mock_scribe_response.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_status_color_changes(mock_scribe_response, mock_ollama_data, app=__main__.Combo()):
    """Ensure color changes when activated"""
    async with app.run_test() as pilot:
        text_insert = "chunk"
        screen = pilot.app.screen.query("#message_panel")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        node.insert(text_insert)
        screen = pilot.app.screen.query("#centre-frame")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        node.focus()
        await pilot.press("tab", "grave_accent")
        expected = frozenset({"tag_line"})
        screen = pilot.app.screen.query("#tag_line")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert node.classes == expected

        mock_scribe_response.assert_called_once()
        last_model = node.current_model

        # test color reverts
        node.current_model = "inactive"
        screen = pilot.app.screen.query("#response_panel")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert not node.is_generating
        expected = {"tag_line"}
        screen = pilot.app.screen.query("#tag_line")
        node = pilot.app.screen.query_one(screen.nodes[0].__class__.__name__)
        assert node.classes == expected
        mock_scribe_response.assert_called_with(last_model, text_insert)

        pilot.app.exit()
