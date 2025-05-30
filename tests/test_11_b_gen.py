#  # # <!-- // /*  SPDX-License-Identifier: MPL-2.0*/ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import asyncio
from enum import Enum
from unittest.mock import patch

import pytest
import pytest_asyncio

from nnll_60 import LIBTYPE_PATH_NAMED, JSONCache


class LibType(Enum):
    """API library constants"""

    # Integers are used to differentiate boolean condition

    OLLAMA: tuple = (True, "OLLAMA")
    HUB: tuple = (True, "HUB")
    LM_STUDIO: tuple = (True, "LM_STUDIO")
    CORTEX: tuple = (True, "CORTEX")
    LLAMAFILE: tuple = (True, "LLAMAFILE")
    VLLM: tuple = (True, "VLLM")


@pytest_asyncio.fixture(loop_scope="module")
async def mock_signature():
    with patch("nnll_11.dspy.Signature", autospec=True) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="module")
async def mock_predict():
    with patch("nnll_11.dspy.Predict", autospec=True) as mocked:
        yield mocked


@pytest_asyncio.fixture(loop_scope="module")
async def has_api():
    with patch("nnll_15.constants.has_api", autospec=True) as mocked:
        mocked.return_value = True
        yield mocked


class PlaceholderClass:
    model = "🤡"
    library = LibType.CORTEX


PlaceholderClass


@pytest_asyncio.fixture(loop_scope="module")
async def mock_get_api():
    with patch("nnll_11.get_api", autospec=True) as mocked:

        class PlaceholderClass:
            model = "🤡"
            library = LibType.CORTEX

        data = JSONCache(LIBTYPE_PATH_NAMED)
        data._load_cache()
        api_data = vars(data).get("_cache")
        print(f" \n{PlaceholderClass.library.value[1]}  == {api_data.get(PlaceholderClass.library.value[1])}")
        api_kwargs = {"model": PlaceholderClass.model} | api_data[PlaceholderClass.library.value[1]].get("api_kwargs")
        mocked.return_value = api_kwargs
        yield mocked


@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="module")
async def test_chat_instance(mock_signature, mock_predict):
    mock_predict.return_value = mock_predict
    mock_signature.return_value = mock_signature

    # Create an instance of ChatMachineWithMemory
    max_workers = 8
    from nnll_11 import ChatMachineWithMemory

    return ChatMachineWithMemory(max_workers=max_workers)  # sig=mock_signature,


@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="module")
async def test_chat_machine_initialization(mock_signature, mock_predict):
    chat_machine = await test_chat_instance(mock_signature, mock_predict)
    with patch("nnll_11.dspy.LM", autospec=True, return_value="🤡") as mock_lm:
        assert hasattr(chat_machine, "mir_db")
        assert chat_machine.mir_db.database is not None
        assert callable(chat_machine.mir_db.find_path)
        assert hasattr(chat_machine, "factory")
        assert chat_machine.factory is not None
        assert callable(chat_machine.factory.create_pipeline)
        assert callable(chat_machine)
        assert callable(chat_machine.forward)


@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="module")
async def test_chat_machine_generation(mock_get_api, mock_signature, mock_predict, has_api):
    # from nnll_11 import ChatMachineWithMemory

    # chat_machine = ChatMachineWithMemory(max_workers=8)
    chat_machine = await test_chat_instance(mock_signature, mock_predict)
    assert chat_machine.pipe is None
    # chat_machine.pipe = None

    with patch("nnll_11.dspy.LM", autospec=True, return_value="🤡") as mock_lm:
        chat_machine(reg_entries=PlaceholderClass, sig=mock_signature)

        assert callable(chat_machine.pipe)
        assert chat_machine.pipe is not None
        mock_lm.assert_called_once_with(**mock_get_api.return_value)
