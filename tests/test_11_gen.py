#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

import asyncio
from enum import Enum
from unittest.mock import patch

import pytest
import pytest_asyncio

from nnll_60 import LIBTYPE_PATH_NAMED, JSONCache


class LibType(Enum):
    """API library constants"""
    #Integers are used to differentiate boolean condition

    OLLAMA   : tuple = (True,"OLLAMA")
    HUB      : tuple = (True,"HUB")
    LM_STUDIO: tuple = (True,"LM_STUDIO")
    CORTEX   : tuple = (True,"CORTEX")
    LLAMAFILE: tuple = (True,"LLAMAFILE")
    VLLM     : tuple = (True,"VLLM")

@pytest_asyncio.fixture(loop_scope='session')
async def mock_signature():
    with patch('nnll_11.dspy.Signature', autospec=True) as mocked:
        yield mocked

@pytest_asyncio.fixture(loop_scope='session')
async def mock_predict():
    with patch('nnll_11.dspy.Predict', autospec=True) as mocked:
        yield mocked

@pytest_asyncio.fixture(loop_scope='session')
async def has_api():
    with patch('nnll_11.has_api', autospec=True) as mocked:
        mocked.return_value = True
        yield mocked

@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_instance(mock_signature, mock_predict):

    mock_predict.return_value = mock_predict
    mock_signature.return_value = mock_signature

    # Create an instance of ChatMachineWithMemory
    max_workers = 8
    from nnll_11 import ChatMachineWithMemory
    return ChatMachineWithMemory(sig=mock_signature, max_workers=max_workers)


chat_machine = asyncio.run(test_chat_instance(mock_signature, mock_predict))



@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_machine_initialization(mock_signature, mock_predict):



    with patch("nnll_11.dspy.LM", autospec=True, return_value="🤡") as mock_lm:
        assert hasattr(chat_machine, "completion")
        assert callable(chat_machine.completion)

@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_machine_generation(mock_signature, mock_predict, has_api):

    data = JSONCache(LIBTYPE_PATH_NAMED)

    data._load_cache()
    api_data = vars(data).get('_cache')
    print(api_data)
    streaming = True
    library = LibType.CORTEX
    api_kwargs = {"model":"🤡"} | api_data[library.value[1]].get('api_kwargs')

    with patch("nnll_11.dspy.LM", autospec=True, return_value="🤡") as mock_lm:

        async for _ in chat_machine.forward(tx_data={"text":"test"}, model="🤡", library=library, streaming=streaming):
            assert _ is not None

    mock_lm.assert_called_once_with(**api_kwargs)
