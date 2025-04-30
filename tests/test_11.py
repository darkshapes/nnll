#  # # <!-- // /*  SPDX-License-Identifier: LAL-1.3 */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

from collections import namedtuple
import unittest
import pytest
from unittest.mock import MagicMock, patch, Mock
import pytest_asyncio
from enum import Enum
from unittest.mock import patch

def has_api(name: str):
    # Mock implementation that always returns True
    return True

class LibType(Enum):
    """API library constants"""
    #Integers are used to differentiate boolean condition

    OLLAMA   : tuple = (has_api("OLLAMA"),"OLLAMA")
    HUB      : tuple = (has_api("HUB"),"HUB")
    LM_STUDIO: tuple = (has_api("LM_STUDIO"),"LM_STUDIO")
    CORTEX   : tuple = (has_api("CORTEX"),"CORTEX")
    LLAMAFILE: tuple = (has_api("LLAMAFILE"),"LLAMAFILE")
    VLLM     : tuple = (has_api("VLLM"),"VLLM")

@pytest_asyncio.fixture(loop_scope="session")
async def mock_has_api():
    with patch("nnll_15.constants.has_api", return_value=True) as mocked:
        yield mocked

@pytest.mark.filterwarnings("ignore:open_text")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:")
@patch('nnll_15.constants.has_api', side_effect=lambda x: False)
def test_libtype(mock_has_api):
    assert LibType.OLLAMA.value[0] is True
    assert LibType.HUB.value[0] is True
    assert LibType.LM_STUDIO.value[0] is True
    assert LibType.CORTEX.value[0] is True
    assert LibType.LLAMAFILE.value[0] is True
    assert LibType.VLLM.value[0] is True

@pytest_asyncio.fixture(loop_scope="session",name="mock_config")
def mock_deco():
    def decorator(func):
        def wrapper(*args, **kwargs):
            data = {
                "OLLAMA": {"api_kwargs": {"key1": "value1"}},
                "HUB": {"api_kwargs": {"key2": "value2"}},
                "LM_STUDIO": {"api_kwargs": {"key3": "value3"}},
                "CORTEX": {"api_kwargs": {"key4": "value4"}},
                "LLAMAFILE": {"api_kwargs": {"key5": "value5"}},
                "VLLM": {"api_kwargs": {"key6": "value6"}},
            }
            return data
        return wrapper
    return decorator

def libtype_config_fixture():
    with patch("nnll_11.LIBTYPE_CONFIG", MagicMock()):
        yield mock_deco

@pytest_asyncio.fixture(loop_scope="session")
async def test_get_api(mock_has_api, mock_config,):
    from nnll_11 import get_api
    from nnll_01 import nfo
    import os
    model = "🤡"
    library = LibType.OLLAMA
    with patch("nnll_11.has_api", autocast=True, return_value=mock_has_api):
        value = namedtuple("OLLAMA",["true", "OLLAMA"])
        nfo(value)
        with patch("nnll_11.LibType", autocast=True, return_value=value):
            req_form = get_api(model, library)
            yield req_form


@pytest.mark.asyncio(loop_scope="session")
async def test_lookup_libtypes(mock_has_api,test_get_api):
    from nnll_11 import get_api
    from nnll_60 import JSONCache

    import os
    model = "🤡"

    for library in LibType.__members__.keys():
        library = getattr(LibType, library)
        # with patch("nnll_11.LibType", autocast=True):
        req_form = await get_api(model, library)
        test_path = os.path.dirname(os.path.abspath(__file__))
        data = JSONCache(os.path.join(os.path.dirname(test_path),"nnll_60", "config", "libtype.json"))
        data._load_cache()
        expected = vars(data).get('_cache')
        assert expected
        assert req_form == {
            "model": model,
            **expected[library.value[1]].get('api_kwargs')

            }
