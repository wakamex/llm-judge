import os

import llm.plugins
import pytest
from llm.plugins import pm


@pytest.fixture(autouse=True)
def setup_test_env():
    # Store original value and loaded state
    original_value = os.environ.get('LLM_LOAD_PLUGINS')
    original_loaded = llm.plugins._loaded
    
    # Reset loaded state and set env var
    llm.plugins._loaded = False
    os.environ['LLM_LOAD_PLUGINS'] = 'llm-judge'
    
    yield
    
    # Restore original values
    llm.plugins._loaded = original_loaded
    if original_value is not None:
        os.environ['LLM_LOAD_PLUGINS'] = original_value
    else:
        os.environ.pop('LLM_LOAD_PLUGINS', None)

def test_plugin_is_installed():
    # Force plugin loading
    llm.plugins.load_plugins()
    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_judge" in names  # Note: using underscore here since that's the module name
