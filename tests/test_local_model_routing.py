"""Local proxy routing without network calls."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import Agent, AgentConfig
from utils import model as model_utils


@pytest.mark.parametrize('model', ['google.gemma-4-26b-a4b', 'gpt-6.1-sol', 'gpt-5.6-sol', 'claude-opus-4-8', 'nvidia.nemotron-super-3-120b'])
@pytest.mark.parametrize('configured', [False, True])
def test_agent_and_helper_route_local_models(monkeypatch, model, configured):
    monkeypatch.delenv('DEBATE_LLM_API_BASE', raising=False)
    monkeypatch.delenv('DEBATE_LLM_API_KEY', raising=False)
    base, key = 'http://127.0.0.1:4000/v1', 'local-proxy'
    if configured:
        base, key = 'http://localhost:4321/v1', 'test-key'
        monkeypatch.setenv('DEBATE_LLM_API_BASE', base)
        monkeypatch.setenv('DEBATE_LLM_API_KEY', key)
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='Ready'), finish_reason='stop')])
    transport = Mock(return_value=response)
    monkeypatch.setattr(model_utils.litellm, 'completion', transport)

    agent = Agent(AgentConfig(model=model))
    agent.client(messages=[{'role': 'user', 'content': 'Prepare'}])
    assert model_utils.HelperClient('Prepare', model=model, json_mode=False) == ['Ready']
    assert transport.call_count == 2
    for call in transport.call_args_list:
        options = call.kwargs
        assert options['model'] == 'openai/' + model
        assert options['api_base'] == base
        assert options['api_key'] == key
        if model == 'gpt-6.1-sol':
            assert 'temperature' not in options and 'stop' not in options
            assert options['extra_body'] == {'reasoning_effort': 'low', 'service_tier': 'default'}
