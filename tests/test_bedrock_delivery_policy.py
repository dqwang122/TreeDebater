"""Offline transport checks for caller budgets and isolated delivery policies."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from utils.model import bedrock_claude_completion
from utils.thinking_policy import THINKING
from streaming.listening_prefix import is_cold_start


@pytest.mark.parametrize('cap', [120, 700, 4096, 16384, 24000])
@pytest.mark.parametrize('critical', [False, True])
def test_explicit_token_cap_survives_bedrock_wrapper(monkeypatch, cap, critical):
    transport = Mock(return_value='ok')
    monkeypatch.setattr('utils.model.litellm.completion', transport)
    with THINKING.latency_critical(critical):
        bedrock_claude_completion(model='bedrock/converse/test-claude', messages=[], max_tokens=cap)
    assert transport.call_args.kwargs['max_tokens'] == cap


def test_delivery_policy_is_propagated_only_to_owned_tasks(monkeypatch):
    monkeypatch.delenv('DEBATE_CLAUDE_THINKING', raising=False)
    monkeypatch.setattr('utils.model.litellm.completion', lambda **kwargs: kwargs)
    def request():
        return bedrock_claude_completion(model='bedrock/converse/test-claude', messages=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        with THINKING.latency_critical():
            own = THINKING.bind(request)
            assert 'thinking' not in pool.submit(request).result()
            assert pool.submit(own).result()['thinking'] == {'type': 'between_tools'}
            with THINKING.latency_critical(False):
                assert 'thinking' not in request()
            assert THINKING.critical
        assert not THINKING.critical
        # Captured work keeps the originating policy after its caller returns.
        assert pool.submit(own).result()['thinking'] == {'type': 'between_tools'}
        assert 'thinking' not in pool.submit(request).result()


def test_delivery_policy_restores_after_failure():
    with pytest.raises(ValueError), THINKING.latency_critical():
        raise ValueError('failed task')
    assert not THINKING.critical


@pytest.mark.parametrize('stage,heard,history,expected', [
    ('opening', '', [], True),
    ('opening', 'Opponent spoke', [], False),
    ('opening', '', [{'content': 'Opponent spoke'}], False),
    ('rebuttal', '', [], False),
    ('closing', '', [], False),
])
def test_cold_start_requires_first_opening(stage, heard, history, expected):
    assert is_cold_start(dict(stage=stage, heard_transcript=heard, debate_history=history)) == expected


def test_known_claude_proxy_alias_keeps_refinement_route(monkeypatch):
    import tts_streaming as tts
    monkeypatch.delenv('DEBATE_LLM_API_BASE', raising=False)
    factory = Mock()
    monkeypatch.setattr(tts, 'OpenAI', factory)
    direct = Mock(side_effect=AssertionError('Known proxy alias must not switch provider'))
    monkeypatch.setattr('utils.model.bedrock_claude_completion', direct)
    tts._text_request(None, 'claude-opus-4-8', [], 700)
    assert factory.return_value.chat.completions.create.call_args.kwargs['max_completion_tokens'] == 700
    factory.return_value.close.assert_called_once()
    direct.assert_not_called()


def test_direct_claude_refinement_uses_bedrock_without_openai_client(monkeypatch):
    import tts_streaming as tts
    monkeypatch.delenv('DEBATE_LLM_API_BASE', raising=False)
    monkeypatch.setattr(tts, 'OpenAI', Mock(side_effect=AssertionError('Unexpected OpenAI client')))
    direct = Mock(return_value='response')
    monkeypatch.setattr('utils.model.bedrock_claude_completion', direct)
    assert tts._text_request(None, 'claude-test-direct', [], 700) == 'response'
    assert direct.call_args.kwargs['max_tokens'] == 700
