"""Measured audio, rather than estimated duration, governs strict delivery."""
from unittest.mock import Mock

import pytest

import tts_streaming as tts
from streaming.config import OutputConfig
from test_full_speech import audio


@pytest.mark.parametrize('oversized_seconds', [24.62, 23.85])
@pytest.mark.parametrize('max_refinements', [1, 2])
def test_nonadaptive_expansion_rechecks_audio_and_preserves_safe_fallback(
        audio, monkeypatch, tmp_path, oversized_seconds, max_refinements):
    query, encoded = audio
    prefix = 'Earlier speech has already consumed this budget.'
    original = 'Effort alone does not justify causing harm.'
    expanded = 'Effort alone does not justify causing harm; consequences must matter more than the intensity of trying.'
    shortened = 'Effort cannot justify harm; consequences matter more than trying.'
    durations = {prefix: 37.213, original: 20., expanded: oversized_seconds, shortened: 22.}
    query.side_effect = lambda client, text, *a, **kw: encoded(durations[text])
    # The estimate admits the expansion; its actual speech is over the hard cap.
    monkeypatch.setattr(tts, '_estimate_duration', lambda text, **kw: 22.5 if text == expanded else 20.)
    rewrite = Mock(side_effect=[expanded, shortened])
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    seen = []
    tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 60.,
        config=OutputConfig(adaptive_delivery=False, enforce_duration_budget=True,
                            allow_expansion=True, normalize_seams=False,
                            first_chunk_seconds=37.213, max_refinements=max_refinements,
                            early_max_refinements=max_refinements, speed_adjust_min=1, speed_adjust_max=1),
        tail_supplier=lambda: original,
        on_chunk=lambda i, p, text, duration: seen.append((text, duration)))
    assert [text for text, _ in seen] == [prefix, shortened if max_refinements == 2 else original]
    assert sum(duration for _, duration in seen) <= 61.
    if max_refinements == 2:
        # The second rewrite uses the measured overshoot and asks for fewer words.
        assert rewrite.call_args.args[2] < len(expanded.split())


def test_nonadaptive_prestart_revalidates_actual_audio_when_budget_changes():
    ctx = tts._ChunkRefineContext(Mock(), 'some words', 25, 1, 1, [], '', 'echo', 3, 0, 'last',
        config=OutputConfig(adaptive_delivery=False, enforce_duration_budget=True, allow_expansion=True))
    try:
        candidate = tts._TtsCandidate(0, 'some words', 22.5, None)
        assert ctx.try_adopt(candidate, {'audio_seconds': 24.62})
        ctx.update_target(22.79, 2.279, 1.)
        assert ctx.chosen_cand is None
        assert not ctx.done_event.is_set()
        assert not ctx.try_adopt(candidate, {'audio_seconds': 24.62})
        assert ctx.try_adopt(candidate, {'audio_seconds': 22.})
    finally:
        ctx.executor.shutdown(wait=True)
