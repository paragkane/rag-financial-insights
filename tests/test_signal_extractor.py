"""Tests for signal extractor — unit tests for parsing/validation, not API calls."""

import pytest

from src.extraction.signal_extractor import (
    FilingSignal,
    SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    _cache_path,
    _load_cached,
    _save_cache,
)


class TestFilingSignalValidation:
    def test_valid_signal(self):
        signal = FilingSignal(
            sentiment_score=0.65,
            guidance_direction="raised",
            guidance_magnitude=2.5,
            guidance_confidence=0.8,
            risk_flags=["competition", "macro"],
            earnings_framing="beat",
            tone="optimistic",
            key_themes=["strong revenue growth", "cloud momentum"],
            reasoning="Revenue exceeded expectations with strong cloud growth.",
        )
        assert signal.sentiment_score == 0.65
        assert signal.tone == "optimistic"

    def test_sentiment_score_bounds(self):
        with pytest.raises(Exception):
            FilingSignal(
                sentiment_score=1.5,  # out of bounds
                guidance_direction="none",
                guidance_magnitude=0.0,
                guidance_confidence=0.0,
                risk_flags=[],
                earnings_framing="not_mentioned",
                tone="neutral",
                key_themes=[],
                reasoning="test",
            )

    def test_negative_sentiment_score_bounds(self):
        with pytest.raises(Exception):
            FilingSignal(
                sentiment_score=-1.5,  # out of bounds
                guidance_direction="none",
                guidance_magnitude=0.0,
                guidance_confidence=0.0,
                risk_flags=[],
                earnings_framing="not_mentioned",
                tone="neutral",
                key_themes=[],
                reasoning="test",
            )

    def test_invalid_guidance_direction(self):
        with pytest.raises(Exception):
            FilingSignal(
                sentiment_score=0.0,
                guidance_direction="unknown",  # invalid
                guidance_magnitude=0.0,
                guidance_confidence=0.0,
                risk_flags=[],
                earnings_framing="not_mentioned",
                tone="neutral",
                key_themes=[],
                reasoning="test",
            )

    def test_invalid_tone(self):
        with pytest.raises(Exception):
            FilingSignal(
                sentiment_score=0.0,
                guidance_direction="none",
                guidance_magnitude=0.0,
                guidance_confidence=0.0,
                risk_flags=[],
                earnings_framing="not_mentioned",
                tone="angry",  # invalid
                key_themes=[],
                reasoning="test",
            )

    def test_model_dump_roundtrip(self):
        signal = FilingSignal(
            sentiment_score=-0.3,
            guidance_direction="lowered",
            guidance_magnitude=1.0,
            guidance_confidence=0.5,
            risk_flags=["liquidity"],
            earnings_framing="miss",
            tone="cautious",
            key_themes=["declining margins"],
            reasoning="Margins compressed due to macro headwinds.",
        )
        data = signal.model_dump()
        reconstructed = FilingSignal(**data)
        assert reconstructed.sentiment_score == signal.sentiment_score
        assert reconstructed.tone == signal.tone


class TestSignalCache:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.extraction.signal_extractor.SIGNALS_CACHE_DIR", tmp_path
        )
        test_signal = {"ticker": "AAPL", "date": "2025-01-30", "sentiment_score": 0.5}
        _save_cache("AAPL", "2025-01-30", test_signal)
        loaded = _load_cached("AAPL", "2025-01-30")
        assert loaded == test_signal

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.extraction.signal_extractor.SIGNALS_CACHE_DIR", tmp_path
        )
        assert _load_cached("FAKE", "2099-01-01") is None

    def test_cache_path_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.extraction.signal_extractor.SIGNALS_CACHE_DIR", tmp_path
        )
        path = _cache_path("aapl", "2025-01-30")
        assert path.name == "AAPL_2025-01-30.json"


class TestPromptTemplates:
    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 50
        assert "financial" in SYSTEM_PROMPT.lower()

    def test_extraction_prompt_has_placeholder(self):
        assert "{text}" in EXTRACTION_PROMPT

    def test_extraction_prompt_requests_json(self):
        assert "JSON" in EXTRACTION_PROMPT
