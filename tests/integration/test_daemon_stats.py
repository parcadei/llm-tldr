"""Tests for daemon session stats tracking (TDD - write tests first).

These tests define the expected behavior for:
1. Per-session token tracking
2. JSONL persistence
3. Multi-session isolation
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTokenCounting:
    """Tests for accurate token counting with tiktoken."""

    def test_count_tokens_basic(self):
        """Should count tokens accurately for simple text."""
        from tldr.stats import count_tokens

        # "Hello world" should be ~2-3 tokens
        result = count_tokens("Hello world")
        assert 2 <= result <= 4

    def test_count_tokens_code(self):
        """Should count tokens for code content."""
        from tldr.stats import count_tokens

        code = """def hello():
    print("Hello, world!")
    return 42
"""
        result = count_tokens(code)
        # Code typically has more tokens per character
        assert result > 5

    def test_count_tokens_empty(self):
        """Should return 0 for empty string."""
        from tldr.stats import count_tokens

        assert count_tokens("") == 0

    def test_count_tokens_large_file(self):
        """Should handle large files efficiently."""
        from tldr.stats import count_tokens

        # 100KB of text
        large_text = "hello world " * 10000
        result = count_tokens(large_text)
        assert result > 0


class TestSessionStats:
    """Tests for per-session stats tracking."""

    def test_session_stats_initialization(self):
        """New session should start with zero stats."""
        from tldr.stats import SessionStats

        stats = SessionStats(session_id="test-123")
        assert stats.session_id == "test-123"
        assert stats.raw_tokens == 0
        assert stats.tldr_tokens == 0
        assert stats.requests == 0

    def test_session_stats_record_request(self):
        """Should accumulate stats for each request."""
        from tldr.stats import SessionStats

        stats = SessionStats(session_id="test-123")
        stats.record_request(raw_tokens=1000, tldr_tokens=150)

        assert stats.raw_tokens == 1000
        assert stats.tldr_tokens == 150
        assert stats.requests == 1

        stats.record_request(raw_tokens=500, tldr_tokens=75)

        assert stats.raw_tokens == 1500
        assert stats.tldr_tokens == 225
        assert stats.requests == 2

    def test_session_stats_savings(self):
        """Should calculate savings percentage correctly."""
        from tldr.stats import SessionStats

        stats = SessionStats(session_id="test-123")
        stats.record_request(raw_tokens=1000, tldr_tokens=100)

        assert stats.savings_tokens == 900
        assert stats.savings_percent == 90.0

    def test_session_stats_to_dict(self):
        """Should serialize to dict for JSON."""
        from tldr.stats import SessionStats

        stats = SessionStats(session_id="test-123")
        stats.record_request(raw_tokens=1000, tldr_tokens=100)

        d = stats.to_dict()
        assert d["session_id"] == "test-123"
        assert d["raw_tokens"] == 1000
        assert d["tldr_tokens"] == 100
        assert d["requests"] == 1
        assert "timestamp" in d


class TestStatsStore:
    """Tests for JSONL persistence."""

    def test_stats_store_append(self):
        """Should append stats to JSONL file."""
        from tldr.stats import SessionStats, StatsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = StatsStore(Path(tmpdir) / "stats.jsonl")

            stats = SessionStats(session_id="test-123")
            stats.record_request(raw_tokens=1000, tldr_tokens=100)

            store.append(stats)

            # Verify file exists and has content
            assert store.path.exists()
            lines = store.path.read_text().strip().split("\n")
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["session_id"] == "test-123"

    def test_stats_store_multiple_sessions(self):
        """Should handle multiple sessions in same file."""
        from tldr.stats import SessionStats, StatsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = StatsStore(Path(tmpdir) / "stats.jsonl")

            # Session 1
            stats1 = SessionStats(session_id="session-1")
            stats1.record_request(raw_tokens=1000, tldr_tokens=100)
            store.append(stats1)

            # Session 2
            stats2 = SessionStats(session_id="session-2")
            stats2.record_request(raw_tokens=2000, tldr_tokens=200)
            store.append(stats2)

            lines = store.path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_stats_store_get_session_history(self):
        """Should retrieve history for specific session."""
        from tldr.stats import SessionStats, StatsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = StatsStore(Path(tmpdir) / "stats.jsonl")

            # Multiple entries for same session
            for i in range(3):
                stats = SessionStats(session_id="test-session")
                stats.record_request(raw_tokens=1000 * (i + 1), tldr_tokens=100 * (i + 1))
                store.append(stats)

            # Different session
            other = SessionStats(session_id="other-session")
            other.record_request(raw_tokens=500, tldr_tokens=50)
            store.append(other)

            history = store.get_session_history("test-session")
            assert len(history) == 3

    def test_stats_store_get_totals(self):
        """Should calculate all-time totals."""
        from tldr.stats import SessionStats, StatsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = StatsStore(Path(tmpdir) / "stats.jsonl")

            stats1 = SessionStats(session_id="s1")
            stats1.record_request(raw_tokens=1000, tldr_tokens=100)
            store.append(stats1)

            stats2 = SessionStats(session_id="s2")
            stats2.record_request(raw_tokens=2000, tldr_tokens=200)
            store.append(stats2)

            totals = store.get_totals()
            assert totals["raw_tokens"] == 3000
            assert totals["tldr_tokens"] == 300
            assert totals["requests"] == 2


class TestDaemonStatsIntegration:
    """Tests for daemon stats integration."""

    def test_daemon_tracks_session_stats(self):
        """Daemon should track stats per session ID."""
        from tldr.daemon import TLDRDaemon

        # This tests that the daemon can accept and track session IDs
        # Implementation will add session tracking to daemon
        pass  # TODO: Implement after daemon changes

    def test_daemon_status_includes_session_stats(self):
        """Status command should include session-specific stats."""
        # Tests that status returns per-session token counts
        pass  # TODO: Implement after daemon changes

    def test_daemon_writes_stats_on_shutdown(self):
        """Daemon should persist stats to JSONL on graceful shutdown."""
        pass  # TODO: Implement after daemon changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
