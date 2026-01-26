"""Tests for daemon query command with --json flag.

Tests for enhanced daemon query command that supports:
- Simple command: tldr daemon query ping
- JSON payload: tldr daemon query --json '{"cmd":"semantic",...}'
- Error handling for invalid JSON
- Error handling for missing arguments

Note: These tests are for the --json flag feature added in commit d7ec148.
Run these tests against the PR branch containing that commit, or apply the commit
to verify the feature works correctly.

Per PR requirements, all tests verify:
- Valid JSON payload parsing and daemon submission
- Invalid JSON error handling
- Missing arguments error (neither cmd nor --json)
- Both arguments provided (--json precedence)
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestDaemonQueryJsonFlag:
    """Tests for daemon query command --json flag functionality."""

    def test_query_with_valid_json_payload(self, capsys):
        """Should parse and submit valid JSON payload to daemon."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        json_payload = '{"cmd": "semantic", "action": "search", "query": "test"}'
        expected_response = {"status": "ok", "results": []}

        # Create a mock function that returns expected_response
        mock_query_func = MagicMock(return_value=expected_response)

        # Patch the daemon module's query_daemon before main() runs
        # We need to patch it in the tldr.daemon module, not cli
        with patch("tldr.daemon.query_daemon", mock_query_func):
            # Simulate CLI call: tldr daemon query --json '{"cmd": "semantic",...}'
            test_args = [
                "daemon",
                "query",
                "--json",
                json_payload,
                "--project",
                str(project_path),
            ]

            with patch("sys.argv", ["tldr", *test_args]):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify query_daemon was called with parsed JSON
            mock_query_func.assert_called_once()
            called_with_project, called_with_command = mock_query_func.call_args[0]
            assert called_with_project == project_path
            assert called_with_command == json.loads(json_payload)

            # Verify output is formatted JSON
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output == expected_response

    def test_query_with_simple_command(self, capsys):
        """Should fall back to simple command when --json not provided."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        expected_response = {"status": "ok", "message": "pong"}

        mock_query_func = MagicMock(return_value=expected_response)

        with patch("tldr.daemon.query_daemon", mock_query_func):
            # Simulate CLI call: tldr daemon query ping
            test_args = ["daemon", "query", "ping", "--project", str(project_path)]

            with patch("sys.argv", ["tldr"] + test_args):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify query_daemon was called with {"cmd": "ping"}
            mock_query_func.assert_called_once()
            called_with_project, called_with_command = mock_query_func.call_args[0]
            assert called_with_project == project_path
            assert called_with_command == {"cmd": "ping"}

            # Verify output is formatted JSON
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output == expected_response

    def test_query_with_invalid_json_error(self, capsys):
        """Should display error for invalid JSON in --json flag."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        invalid_json = '{"cmd": "semantic", invalid}'

        mock_query_func = MagicMock()

        with patch("tldr.daemon.query_daemon", mock_query_func):
            # Simulate CLI call with invalid JSON
            test_args = [
                "daemon",
                "query",
                "--json",
                invalid_json,
                "--project",
                str(project_path),
            ]

            with patch("sys.argv", ["tldr"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()

            # Should exit with error code 1
            assert exc_info.value.code == 1

            # Verify query_daemon was NOT called
            mock_query_func.assert_not_called()

            # Verify error message contains JSON decode error
            captured = capsys.readouterr()
            assert "Error: invalid JSON for --json" in captured.err

    def test_query_missing_both_arguments_error(self, capsys):
        """Should display error when neither cmd nor --json is provided."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()

        mock_query_func = MagicMock()

        with patch("tldr.daemon.query_daemon", mock_query_func):
            # Simulate CLI call without cmd or --json
            test_args = ["daemon", "query", "--project", str(project_path)]

            with patch("sys.argv", ["tldr"] + test_args):
                with pytest.raises(SystemExit) as exc_info:
                    main()

            # Should exit with error code 1
            assert exc_info.value.code == 1

            # Verify query_daemon was NOT called
            mock_query_func.assert_not_called()

            # Verify error message
            captured = capsys.readouterr()
            assert "Error: either CMD or --json must be provided" in captured.err

    def test_query_json_precedence_over_cmd(self):
        """Should use --json payload and ignore cmd argument when both provided."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        json_payload = '{"cmd": "semantic", "action": "search"}'
        expected_response = {"status": "ok", "results": []}

        mock_query_func = MagicMock(return_value=expected_response)

        with patch("tldr.daemon.query_daemon", mock_query_func):
            # Simulate CLI call with both cmd and --json
            # --json should take precedence
            test_args = [
                "daemon",
                "query",
                "ping",  # This should be ignored
                "--json",
                json_payload,  # This should be used
                "--project",
                str(project_path),
            ]

            with patch("sys.argv", ["tldr"] + test_args):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify query_daemon was called with JSON payload, not {"cmd": "ping"}
            mock_query_func.assert_called_once()
            called_with_project, called_with_command = mock_query_func.call_args[0]
            assert called_with_project == project_path
            assert called_with_command == json.loads(json_payload)
            assert called_with_command != {"cmd": "ping"}

    def test_query_complex_json_payload(self):
        """Should handle complex nested JSON payloads."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        complex_json = json.dumps(
            {
                "cmd": "semantic",
                "action": "search",
                "query": "validate tokens",
                "filters": {"language": "python", "min_tokens": 100},
                "limit": 10,
            }
        )
        expected_response = {
            "status": "ok",
            "results": [
                {
                    "file": "auth.py",
                    "function": "verify_token",
                    "score": 0.95,
                }
            ],
        }

        mock_query_func = MagicMock(return_value=expected_response)

        with patch("tldr.daemon.query_daemon", mock_query_func):
            test_args = [
                "daemon",
                "query",
                "--json",
                complex_json,
                "--project",
                str(project_path),
            ]

            with patch("sys.argv", ["tldr"] + test_args):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify complex payload was parsed correctly
            mock_query_func.assert_called_once()
            called_with_project, called_with_command = mock_query_func.call_args[0]
            assert called_with_project == project_path
            assert called_with_command == json.loads(complex_json)
            assert called_with_command["filters"]["language"] == "python"
            assert called_with_command["limit"] == 10

    def test_query_json_with_special_characters(self):
        """Should handle JSON with special characters and unicode."""
        from tldr.cli import main

        project_path = Path("/fake/project").resolve()
        json_with_special = json.dumps(
            {"cmd": "search", "query": "función 中文 🚀", "regex": "^test_.*"}
        )
        expected_response = {"status": "ok", "results": []}

        mock_query_func = MagicMock(return_value=expected_response)

        with patch("tldr.daemon.query_daemon", mock_query_func):
            test_args = [
                "daemon",
                "query",
                "--json",
                json_with_special,
                "--project",
                str(project_path),
            ]

            with patch("sys.argv", ["tldr"] + test_args):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify special characters preserved
            mock_query_func.assert_called_once()
            _, called_with_command = mock_query_func.call_args[0]
            assert called_with_command["query"] == "función 中文 🚀"
            assert called_with_command["regex"] == "^test_.*"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDaemonSocketTruncationFix:
    """Tests for socket receive truncation bug fix.

    Bug: query_daemon() used client.recv(65536) which only returns data
    currently available in the socket buffer, not up to the buffer size.
    Large responses were truncated at ~8192 bytes on macOS.

    Fix: _recv_all() loops until newline delimiter is received.
    """

    def test_recv_all_handles_large_response_in_chunks(self):
        """Should receive complete response even when delivered in multiple chunks."""
        from tldr.daemon.startup import _recv_all
        from unittest.mock import MagicMock
        import json

        # Create a mock socket that returns data in 4096-byte chunks
        mock_socket = MagicMock()

        # Simulate a 20KB response (larger than typical socket buffer)
        large_response = {"status": "ok", "data": "x" * 20000}
        response_bytes = (json.dumps(large_response) + "\n").encode()

        # Split into chunks smaller than the response to simulate socket behavior
        chunk_size = 4096
        chunks = [response_bytes[i:i+chunk_size] for i in range(0, len(response_bytes), chunk_size)]

        # Mock recv to return chunks sequentially, then empty bytes
        call_count = [0]
        def mock_recv(size):
            if call_count[0] < len(chunks):
                chunk = chunks[call_count[0]]
                call_count[0] += 1
                return chunk
            return b""  # EOF

        mock_socket.recv.side_effect = mock_recv

        # Call _recv_all
        result = _recv_all(mock_socket)

        # Verify we got the complete response
        expected = response_bytes.rstrip(b"\n")
        assert result == expected, f"Expected {len(expected)} bytes, got {len(result)} bytes"
        assert len(result) == 20028  # JSON + "x" * 20000 + "\n" minus stripped newline

    def test_recv_all_handles_single_chunk(self):
        """Should handle responses that fit in a single chunk."""
        from tldr.daemon.startup import _recv_all

        mock_socket = MagicMock()
        test_data = b'{"status": "ok"}\n'
        mock_socket.recv.return_value = test_data

        result = _recv_all(mock_socket)

        assert result == b'{"status": "ok"}'
        assert mock_socket.recv.call_count == 1

    def test_recv_all_strips_newline_delimiter(self):
        """Should strip the newline delimiter from the response."""
        from tldr.daemon.startup import _recv_all

        mock_socket = MagicMock()
        test_data = b'{"status": "ok"}\n'
        mock_socket.recv.return_value = test_data

        result = _recv_all(mock_socket)

        assert result == b'{"status": "ok"}'
        assert not result.endswith(b"\n")

    def test_query_daemon_with_large_response(self):
        """Should handle large daemon responses without truncation."""
        from tldr.daemon.startup import query_daemon
        from unittest.mock import patch, MagicMock

        # Create a 15KB response (larger than 8192 byte bug threshold)
        large_response = {
            "status": "ok",
            "results": [{"file": f"file_{i}.py", "data": "x" * 1000} for i in range(15)]
        }
        response_json = json.dumps(large_response) + "\n"
        response_bytes = response_json.encode()

        # Mock socket that returns data in chunks
        mock_client = MagicMock()
        chunk_size = 4096
        chunks = [response_bytes[i:i+chunk_size] for i in range(0, len(response_bytes), chunk_size)]
        call_count = [0]
        def mock_recv(size):
            if call_count[0] < len(chunks):
                chunk = chunks[call_count[0]]
                call_count[0] += 1
                return chunk
            return b""
        mock_client.recv.side_effect = mock_recv

        with patch("tldr.daemon.startup._create_client_socket", return_value=mock_client):
            result = query_daemon("/fake/project", {"cmd": "test"})

        # Verify complete response was received
        assert result == large_response
        assert len(result["results"]) == 15
