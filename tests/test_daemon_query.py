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

            with patch("sys.argv", ["tldr"] + test_args):
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

    def test_query_json_precedence_over_cmd(self, capsys):
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

    def test_query_complex_json_payload(self, capsys):
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

    def test_query_json_with_special_characters(self, capsys):
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
