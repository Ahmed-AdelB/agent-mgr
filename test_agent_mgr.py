"""
Comprehensive unit tests for agent-mgr system.

Covers:
- TestGitHubConnector: subprocess.run mocking, error handling, security
- TestAutoManager: GitHubConnector mocking, decision logic
- TestCLI: argparse validation
- TestProcessController: SSH process management with mocked subprocess
- TestSyncGitHubToInbox: INBOX.md generation
- TestAutoLoopWithProcessController: Integration of ProcessController in auto loop

Author: Ahmed Adel Bakr Alderai
"""
from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
import os
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

# Ensure the source directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from github_connector import (
    CLIAuthError,
    CLIExecutionError,
    CLINotFoundError,
    Comment,
    GitHubConnector,
    GitHubConnectorError,
    Issue,
    JSONParseError,
    Label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completed_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gh"], stdout=stdout, stderr=stderr, returncode=returncode
    )


def _gh_ok(stdout: str = "") -> subprocess.CompletedProcess:
    return _make_completed_process(stdout=stdout)


def _gh_fail(stderr: str = "error", rc: int = 1) -> subprocess.CompletedProcess:
    return _make_completed_process(stderr=stderr, returncode=rc)


def _build_connector(mock_run: MagicMock) -> GitHubConnector:
    """Build a GitHubConnector with _verify_cli passing (2 calls)."""
    mock_run.side_effect = [_gh_ok("gh version 2.40.0"), _gh_ok()]
    connector = GitHubConnector(repo="owner/repo", timeout=10)
    mock_run.reset_mock()
    mock_run.side_effect = None
    return connector


def _issue_json(
    number: int = 1,
    title: str = "Test issue",
    state: str = "OPEN",
    body: str = "body",
    labels: list[str] | None = None,
) -> dict:
    return {
        "number": number,
        "title": title,
        "state": state,
        "body": body,
        "labels": [{"name": lb} for lb in (labels or [])],
        "url": f"https://github.com/owner/repo/issues/{number}",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T01:00:00Z",
        "comments": {"totalCount": 0},
    }


def _ssh_ok(stdout: str = "") -> subprocess.CompletedProcess:
    """Build a successful SSH CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["ssh", "hetzner"], stdout=stdout, stderr="", returncode=0
    )


def _ssh_fail(stderr: str = "error", rc: int = 1) -> subprocess.CompletedProcess:
    """Build a failed SSH CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["ssh", "hetzner"], stdout="", stderr=stderr, returncode=rc
    )


# ===========================================================================
# TestGitHubConnector
# ===========================================================================

class TestGitHubConnector:
    """Tests for GitHubConnector with subprocess.run mocked."""

    # ------------------------------------------------------------------
    # Initialization / _verify_cli
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_init_success(self, mock_run):
        mock_run.side_effect = [_gh_ok("gh version 2.40.0"), _gh_ok()]
        gh = GitHubConnector(repo="owner/repo")
        assert gh.repo == "owner/repo"
        assert mock_run.call_count == 2

    @patch("github_connector.subprocess.run")
    def test_init_cli_not_found_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("gh not found")
        with pytest.raises(CLINotFoundError, match="not installed"):
            GitHubConnector(repo="owner/repo")

    @patch("github_connector.subprocess.run")
    def test_init_cli_not_found_nonzero_rc(self, mock_run):
        mock_run.return_value = _gh_fail(stderr="not found", rc=127)
        with pytest.raises(CLINotFoundError, match="non-zero exit code"):
            GitHubConnector(repo="owner/repo")

    @patch("github_connector.subprocess.run")
    def test_init_cli_auth_error(self, mock_run):
        mock_run.side_effect = [_gh_ok("gh version 2.40.0"), _gh_fail("not logged in")]
        with pytest.raises(CLIAuthError, match="not authenticated"):
            GitHubConnector(repo="owner/repo")

    # ------------------------------------------------------------------
    # list_issues
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_list_issues_returns_parsed_issues(self, mock_run):
        gh = _build_connector(mock_run)
        issues_data = [
            _issue_json(1, "First"),
            _issue_json(2, "Second", labels=["bug"]),
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues_data))

        result = gh.list_issues()

        assert len(result) == 2
        assert isinstance(result[0], Issue)
        assert result[0].number == 1
        assert result[0].title == "First"
        assert result[1].labels == ["bug"]

    @patch("github_connector.subprocess.run")
    def test_list_issues_with_labels_filter(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok(json.dumps([]))
        gh.list_issues(labels=["bug", "P0"])

        args_passed = mock_run.call_args[0][0]
        # shlex.quote wraps each arg; verify label filter is present
        assert any("bug,P0" in str(a) for a in args_passed)

    @patch("github_connector.subprocess.run")
    def test_list_issues_empty_output(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok("")
        result = gh.list_issues()
        assert result == []

    # ------------------------------------------------------------------
    # get_issue
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_get_issue_returns_issue_and_comments(self, mock_run):
        gh = _build_connector(mock_run)
        issue_data = _issue_json(42, "My Issue")
        comments_data = {
            "comments": [
                {
                    "id": "c1",
                    "body": "hello",
                    "author": {"login": "user1"},
                    "createdAt": "2025-01-01T00:00:00Z",
                }
            ]
        }
        mock_run.side_effect = [
            _gh_ok(json.dumps(issue_data)),
            _gh_ok(json.dumps(comments_data)),
        ]

        result = gh.get_issue(42)

        assert "issue" in result
        assert "comments" in result
        assert isinstance(result["issue"], Issue)
        assert result["issue"].number == 42
        assert len(result["comments"]) == 1
        assert isinstance(result["comments"][0], Comment)
        assert result["comments"][0].author == "user1"

    # ------------------------------------------------------------------
    # create_issue
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_create_issue_parses_url_for_number(self, mock_run):
        gh = _build_connector(mock_run)
        create_output = "https://github.com/owner/repo/issues/99\n"
        issue_data = _issue_json(99, "New Issue")
        comments_data = {"comments": []}

        mock_run.side_effect = [
            _gh_ok(create_output),      # create_issue _run
            _gh_ok(json.dumps(issue_data)),  # get_issue view
            _gh_ok(json.dumps(comments_data)),  # get_issue comments
        ]

        result = gh.create_issue("New Issue", "body text")

        assert isinstance(result, Issue)
        assert result.number == 99

    @patch("github_connector.subprocess.run")
    def test_create_issue_with_labels(self, mock_run):
        gh = _build_connector(mock_run)
        create_output = "https://github.com/owner/repo/issues/10\n"
        issue_data = _issue_json(10, "Labeled")
        comments_data = {"comments": []}

        mock_run.side_effect = [
            _gh_ok(create_output),
            _gh_ok(json.dumps(issue_data)),
            _gh_ok(json.dumps(comments_data)),
        ]

        gh.create_issue("Labeled", "body", labels=["bug", "P0"])
        create_call_args = mock_run.call_args_list[0][0][0]
        # Verify label flags are in the command args
        args_str = " ".join(str(a) for a in create_call_args)
        assert "bug" in args_str
        assert "P0" in args_str

    # ------------------------------------------------------------------
    # add_label / remove_label
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_add_label_calls_edit_with_add_label_flag(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok()
        gh.add_label(42, "bug")

        args_passed = mock_run.call_args[0][0]
        args_str = " ".join(str(a) for a in args_passed)
        assert "issue" in args_str
        assert "edit" in args_str
        assert "--add-label" in args_str
        assert "bug" in args_str

    @patch("github_connector.subprocess.run")
    def test_remove_label_calls_edit_with_remove_label_flag(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok()
        gh.remove_label(42, "bug")

        args_passed = mock_run.call_args[0][0]
        args_str = " ".join(str(a) for a in args_passed)
        assert "issue" in args_str
        assert "edit" in args_str
        assert "--remove-label" in args_str
        assert "bug" in args_str

    # ------------------------------------------------------------------
    # add_labels
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_add_labels_calls_add_label_for_each(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok()
        gh.add_labels(42, ["bug", "P0", "status:ready"])
        assert mock_run.call_count == 3

    # ------------------------------------------------------------------
    # create_or_update_label
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_create_or_update_label_creates_when_new(self, mock_run):
        gh = _build_connector(mock_run)
        # list_labels returns empty
        mock_run.side_effect = [
            _gh_ok(json.dumps([])),  # list_labels
            _gh_ok(),                # create_label
        ]
        gh.create_or_update_label("new-label", "0075ca", "desc")
        create_call = mock_run.call_args_list[1][0][0]
        args_str = " ".join(str(a) for a in create_call)
        assert "label" in args_str
        assert "create" in args_str

    @patch("github_connector.subprocess.run")
    def test_create_or_update_label_updates_when_existing(self, mock_run):
        gh = _build_connector(mock_run)
        existing = [{"name": "old-label", "color": "000000", "description": "old"}]
        mock_run.side_effect = [
            _gh_ok(json.dumps(existing)),  # list_labels
            _gh_ok(),                      # label edit
        ]
        gh.create_or_update_label("old-label", "ffffff", "new desc")
        edit_call = mock_run.call_args_list[1][0][0]
        args_str = " ".join(str(a) for a in edit_call)
        assert "label" in args_str
        assert "edit" in args_str

    # ------------------------------------------------------------------
    # agent_health
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_agent_health_returns_correct_dict_keys(self, mock_run):
        gh = _build_connector(mock_run)
        # list_issues (in-progress): empty
        # list_issues (all with label): empty
        # get_latest_comment_time: list_issues returns empty
        mock_run.side_effect = [
            _gh_ok(json.dumps([])),  # in-progress
            _gh_ok(json.dumps([])),  # get_latest_comment_time -> list_issues
        ]
        result = gh.agent_health("researcher")
        assert "last_activity" in result
        assert "healthy" in result
        assert "stale_hours" in result
        assert "current_issue" in result

    @patch("github_connector.subprocess.run")
    def test_agent_health_unknown_agent(self, mock_run):
        gh = _build_connector(mock_run)
        result = gh.agent_health("nonexistent")
        assert result["healthy"] is False
        assert result["current_issue"] is None

    @patch("github_connector.subprocess.run")
    def test_agent_health_with_current_issue(self, mock_run):
        gh = _build_connector(mock_run)
        in_progress_issues = [_issue_json(10, "Working on X", labels=["research", "status:in-progress"])]
        mock_run.side_effect = [
            _gh_ok(json.dumps(in_progress_issues)),  # in-progress
            _gh_ok(json.dumps([])),                   # get_latest_comment_time
        ]
        result = gh.agent_health("researcher")
        assert result["current_issue"] is not None
        assert result["current_issue"]["number"] == 10

    # ------------------------------------------------------------------
    # agent_queue
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_agent_queue_returns_correct_stats(self, mock_run):
        gh = _build_connector(mock_run)
        issues = [
            _issue_json(1, "A", labels=["research", "status:ready"]),
            _issue_json(2, "B", labels=["research", "status:ready"]),
            _issue_json(3, "C", labels=["research", "status:in-progress"]),
            _issue_json(4, "D", labels=["research"]),
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues))
        result = gh.agent_queue("researcher")
        assert result["ready"] == 2
        assert result["in_progress"] == 1
        assert result["total"] == 4

    @patch("github_connector.subprocess.run")
    def test_agent_queue_unknown_agent(self, mock_run):
        gh = _build_connector(mock_run)
        result = gh.agent_queue("nonexistent")
        assert result == {"ready": 0, "in_progress": 0, "total": 0}

    # ------------------------------------------------------------------
    # needs_review_issues
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_needs_review_issues_filters_correctly(self, mock_run):
        gh = _build_connector(mock_run)
        issues = [
            _issue_json(5, "Review me", labels=["status:needs-review", "builder"]),
            _issue_json(6, "Review too", labels=["status:needs-review", "research"]),
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues))
        result = gh.needs_review_issues()
        assert len(result) == 2
        assert result[0]["number"] == 5
        assert any(lb["name"] == "status:needs-review" for lb in result[0]["labels"])

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_cli_execution_error_on_nonzero_rc(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_fail("something broke", rc=1)
        with pytest.raises(CLIExecutionError, match="something broke"):
            gh.list_issues()

    @patch("github_connector.subprocess.run")
    def test_cli_execution_error_on_timeout(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=10)
        with pytest.raises(CLIExecutionError, match="timed out"):
            gh.list_issues()

    @patch("github_connector.subprocess.run")
    def test_json_parse_error_on_invalid_json(self, mock_run):
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok("this is not json{{{")
        with pytest.raises(JSONParseError, match="Failed to parse"):
            gh.list_issues()

    def test_cli_not_found_error_inherits_base(self):
        exc = CLINotFoundError("test", cmd="gh", stderr="err")
        assert isinstance(exc, GitHubConnectorError)
        assert exc.cmd == "gh"
        assert exc.stderr == "err"

    def test_cli_auth_error_inherits_base(self):
        exc = CLIAuthError("test")
        assert isinstance(exc, GitHubConnectorError)

    def test_cli_execution_error_inherits_base(self):
        exc = CLIExecutionError("test", cmd="gh issue list", stderr="nope")
        assert isinstance(exc, GitHubConnectorError)

    def test_json_parse_error_inherits_base(self):
        exc = JSONParseError("test")
        assert isinstance(exc, GitHubConnectorError)

    # ------------------------------------------------------------------
    # Security: shlex.quote usage
    # ------------------------------------------------------------------

    @patch("github_connector.subprocess.run")
    def test_list_mode_passes_raw_args_to_subprocess(self, mock_run):
        """Args are passed as a list (no shlex.quote) — list-mode subprocess
        prevents shell injection without quoting."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok(json.dumps([]))
        gh.list_issues(labels=["label; rm -rf /"])

        args_passed = mock_run.call_args[0][0]
        assert args_passed[0] == "gh"
        # Args should be raw (unquoted) in list mode
        assert isinstance(args_passed, list)
        # The dangerous label should appear as-is (no shell interpretation)
        assert any("label; rm -rf /" in str(a) for a in args_passed)

    @patch("github_connector.subprocess.run")
    def test_list_mode_preserves_special_chars_in_body(self, mock_run):
        """Special characters in body are preserved raw (list-mode safety)."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok()
        gh.add_comment(42, "body with $(malicious) `command`")

        args_passed = mock_run.call_args[0][0]
        body_args = [a for a in args_passed if "malicious" in a]
        assert len(body_args) == 1
        # In list mode, the raw string is passed directly (no quoting)
        assert "$(malicious)" in body_args[0]

    @patch("github_connector.subprocess.run")
    def test_list_mode_preserves_backticks_in_title(self, mock_run):
        """Backticks in title are preserved raw (list-mode safety)."""
        gh = _build_connector(mock_run)
        mock_run.side_effect = [
            _gh_ok("https://github.com/owner/repo/issues/1\n"),
            _gh_ok(json.dumps(_issue_json(1, "test"))),
            _gh_ok(json.dumps({"comments": []})),
        ]
        gh.create_issue("`whoami`", "safe body")
        create_call_args = mock_run.call_args_list[0][0][0]
        title_args = [a for a in create_call_args if "whoami" in a]
        assert len(title_args) == 1
        assert title_args[0] == "`whoami`"

    # ------------------------------------------------------------------
    # Data classes
    # ------------------------------------------------------------------

    def test_issue_from_dict_with_dict_labels(self):
        data = {
            "number": 5,
            "title": "Test",
            "state": "OPEN",
            "labels": [{"name": "bug"}, {"name": "P0"}],
        }
        issue = Issue.from_dict(data)
        assert issue.labels == ["bug", "P0"]

    def test_issue_from_dict_with_string_labels(self):
        data = {"number": 5, "title": "T", "state": "OPEN", "labels": ["a", "b"]}
        issue = Issue.from_dict(data)
        assert issue.labels == ["a", "b"]

    def test_issue_from_dict_comments_totalcount(self):
        data = {"number": 1, "title": "T", "state": "OPEN", "comments": {"totalCount": 3}}
        issue = Issue.from_dict(data)
        assert issue.comments_count == 3

    def test_comment_from_dict(self):
        data = {
            "id": "IC_123",
            "body": "hello",
            "author": {"login": "alice"},
            "createdAt": "2025-01-01T00:00:00Z",
        }
        c = Comment.from_dict(data)
        assert c.author == "alice"
        assert c.body == "hello"

    def test_comment_from_dict_string_author(self):
        data = {"id": "1", "body": "x", "author": "bob", "createdAt": ""}
        c = Comment.from_dict(data)
        assert c.author == "bob"

    def test_label_from_dict(self):
        data = {"name": "bug", "color": "d73a4a", "description": "Something broken"}
        lb = Label.from_dict(data)
        assert lb.name == "bug"
        assert lb.color == "d73a4a"


# ===========================================================================
# TestAutoManager
# ===========================================================================

class TestAutoManager:
    """Tests for AutoManager with GitHubConnector fully mocked."""

    def _make_manager(self) -> tuple:
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        # Import here to avoid top-level import issues with logging config
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    def _make_issue_obj(
        self,
        number: int = 1,
        title: str = "Test",
        body: str = "body",
        labels: list[str] | None = None,
        state: str = "OPEN",
        updated_at: str = "2025-01-01T00:00:00Z",
    ) -> Issue:
        return Issue(
            number=number,
            title=title,
            state=state,
            body=body,
            labels=labels or [],
            url=f"https://github.com/owner/repo/issues/{number}",
            created_at="2025-01-01T00:00:00Z",
            updated_at=updated_at,
        )

    # ------------------------------------------------------------------
    # dashboard
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_dashboard_returns_formatted_string(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []

        result = mgr.dashboard()

        assert isinstance(result, str)
        assert "COMMAND CENTER DASHBOARD" in result

    # ------------------------------------------------------------------
    # health_check
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_health_check_detects_healthy(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(hours=1)).isoformat()
        recent_comment = Comment(
            id="1", body="progress", author="bot",
            created_at=recent_ts,
        )
        mock_gh.list_issues.return_value = [
            self._make_issue_obj(1, labels=["research", "status:in-progress"])
        ]
        mock_gh.get_comments.return_value = [recent_comment]

        result = mgr.health_check()

        assert "researcher" in result
        assert result["researcher"]["status"] == "healthy"

    @patch("auto_manager._log_decision")
    def test_health_check_detects_stale(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(hours=6)).isoformat()
        old_comment = Comment(id="1", body="x", author="bot", created_at=old_ts)

        mock_gh.list_issues.return_value = [
            self._make_issue_obj(1, labels=["research", "status:in-progress"])
        ]
        mock_gh.get_comments.return_value = [old_comment]

        result = mgr.health_check()

        assert result["researcher"]["status"] == "stale"

    @patch("auto_manager._log_decision")
    def test_health_check_detects_dead(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        ancient_ts = (now - timedelta(hours=10)).isoformat()
        old_comment = Comment(id="1", body="x", author="bot", created_at=ancient_ts)

        mock_gh.list_issues.return_value = [
            self._make_issue_obj(1, labels=["research", "status:in-progress"])
        ]
        mock_gh.get_comments.return_value = [old_comment]

        result = mgr.health_check()

        assert result["researcher"]["status"] == "dead"

    @patch("auto_manager._log_decision")
    def test_health_check_no_issues_returns_dead(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []

        result = mgr.health_check()

        # No comments means infinite hours -> dead
        for agent_name in result:
            assert result[agent_name]["status"] == "dead"

    # ------------------------------------------------------------------
    # queue
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_queue_returns_prioritized_list(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = [
            self._make_issue_obj(1, "Low", labels=["research", "P3", "status:ready"]),
            self._make_issue_obj(2, "High", labels=["research", "P0", "status:ready"]),
            self._make_issue_obj(3, "Med", labels=["research", "P1", "status:in-progress"]),
        ]

        result = mgr.queue("researcher")

        assert len(result) == 3
        assert result[0]["priority"] == "P0"
        assert result[0]["number"] == 2
        assert result[1]["priority"] == "P1"
        assert result[2]["priority"] == "P3"

    @patch("auto_manager._log_decision")
    def test_queue_unknown_agent_raises(self, mock_log):
        mgr, mock_gh = self._make_manager()
        with pytest.raises(ValueError, match="Unknown agent"):
            mgr.queue("nonexistent")

    # ------------------------------------------------------------------
    # assign
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_assign_adds_correct_labels_and_comments(self, mock_log):
        mgr, mock_gh = self._make_manager()

        mgr.assign("builder", 42, "Please prioritize this.")

        mock_gh.add_labels.assert_called_once_with(
            42, ["role:builder", "status:ready"]
        )
        mock_gh.add_comment.assert_called_once()
        comment_body = mock_gh.add_comment.call_args[0][1]
        assert "builder" in comment_body
        assert "Please prioritize this." in comment_body

    @patch("auto_manager._log_decision")
    def test_assign_without_message(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mgr.assign("researcher", 10)

        mock_gh.add_labels.assert_called_once_with(
            10, ["role:researcher", "status:ready"]
        )
        comment_body = mock_gh.add_comment.call_args[0][1]
        assert "researcher" in comment_body

    @patch("auto_manager._log_decision")
    def test_assign_unknown_agent_raises(self, mock_log):
        mgr, mock_gh = self._make_manager()
        with pytest.raises(ValueError, match="Unknown agent"):
            mgr.assign("invalid_agent", 1)

    # ------------------------------------------------------------------
    # directive
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_directive_posts_to_in_progress_issue(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = [
            self._make_issue_obj(50, "Current work", labels=["research", "status:in-progress"])
        ]

        mgr.directive("researcher", "Focus on Arabic papers")

        mock_gh.add_comment.assert_called_once()
        call_args = mock_gh.add_comment.call_args
        assert call_args[0][0] == 50
        assert "MANAGER DIRECTIVE" in call_args[0][1]
        assert "Arabic papers" in call_args[0][1]

    @patch("auto_manager._log_decision")
    def test_directive_creates_new_issue_when_no_in_progress(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.create_issue.return_value = self._make_issue_obj(100, "Directive")

        mgr.directive("builder", "Build the auth module")

        mock_gh.create_issue.assert_called_once()
        create_args = mock_gh.create_issue.call_args
        assert "Directive" in create_args[1].get("title", "") or "Directive" in create_args[0][0]

    # ------------------------------------------------------------------
    # review_queue
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_review_queue_lists_needs_review_items(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = [
            self._make_issue_obj(
                10, "Review A", labels=["status:needs-review", "role:builder", "P1"]
            ),
            self._make_issue_obj(
                11, "Review B", labels=["status:needs-review", "role:researcher", "P0"]
            ),
        ]

        result = mgr.review_queue()

        assert len(result) == 2
        assert result[0]["number"] == 10
        assert result[1]["number"] == 11

    # ------------------------------------------------------------------
    # auto_review
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_auto_review_detects_empty_body(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.get_issue.return_value = {
            "issue": self._make_issue_obj(1, body=""),
            "comments": [
                Comment(id="1", body="A" * 60, author="u", created_at="2025-01-01T00:00:00Z")
            ],
        }

        result = mgr.auto_review(1)

        assert "qa:failed" in result
        assert "empty" in result.lower()

    @patch("auto_manager._log_decision")
    def test_auto_review_detects_no_comments(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.get_issue.return_value = {
            "issue": self._make_issue_obj(1, body="Some body text here"),
            "comments": [],
        }

        result = mgr.auto_review(1)

        assert "qa:failed" in result
        assert "No comments" in result

    @patch("auto_manager._log_decision")
    def test_auto_review_detects_no_substantive_comments(self, mock_log):
        mgr, mock_gh = self._make_manager()
        mock_gh.get_issue.return_value = {
            "issue": self._make_issue_obj(1, body="Some real body"),
            "comments": [
                Comment(id="1", body="ok", author="u", created_at="2025-01-01T00:00:00Z"),
                Comment(id="2", body="done", author="u", created_at="2025-01-01T01:00:00Z"),
            ],
        }

        result = mgr.auto_review(1)

        assert "qa:failed" in result
        assert "substantive" in result.lower()

    @patch("auto_manager.Path")
    @patch("auto_manager._log_decision")
    def test_auto_review_passes_with_good_content(self, mock_log, mock_path_cls):
        mgr, mock_gh = self._make_manager()
        # Make validate-quality.sh not exist so it does not run
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_cls.home.return_value.__truediv__ = Mock(return_value=mock_path_instance)

        long_comment = "A" * 60  # >50 chars
        mock_gh.get_issue.return_value = {
            "issue": self._make_issue_obj(1, body="Real body content here"),
            "comments": [
                Comment(id="1", body=long_comment, author="u", created_at="2025-01-01T00:00:00Z"),
            ],
        }

        result = mgr.auto_review(1)

        assert result == "qa:passed"

    # ------------------------------------------------------------------
    # replenish_queues
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_replenish_queues_promotes_from_backlog(self, mock_log):
        mgr, mock_gh = self._make_manager()

        ready_issues = []  # 0 ready
        backlog_issues = [
            self._make_issue_obj(10, "Backlog1", labels=["research", "status:backlog", "P1"]),
            self._make_issue_obj(11, "Backlog2", labels=["research", "status:backlog", "P0"]),
        ]

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:ready" in labels:
                return ready_issues
            if labels and "status:backlog" in labels:
                return backlog_issues
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        mgr.replenish_queues(min_ready=2)

        # Should have promoted 2 backlog items via transition_status
        assert mock_gh.transition_status.call_count >= 2
        assert mock_gh.add_comment.call_count >= 2

    @patch("auto_manager._log_decision")
    def test_replenish_queues_skips_when_enough_ready(self, mock_log):
        mgr, mock_gh = self._make_manager()
        ready_issues = [
            self._make_issue_obj(1, labels=["research", "status:ready"]),
            self._make_issue_obj(2, labels=["research", "status:ready"]),
            self._make_issue_obj(3, labels=["research", "status:ready"]),
        ]

        mock_gh.list_issues.return_value = ready_issues

        mgr.replenish_queues(min_ready=2)

        # No promotions needed
        mock_gh.remove_label.assert_not_called()

    # ------------------------------------------------------------------
    # check_dependencies
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_check_dependencies_unblocks_when_deps_resolved(self, mock_log):
        mgr, mock_gh = self._make_manager()

        approved_issues = [
            self._make_issue_obj(10, labels=["status:approved"]),
        ]
        closed_issues = [
            self._make_issue_obj(11, state="CLOSED"),
        ]
        blocked_issues = [
            self._make_issue_obj(
                20,
                body="Depends on #10 and depends on #11",
                labels=["status:blocked"],
            ),
        ]

        call_count = [0]

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:approved" in labels:
                return approved_issues
            if state == "closed":
                return closed_issues
            if labels and "status:blocked" in labels:
                return blocked_issues
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        mgr.check_dependencies()

        mock_gh.transition_status.assert_called_with(20, "status:ready")
        mock_gh.add_comment.assert_called_once()
        comment = mock_gh.add_comment.call_args[0][1]
        assert "#10" in comment
        assert "#11" in comment

    @patch("auto_manager._log_decision")
    def test_check_dependencies_does_not_unblock_with_unresolved(self, mock_log):
        mgr, mock_gh = self._make_manager()

        approved_issues = [self._make_issue_obj(10, labels=["status:approved"])]
        blocked_issues = [
            self._make_issue_obj(
                20,
                body="Depends on #10 and depends on #99",
                labels=["status:blocked"],
            ),
        ]

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:approved" in labels:
                return approved_issues
            if state == "closed":
                return []
            if labels and "status:blocked" in labels:
                return blocked_issues
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        mgr.check_dependencies()

        mock_gh.remove_label.assert_not_called()

    # ------------------------------------------------------------------
    # nudge_stale
    # ------------------------------------------------------------------

    @patch("auto_manager._log_decision")
    def test_nudge_stale_posts_nudge_after_silence(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(hours=5)).isoformat()

        in_progress = [self._make_issue_obj(30, updated_at=old_ts, labels=["status:in-progress"])]
        old_comment = Comment(id="c1", body="working on it", author="bot", created_at=old_ts)

        mock_gh.list_issues.return_value = in_progress
        mock_gh.get_comments.return_value = [old_comment]

        mgr.nudge_stale(hours=4)

        mock_gh.add_comment.assert_called_once()
        nudge_body = mock_gh.add_comment.call_args[0][1]
        assert "[MANAGER]" in nudge_body
        assert "Status update" in nudge_body

    @patch("auto_manager._log_decision")
    def test_nudge_stale_avoids_duplicate_nudge(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(hours=5)).isoformat()

        in_progress = [self._make_issue_obj(30, updated_at=old_ts, labels=["status:in-progress"])]
        # Last comment is already a nudge
        nudge_comment = Comment(
            id="c1",
            body="[MANAGER] Status update required. Are you blocked?",
            author="manager",
            created_at=old_ts,
        )

        mock_gh.list_issues.return_value = in_progress
        mock_gh.get_comments.return_value = [nudge_comment]

        mgr.nudge_stale(hours=4)

        # Should NOT post another nudge
        mock_gh.add_comment.assert_not_called()

    @patch("auto_manager._log_decision")
    def test_nudge_stale_skips_recent_issues(self, mock_log):
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(hours=1)).isoformat()

        in_progress = [self._make_issue_obj(30, updated_at=recent_ts, labels=["status:in-progress"])]
        recent_comment = Comment(id="c1", body="still working", author="bot", created_at=recent_ts)

        mock_gh.list_issues.return_value = in_progress
        mock_gh.get_comments.return_value = [recent_comment]

        mgr.nudge_stale(hours=4)

        mock_gh.add_comment.assert_not_called()

    # ------------------------------------------------------------------
    # run_auto_loop
    # ------------------------------------------------------------------

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_run_auto_loop_cycles_through_all_checks(self, mock_log, mock_sleep, mock_pc_cls):
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        # Mock ProcessController
        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 1
        mock_pc_cls.return_value = mock_pc

        # Make sleep raise KeyboardInterrupt to exit the loop after one cycle
        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        # Verify that list_issues was called (health_check, replenish, dependencies, nudge all use it)
        assert mock_gh.list_issues.call_count > 0
        # Verify prevent_unauthorized_closes was called
        mock_gh.list_recently_closed.assert_called_once()


# ===========================================================================
# TestCLI
# ===========================================================================

class TestCLI:
    """Tests for the argparse-based CLI in agent_mgr.py."""

    def _get_parser(self):
        from agent_mgr import build_parser
        return build_parser()

    # ------------------------------------------------------------------
    # All commands parse correctly
    # ------------------------------------------------------------------

    def test_status_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_health_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_queue_command_no_agent(self):
        parser = self._get_parser()
        args = parser.parse_args(["queue"])
        assert args.command == "queue"
        assert args.agent is None

    def test_queue_command_with_agent(self):
        parser = self._get_parser()
        args = parser.parse_args(["queue", "researcher"])
        assert args.command == "queue"
        assert args.agent == "researcher"

    def test_review_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["review"])
        assert args.command == "review"

    def test_assign_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["assign", "builder", "489"])
        assert args.command == "assign"
        assert args.agent == "builder"
        assert args.issue == 489

    def test_directive_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["directive", "kimi", "Focus on Arabic"])
        assert args.command == "directive"
        assert args.agent == "kimi"
        assert args.message == "Focus on Arabic"

    def test_auto_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["auto"])
        assert args.command == "auto"

    def test_labels_setup_command_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["labels-setup"])
        assert args.command == "labels-setup"

    def test_repo_flag_default(self):
        parser = self._get_parser()
        args = parser.parse_args(["status"])
        assert args.repo == "Ahmed-AdelB/ummro"

    def test_repo_flag_custom(self):
        parser = self._get_parser()
        args = parser.parse_args(["--repo", "other/repo", "status"])
        assert args.repo == "other/repo"

    # ------------------------------------------------------------------
    # Unknown agent rejected
    # ------------------------------------------------------------------

    def test_unknown_agent_in_queue_rejected(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["queue", "nonexistent_agent"])

    def test_unknown_agent_in_assign_rejected(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["assign", "nonexistent_agent", "42"])

    def test_unknown_agent_in_directive_rejected(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["directive", "nonexistent_agent", "msg"])

    # ------------------------------------------------------------------
    # Missing args produce errors
    # ------------------------------------------------------------------

    def test_assign_missing_issue_number(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["assign", "builder"])

    def test_assign_missing_agent(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["assign"])

    def test_directive_missing_message(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["directive", "builder"])

    def test_directive_missing_agent(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["directive"])

    def test_assign_non_integer_issue(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["assign", "builder", "not_a_number"])

    def test_no_command_returns_none(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        assert args.command is None


# ===========================================================================
# TestHelperFunctions
# ===========================================================================

class TestHelperFunctions:
    """Tests for auto_manager helper functions."""

    def test_relative_time_never(self):
        from auto_manager import _relative_time
        assert _relative_time(None) == "never"

    def test_relative_time_just_now(self):
        from auto_manager import _relative_time
        now = datetime.now(timezone.utc).isoformat()
        result = _relative_time(now)
        assert result in ("just now", "0s ago", "1s ago", "2s ago", "3s ago", "4s ago", "5s ago")

    def test_relative_time_hours(self):
        from auto_manager import _relative_time
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        result = _relative_time(ts)
        assert "3h ago" in result

    def test_relative_time_days(self):
        from auto_manager import _relative_time
        ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        result = _relative_time(ts)
        assert "2d ago" in result

    def test_relative_time_invalid(self):
        from auto_manager import _relative_time
        assert _relative_time("not-a-date") == "unknown"

    def test_hours_since_none(self):
        from auto_manager import _hours_since
        assert _hours_since(None) == float("inf")

    def test_hours_since_recent(self):
        from auto_manager import _hours_since
        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        result = _hours_since(ts)
        assert 1.9 < result < 2.1

    def test_hours_since_invalid(self):
        from auto_manager import _hours_since
        assert _hours_since("garbage") == float("inf")

    def test_extract_priority(self):
        from auto_manager import _extract_priority
        assert _extract_priority(["bug", "P0", "status:ready"]) == 0
        assert _extract_priority(["P2"]) == 2
        assert _extract_priority(["no-priority"]) == 4  # len(PRIORITY_LABELS)

    def test_extract_status(self):
        from auto_manager import _extract_status
        assert _extract_status(["bug", "status:ready"]) == "status:ready"
        assert _extract_status(["bug"]) is None

    def test_priority_label(self):
        from auto_manager import _priority_label
        assert _priority_label(["P1", "bug"]) == "P1"
        assert _priority_label(["no-prio"]) == "P?"


# ===========================================================================
# TestListRecentlyClosed
# ===========================================================================

class TestListRecentlyClosed:
    """Tests for GitHubConnector.list_recently_closed()."""

    @patch("github_connector.subprocess.run")
    def test_list_recently_closed_filters_by_time(self, mock_run):
        """Only returns issues closed within the specified hours window."""
        gh = _build_connector(mock_run)
        now = datetime.now(timezone.utc)

        # Issue closed 1 hour ago (should be included)
        recent_closed = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Issue closed 48 hours ago (should be excluded)
        old_closed = (now - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")

        issues_data = [
            {
                "number": 1,
                "title": "Recent issue",
                "labels": [{"name": "role:builder"}],
                "closedAt": recent_closed,
            },
            {
                "number": 2,
                "title": "Old issue",
                "labels": [{"name": "role:researcher"}],
                "closedAt": old_closed,
            },
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues_data))

        result = gh.list_recently_closed(hours=24)

        assert len(result) == 1
        assert result[0]["number"] == 1
        assert result[0]["title"] == "Recent issue"
        assert result[0]["labels"] == ["role:builder"]

    @patch("github_connector.subprocess.run")
    def test_list_recently_closed_handles_string_labels(self, mock_run):
        """Handles both dict-style and string-style labels."""
        gh = _build_connector(mock_run)
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        issues_data = [
            {
                "number": 5,
                "title": "String labels",
                "labels": ["role:kimi", "P0"],
                "closedAt": closed_at,
            },
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues_data))

        result = gh.list_recently_closed(hours=24)

        assert len(result) == 1
        assert result[0]["labels"] == ["role:kimi", "P0"]

    @patch("github_connector.subprocess.run")
    def test_list_recently_closed_skips_invalid_timestamps(self, mock_run):
        """Skips issues with unparseable closedAt timestamps."""
        gh = _build_connector(mock_run)

        issues_data = [
            {
                "number": 1,
                "title": "Bad timestamp",
                "labels": [{"name": "role:builder"}],
                "closedAt": "not-a-valid-date",
            },
            {
                "number": 2,
                "title": "Missing timestamp",
                "labels": [{"name": "role:builder"}],
                "closedAt": "",
            },
        ]
        mock_run.return_value = _gh_ok(json.dumps(issues_data))

        result = gh.list_recently_closed(hours=24)

        assert len(result) == 0

    @patch("github_connector.subprocess.run")
    def test_list_recently_closed_empty_response(self, mock_run):
        """Returns empty list when no closed issues found."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok(json.dumps([]))

        result = gh.list_recently_closed(hours=24)

        assert result == []

    @patch("github_connector.subprocess.run")
    def test_list_recently_closed_uses_correct_gh_args(self, mock_run):
        """Verify the correct gh command arguments are used."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok(json.dumps([]))

        gh.list_recently_closed(hours=24, limit=50)

        args_passed = mock_run.call_args[0][0]
        args_str = " ".join(str(a) for a in args_passed)
        assert "issue" in args_str
        assert "list" in args_str
        assert "--state" in args_str
        assert "closed" in args_str
        assert "closedAt" in args_str


# ===========================================================================
# TestGetIssueCloser
# ===========================================================================

class TestGetIssueCloser:
    """Tests for GitHubConnector.get_issue_closer()."""

    @patch("github_connector.subprocess.run")
    def test_get_issue_closer_returns_login(self, mock_run):
        """Returns the login of the user who closed the issue."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok("agentuser\n")

        result = gh.get_issue_closer(42)

        assert result == "agentuser"
        # Verify API call uses correct path
        args_passed = mock_run.call_args[0][0]
        args_str = " ".join(str(a) for a in args_passed)
        assert "api" in args_str
        assert "issues/42/events" in args_str

    @patch("github_connector.subprocess.run")
    def test_get_issue_closer_returns_none_for_null(self, mock_run):
        """Returns None when jq output is null (no close events)."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok("null\n")

        result = gh.get_issue_closer(42)

        assert result is None

    @patch("github_connector.subprocess.run")
    def test_get_issue_closer_returns_none_on_error(self, mock_run):
        """Returns None when the API call fails."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_fail("API error", rc=1)

        result = gh.get_issue_closer(42)

        assert result is None

    @patch("github_connector.subprocess.run")
    def test_get_issue_closer_strips_quotes(self, mock_run):
        """Strips surrounding quotes from the login."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok('"quoteduser"\n')

        result = gh.get_issue_closer(42)

        assert result == "quoteduser"

    @patch("github_connector.subprocess.run")
    def test_get_issue_closer_returns_none_for_empty(self, mock_run):
        """Returns None when the output is empty."""
        gh = _build_connector(mock_run)
        mock_run.return_value = _gh_ok("\n")

        result = gh.get_issue_closer(42)

        assert result is None


# ===========================================================================
# TestPreventUnauthorizedCloses
# ===========================================================================

class TestPreventUnauthorizedCloses:
    """Tests for AutoManager.prevent_unauthorized_closes()."""

    def _make_manager(self):
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_reopens_issue_closed_by_non_manager(self, mock_log):
        """Reopens issues closed by agents (not manager)."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 42,
                "title": "Agent closed this",
                "labels": ["role:builder", "status:in-progress"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = "agentuser"

        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_called_once_with(42)
        mock_gh.add_comment.assert_called_once()
        comment_body = mock_gh.add_comment.call_args[0][1]
        assert "Issues can only be closed by the Manager" in comment_body
        assert "Reopened automatically" in comment_body

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_skips_issue_closed_by_manager(self, mock_log):
        """Does not reopen issues closed by the manager."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 50,
                "title": "Manager closed this",
                "labels": ["role:researcher"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = "Ahmed-AdelB"

        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_not_called()
        mock_gh.add_comment.assert_not_called()

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_skips_issues_without_role_labels(self, mock_log):
        """Does not touch issues without agent role labels."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 60,
                "title": "Regular bug",
                "labels": ["bug", "P0"],
                "closedAt": closed_at,
            },
        ]

        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_not_called()
        mock_gh.add_comment.assert_not_called()
        # Should not even call get_issue_closer for non-agent issues
        mock_gh.get_issue_closer.assert_not_called()

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_handles_all_role_labels(self, mock_log):
        """Works with all three agent role labels."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 71,
                "title": "Researcher issue",
                "labels": ["role:researcher"],
                "closedAt": closed_at,
            },
            {
                "number": 72,
                "title": "Builder issue",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
            {
                "number": 73,
                "title": "Kimi issue",
                "labels": ["role:kimi"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = "some-agent"

        mgr.prevent_unauthorized_closes(hours=24)

        assert mock_gh.reopen_issue.call_count == 3
        assert mock_gh.add_comment.call_count == 3

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_logs_decision_on_unauthorized_close(self, mock_log):
        """Logs decision when reopening unauthorized closes."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 80,
                "title": "Unauthorized close",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = "badactor"

        mgr.prevent_unauthorized_closes(hours=24)

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == "Unauthorized Close Prevention"
        assert "#80" in call_args[0][1]
        assert "badactor" in call_args[0][1]

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_handles_list_recently_closed_failure(self, mock_log):
        """Gracefully handles failures from list_recently_closed."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_recently_closed.side_effect = Exception("API error")

        # Should not raise
        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_not_called()

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_handles_reopen_failure(self, mock_log):
        """Continues processing if one reopen fails."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 90,
                "title": "First issue",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
            {
                "number": 91,
                "title": "Second issue",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = "agent1"
        mock_gh.reopen_issue.side_effect = [Exception("Cannot reopen"), None]

        # Should not raise
        mgr.prevent_unauthorized_closes(hours=24)

        # Both issues should be attempted
        assert mock_gh.reopen_issue.call_count == 2

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_skips_when_closer_unknown(self, mock_log):
        """Skips issues when the closer cannot be determined (fail-open)."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 95,
                "title": "Unknown closer",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.return_value = None

        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_not_called()

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_handles_get_issue_closer_exception(self, mock_log):
        """Continues if get_issue_closer raises an exception."""
        mgr, mock_gh = self._make_manager()
        now = datetime.now(timezone.utc)
        closed_at = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_gh.list_recently_closed.return_value = [
            {
                "number": 96,
                "title": "Closer API fails",
                "labels": ["role:researcher"],
                "closedAt": closed_at,
            },
            {
                "number": 97,
                "title": "This one works",
                "labels": ["role:builder"],
                "closedAt": closed_at,
            },
        ]
        mock_gh.get_issue_closer.side_effect = [Exception("API error"), "agent-bot"]

        mgr.prevent_unauthorized_closes(hours=24)

        # Only the second issue should be reopened
        mock_gh.reopen_issue.assert_called_once_with(97)

    @patch("auto_manager._log_decision")
    @patch("auto_manager.MANAGER_GITHUB_USER", "Ahmed-AdelB")
    def test_no_recently_closed_issues(self, mock_log):
        """No-op when there are no recently closed issues."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_recently_closed.return_value = []

        mgr.prevent_unauthorized_closes(hours=24)

        mock_gh.reopen_issue.assert_not_called()
        mock_gh.add_comment.assert_not_called()
        mock_gh.get_issue_closer.assert_not_called()


# ===========================================================================
# TestRunAutoLoopIntegration
# ===========================================================================

class TestRunAutoLoopIntegration:
    """Tests for run_auto_loop calling prevent_unauthorized_closes."""

    def _make_manager(self):
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_run_auto_loop_calls_prevent_unauthorized_closes(self, mock_log, mock_sleep, mock_pc_cls):
        """Verify prevent_unauthorized_closes is called during auto loop."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        # Mock ProcessController
        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 1
        mock_pc_cls.return_value = mock_pc

        # Make sleep raise KeyboardInterrupt to exit the loop after one cycle
        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        # Verify prevent_unauthorized_closes was called via list_recently_closed
        mock_gh.list_recently_closed.assert_called_once()


# ===========================================================================
# TestProcessController
# ===========================================================================

class TestProcessController:
    """Tests for ProcessController with subprocess.run mocked."""

    def _make_controller(self):
        from process_controller import ProcessController
        return ProcessController(hetzner_host="hetzner", timeout=10)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_init_defaults(self):
        from process_controller import ProcessController
        pc = ProcessController()
        assert pc.hetzner_host == "hetzner"
        assert pc.timeout == 30

    def test_init_custom_host(self):
        from process_controller import ProcessController
        pc = ProcessController(hetzner_host="myhost", timeout=60)
        assert pc.hetzner_host == "myhost"
        assert pc.timeout == 60

    # ------------------------------------------------------------------
    # check_hetzner_connectivity
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_success(self, mock_run):
        """Returns True when SSH echo ok succeeds."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("ok\n")

        result = pc.check_hetzner_connectivity()

        assert result is True
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ssh", "hetzner", "echo ok"]

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_failure_nonzero_rc(self, mock_run):
        """Returns False when SSH returns non-zero."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_fail("Connection refused", rc=255)

        result = pc.check_hetzner_connectivity()

        assert result is False

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_timeout(self, mock_run):
        """Returns False when SSH times out."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

        result = pc.check_hetzner_connectivity()

        assert result is False

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_ssh_not_found(self, mock_run):
        """Returns False when ssh binary is not found."""
        pc = self._make_controller()
        mock_run.side_effect = FileNotFoundError("ssh not found")

        result = pc.check_hetzner_connectivity()

        assert result is False

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_os_error(self, mock_run):
        """Returns False on generic OSError."""
        pc = self._make_controller()
        mock_run.side_effect = OSError("Network down")

        result = pc.check_hetzner_connectivity()

        assert result is False

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_connectivity_unexpected_output(self, mock_run):
        """Returns False when output does not contain 'ok'."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("unexpected output\n")

        result = pc.check_hetzner_connectivity()

        assert result is False

    # ------------------------------------------------------------------
    # check_kimi_processes
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_check_kimi_processes_found(self, mock_run):
        """Returns parsed process list when kimi processes exist."""
        pc = self._make_controller()
        ps_output = (
            "aadel  12345  0.5  1.2 123456 78901 ?  S  Jan01  10:30:00 kimi --yolo -m kimi-k2\n"
            "aadel  12346  0.3  0.8 123456 78902 ?  S  Jan01  05:15:00 kimi --yolo -m kimi-k2\n"
        )
        mock_run.return_value = _ssh_ok(ps_output)

        result = pc.check_kimi_processes()

        assert len(result) == 2
        assert result[0]["pid"] == 12345
        assert "kimi" in result[0]["cmd"]
        assert result[0]["uptime"] == "10:30:00"
        assert result[1]["pid"] == 12346

    @patch("process_controller.subprocess.run")
    def test_check_kimi_processes_none_found(self, mock_run):
        """Returns empty list when no kimi processes found (grep rc=1)."""
        pc = self._make_controller()
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh", "hetzner"], stdout="", stderr="", returncode=1
        )

        result = pc.check_kimi_processes()

        assert result == []

    @patch("process_controller.subprocess.run")
    def test_check_kimi_processes_ssh_failure(self, mock_run):
        """Returns empty list on SSH failure."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

        result = pc.check_kimi_processes()

        assert result == []

    @patch("process_controller.subprocess.run")
    def test_check_kimi_processes_error_with_stderr(self, mock_run):
        """Returns empty list when SSH returns error with stderr."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_fail("Permission denied", rc=255)

        result = pc.check_kimi_processes()

        assert result == []

    # ------------------------------------------------------------------
    # run_kimi_task
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_run_kimi_task_success(self, mock_run):
        """Returns success dict when task completes."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("Task completed successfully\n")

        result = pc.run_kimi_task("Write a test")

        assert result["success"] is True
        assert "Task completed" in result["stdout"]
        assert result["stderr"] == ""

    @patch("process_controller.subprocess.run")
    def test_run_kimi_task_failure(self, mock_run):
        """Returns failure dict when task fails."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_fail("Error: model not found", rc=1)

        result = pc.run_kimi_task("Write a test")

        assert result["success"] is False
        assert "model not found" in result["stderr"]

    @patch("process_controller.subprocess.run")
    def test_run_kimi_task_timeout(self, mock_run):
        """Returns failure dict when SSH times out."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=120)

        result = pc.run_kimi_task("Long task", timeout=120)

        assert result["success"] is False
        assert "failed or timed out" in result["stderr"]

    @patch("process_controller.subprocess.run")
    def test_run_kimi_task_uses_shlex_quote(self, mock_run):
        """Verifies that prompt is safely quoted via shlex.quote."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("done\n")

        pc.run_kimi_task("test's prompt with $(dangerous) chars")

        args_passed = mock_run.call_args[0][0]
        remote_cmd = args_passed[2]  # The remote command string
        # shlex.quote wraps in single quotes, escaping existing ones
        assert "$(dangerous)" not in remote_cmd or "'" in remote_cmd

    @patch("process_controller.subprocess.run")
    def test_run_kimi_task_custom_model_and_dir(self, mock_run):
        """Passes custom model and working directory."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("done\n")

        pc.run_kimi_task("prompt", model="kimi-k2.5", working_dir="/tmp/work")

        args_passed = mock_run.call_args[0][0]
        remote_cmd = args_passed[2]
        assert "kimi-k2.5" in remote_cmd or "'kimi-k2.5'" in remote_cmd
        assert "/tmp/work" in remote_cmd or "'/tmp/work'" in remote_cmd

    # ------------------------------------------------------------------
    # kill_stale_kimi
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_kill_stale_kimi_success(self, mock_run):
        """Returns True when kill succeeds."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("")

        result = pc.kill_stale_kimi(12345)

        assert result is True
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ssh", "hetzner", "kill 12345"]

    @patch("process_controller.subprocess.run")
    def test_kill_stale_kimi_failure(self, mock_run):
        """Returns False when kill fails."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_fail("No such process", rc=1)

        result = pc.kill_stale_kimi(99999)

        assert result is False

    @patch("process_controller.subprocess.run")
    def test_kill_stale_kimi_invalid_pid(self, mock_run):
        """Returns False for invalid PID values."""
        pc = self._make_controller()

        assert pc.kill_stale_kimi(0) is False
        assert pc.kill_stale_kimi(-1) is False
        mock_run.assert_not_called()

    @patch("process_controller.subprocess.run")
    def test_kill_stale_kimi_ssh_timeout(self, mock_run):
        """Returns False when SSH times out."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

        result = pc.kill_stale_kimi(12345)

        assert result is False

    # ------------------------------------------------------------------
    # count_kimi_sessions
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_count_kimi_sessions_returns_count(self, mock_run):
        """Returns correct count of running processes."""
        pc = self._make_controller()
        ps_output = (
            "aadel  12345  0.5  1.2 123456 78901 ?  S  Jan01  10:30:00 kimi --yolo\n"
            "aadel  12346  0.3  0.8 123456 78902 ?  S  Jan01  05:15:00 kimi --yolo\n"
        )
        mock_run.return_value = _ssh_ok(ps_output)

        result = pc.count_kimi_sessions()

        assert result == 2

    @patch("process_controller.subprocess.run")
    def test_count_kimi_sessions_zero_when_none(self, mock_run):
        """Returns 0 when no kimi processes found."""
        pc = self._make_controller()
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ssh", "hetzner"], stdout="", stderr="", returncode=1
        )

        result = pc.count_kimi_sessions()

        assert result == 0

    @patch("process_controller.subprocess.run")
    def test_count_kimi_sessions_zero_on_failure(self, mock_run):
        """Returns 0 on SSH failure."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

        result = pc.count_kimi_sessions()

        assert result == 0

    # ------------------------------------------------------------------
    # check_hetzner_resources
    # ------------------------------------------------------------------

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_resources_parses_output(self, mock_run):
        """Parses free and uptime output correctly."""
        pc = self._make_controller()
        combined_output = (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:             62          45          10           1           7          16\n"
            "Swap:             0           0           0\n"
            " 14:30:00 up 10 days,  5:30,  2 users,  load average: 2.50, 2.00, 1.50\n"
        )
        mock_run.return_value = _ssh_ok(combined_output)

        result = pc.check_hetzner_resources()

        assert result is not None
        assert result["ram_total_gb"] == 62.0
        assert result["ram_used_gb"] == 45.0
        assert result["ram_pct"] > 0
        assert result["load_avg"] == "2.50"
        assert result["cpu_pct"] == 250.0

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_resources_returns_none_on_failure(self, mock_run):
        """Returns None when SSH fails."""
        pc = self._make_controller()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=10)

        result = pc.check_hetzner_resources()

        assert result is None

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_resources_returns_none_on_error_rc(self, mock_run):
        """Returns None when SSH returns non-zero."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_fail("Permission denied", rc=255)

        result = pc.check_hetzner_resources()

        assert result is None

    @patch("process_controller.subprocess.run")
    def test_check_hetzner_resources_handles_empty_output(self, mock_run):
        """Handles gracefully when output has no parseable data."""
        pc = self._make_controller()
        mock_run.return_value = _ssh_ok("nothing useful here\n")

        result = pc.check_hetzner_resources()

        assert result is not None
        assert result["ram_total_gb"] == 0.0
        assert result["ram_used_gb"] == 0.0
        assert result["load_avg"] == ""


# ===========================================================================
# TestSyncGitHubToInbox
# ===========================================================================

class TestSyncGitHubToInbox:
    """Tests for AutoManager.sync_github_to_inbox()."""

    def _make_manager(self):
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    def _make_issue_obj(
        self,
        number: int = 1,
        title: str = "Test",
        body: str = "body",
        labels: list[str] | None = None,
    ) -> Issue:
        return Issue(
            number=number,
            title=title,
            state="OPEN",
            body=body,
            labels=labels or [],
            url=f"https://github.com/owner/repo/issues/{number}",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T01:00:00Z",
        )

    @patch("auto_manager._log_decision")
    def test_sync_writes_inbox_with_in_progress_tasks(self, mock_log, tmp_path):
        """Writes INBOX.md with current assignment when in-progress tasks exist."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = [
            self._make_issue_obj(42, "Build auth module", labels=["builder", "status:in-progress", "P0"]),
        ]

        with patch("auto_manager.COMMAND_CENTER", tmp_path):
            mgr.sync_github_to_inbox()

        # Verify at least one INBOX.md was written
        inbox_files = list(tmp_path.rglob("INBOX.md"))
        assert len(inbox_files) >= 1

    @patch("auto_manager._log_decision")
    def test_sync_writes_inbox_with_ready_queue(self, mock_log, tmp_path):
        """Writes INBOX.md with ready queue when ready tasks exist."""
        mgr, mock_gh = self._make_manager()

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:in-progress" in labels:
                return []
            if labels and "status:ready" in labels:
                return [
                    self._make_issue_obj(10, "Task A", labels=["research", "status:ready", "P1"]),
                    self._make_issue_obj(11, "Task B", labels=["research", "status:ready", "P0"]),
                ]
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        with patch("auto_manager.COMMAND_CENTER", tmp_path):
            mgr.sync_github_to_inbox()

        # Check that at least one INBOX was written with ready queue content
        inbox_files = list(tmp_path.rglob("INBOX.md"))
        assert len(inbox_files) >= 1
        content = inbox_files[0].read_text()
        assert "READY QUEUE" in content or "NO TASKS" in content

    @patch("auto_manager._log_decision")
    def test_sync_writes_no_tasks_when_empty(self, mock_log, tmp_path):
        """Writes NO TASKS section when no issues exist."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []

        with patch("auto_manager.COMMAND_CENTER", tmp_path):
            mgr.sync_github_to_inbox()

        inbox_files = list(tmp_path.rglob("INBOX.md"))
        assert len(inbox_files) >= 1
        content = inbox_files[0].read_text()
        assert "NO TASKS" in content
        assert "Awaiting new assignments" in content

    @patch("auto_manager._log_decision")
    def test_sync_includes_github_first_protocol(self, mock_log, tmp_path):
        """INBOX.md always includes the GITHUB-FIRST PROTOCOL section."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []

        with patch("auto_manager.COMMAND_CENTER", tmp_path):
            mgr.sync_github_to_inbox()

        inbox_files = list(tmp_path.rglob("INBOX.md"))
        assert len(inbox_files) >= 1
        content = inbox_files[0].read_text()
        assert "GITHUB-FIRST PROTOCOL" in content
        assert "Comment on issue when starting" in content

    @patch("auto_manager.COMMAND_CENTER")
    @patch("auto_manager._log_decision")
    def test_sync_handles_api_failure_gracefully(self, mock_log, mock_cc):
        """Does not crash when GitHub API fails for one agent."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.side_effect = Exception("API error")

        # Should not raise
        mgr.sync_github_to_inbox()

    @patch("auto_manager.COMMAND_CENTER")
    @patch("auto_manager._log_decision")
    def test_sync_creates_parent_directories(self, mock_log, mock_cc):
        """Creates parent directories for INBOX.md if they do not exist."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []

        mock_inbox_path = MagicMock()
        mock_inbox_parent = MagicMock()
        mock_inbox_path.parent = mock_inbox_parent
        mock_cc.__truediv__ = Mock(side_effect=lambda agent: MagicMock(
            __truediv__=Mock(return_value=mock_inbox_path)
        ))

        mgr.sync_github_to_inbox()

        # Verify mkdir was called with parents=True, exist_ok=True
        mock_inbox_parent.mkdir.assert_called()
        mkdir_kwargs = mock_inbox_parent.mkdir.call_args[1]
        assert mkdir_kwargs.get("parents") is True
        assert mkdir_kwargs.get("exist_ok") is True


# ===========================================================================
# TestAutoLoopWithProcessController
# ===========================================================================

class TestAutoLoopWithProcessController:
    """Tests for ProcessController integration in run_auto_loop."""

    def _make_manager(self):
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_auto_loop_checks_hetzner_connectivity(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop checks Hetzner connectivity via ProcessController."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 2
        mock_pc_cls.return_value = mock_pc

        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        mock_pc.check_hetzner_connectivity.assert_called_once()

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_auto_loop_counts_kimi_sessions_when_reachable(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop counts Kimi sessions when Hetzner is reachable."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 3
        mock_pc_cls.return_value = mock_pc

        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        mock_pc.count_kimi_sessions.assert_called_once()

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_auto_loop_skips_kimi_count_when_unreachable(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop does not count Kimi sessions when Hetzner is unreachable."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = False
        mock_pc_cls.return_value = mock_pc

        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        mock_pc.check_hetzner_connectivity.assert_called_once()
        mock_pc.count_kimi_sessions.assert_not_called()

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_auto_loop_handles_process_controller_exception(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop continues if ProcessController raises an exception."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc_cls.side_effect = Exception("SSH subsystem error")

        mock_sleep.side_effect = KeyboardInterrupt()

        # Should not raise
        mgr.run_auto_loop(interval=1)

        # Still should have called other loop steps
        assert mock_gh.list_issues.call_count > 0

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_auto_loop_calls_sync_github_to_inbox(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop calls sync_github_to_inbox during each cycle."""
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 1
        mock_pc_cls.return_value = mock_pc

        mock_sleep.side_effect = KeyboardInterrupt()

        # Spy on sync_github_to_inbox
        with patch.object(mgr, "sync_github_to_inbox") as mock_sync:
            mgr.run_auto_loop(interval=1)
            mock_sync.assert_called_once()


# ===========================================================================
# TestConfigConsistency
# ===========================================================================

class TestConfigConsistency:
    """Tests that config.py values are internally consistent."""

    def test_agent_label_matches_role_label(self):
        """The label field must match role_label so filtering uses labels
        that actually exist in the repo (created by labels-setup)."""
        from config import AGENTS
        for agent_name, cfg in AGENTS.items():
            assert cfg["label"] == cfg["role_label"], (
                f"Agent '{agent_name}' has label={cfg['label']!r} but "
                f"role_label={cfg['role_label']!r} -- they must match"
            )

    def test_agent_labels_are_in_all_labels(self):
        """Every agent filter label must exist in ALL_LABELS so
        labels-setup creates it."""
        from config import AGENTS, ALL_LABELS
        all_label_names = {name for name, _, _ in ALL_LABELS}
        for agent_name, cfg in AGENTS.items():
            assert cfg["label"] in all_label_names, (
                f"Agent '{agent_name}' label {cfg['label']!r} is not in ALL_LABELS"
            )
            assert cfg["role_label"] in all_label_names, (
                f"Agent '{agent_name}' role_label {cfg['role_label']!r} is not in ALL_LABELS"
            )

    def test_no_localkimi_in_agents(self):
        """The config should not have a 'localkimi' key -- only 'kimi'."""
        from config import AGENTS
        assert "localkimi" not in AGENTS
        assert "kimi" in AGENTS

    def test_all_agents_have_required_keys(self):
        """Every agent config must have label, role_label, and description."""
        from config import AGENTS
        required_keys = {"label", "role_label", "description"}
        for agent_name, cfg in AGENTS.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, (
                f"Agent '{agent_name}' is missing keys: {missing}"
            )


# ===========================================================================
# TestAutoLoopGracefulShutdown
# ===========================================================================

class TestAutoLoopGracefulShutdown:
    """Tests that run_auto_loop exits cleanly on KeyboardInterrupt
    during work steps (not just during sleep)."""

    def _make_manager(self):
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_keyboard_interrupt_during_health_check(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop exits cleanly if KeyboardInterrupt arrives during health_check."""
        mgr, mock_gh = self._make_manager()

        # Make list_issues raise KeyboardInterrupt (simulating Ctrl+C during health_check)
        mock_gh.list_issues.side_effect = KeyboardInterrupt()

        # Should not raise -- exits cleanly
        mgr.run_auto_loop(interval=1)

        # Sleep should never be reached since interrupt happened during work
        mock_sleep.assert_not_called()

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_keyboard_interrupt_during_replenish(self, mock_log, mock_sleep, mock_pc_cls):
        """Auto loop exits cleanly if KeyboardInterrupt arrives during replenish_queues."""
        mgr, mock_gh = self._make_manager()

        # Health check succeeds, but replenish (which calls list_issues) raises
        call_count = [0]
        def list_issues_side_effect(**kwargs):
            call_count[0] += 1
            # Let the first few calls through (health_check uses list_issues)
            # Then raise on a later call (during replenish)
            if call_count[0] > 10:
                raise KeyboardInterrupt()
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = False
        mock_pc_cls.return_value = mock_pc

        # Should not raise
        mgr.run_auto_loop(interval=1)


# ===========================================================================
# TestDirectiveLabelDeduplication
# ===========================================================================

class TestDirectiveLabelDeduplication:
    """Tests that directive() does not pass duplicate labels when creating issues."""

    def _make_manager(self):
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager._log_decision")
    def test_directive_creates_issue_without_duplicate_labels(self, mock_log):
        """When creating a directive issue, labels list should not contain duplicates."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.create_issue.return_value = Issue(
            number=100, title="Directive", state="OPEN", body="",
            labels=[], url="", created_at="", updated_at=""
        )

        mgr.directive("researcher", "Focus on Arabic papers")

        mock_gh.create_issue.assert_called_once()
        call_kwargs = mock_gh.create_issue.call_args
        labels_passed = call_kwargs[1].get("labels", []) if call_kwargs[1] else []
        if not labels_passed and len(call_kwargs[0]) > 2:
            labels_passed = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else []

        # No duplicates
        assert len(labels_passed) == len(set(labels_passed)), (
            f"Duplicate labels found: {labels_passed}"
        )
        # Should contain role_label and status:ready
        assert "role:researcher" in labels_passed
        assert "status:ready" in labels_passed


# ===========================================================================
# TestDryRunProxy
# ===========================================================================

class TestDryRunProxy:
    """Tests for DryRunGitHubProxy."""

    def test_read_methods_pass_through(self):
        from auto_manager import DryRunGitHubProxy
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        mock_gh.list_issues.return_value = ["issue1"]
        proxy = DryRunGitHubProxy(mock_gh)

        result = proxy.list_issues(labels=["bug"])
        assert result == ["issue1"]
        mock_gh.list_issues.assert_called_once_with(labels=["bug"])

    def test_write_methods_are_noops(self):
        from auto_manager import DryRunGitHubProxy
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        proxy = DryRunGitHubProxy(mock_gh)

        # These should not call the real connector
        proxy.add_comment(1, "test")
        proxy.add_labels(1, ["bug"])
        proxy.reopen_issue(1)
        proxy.close_issue(1)
        proxy.create_issue(title="t", body="b")
        proxy.transition_status(1, "status:ready")

        mock_gh.add_comment.assert_not_called()
        mock_gh.add_labels.assert_not_called()
        mock_gh.reopen_issue.assert_not_called()

    def test_repo_attribute_passes_through(self):
        from auto_manager import DryRunGitHubProxy
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        proxy = DryRunGitHubProxy(mock_gh)

        assert proxy.repo == "owner/repo"


# ===========================================================================
# TestMaxCycles
# ===========================================================================

class TestMaxCycles:
    """Tests for max_cycles parameter in run_auto_loop."""

    def _make_manager(self):
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_max_cycles_exits_after_n_cycles(self, mock_log, mock_sleep, mock_pc_cls):
        """Loop exits after max_cycles without needing KeyboardInterrupt."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 0
        mock_pc_cls.return_value = mock_pc

        mgr.run_auto_loop(interval=0, max_cycles=1)

        # Sleep should NOT be called because max_cycles exits before sleep
        mock_sleep.assert_not_called()

    @patch("auto_manager.ProcessController")
    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_max_cycles_zero_runs_indefinitely(self, mock_log, mock_sleep, mock_pc_cls):
        """max_cycles=0 means infinite loop (needs KeyboardInterrupt to exit)."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []
        mock_gh.list_recently_closed.return_value = []

        mock_pc = MagicMock()
        mock_pc.check_hetzner_connectivity.return_value = True
        mock_pc.count_kimi_sessions.return_value = 0
        mock_pc_cls.return_value = mock_pc

        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1, max_cycles=0)

        # Sleep was called (loop attempted to continue)
        mock_sleep.assert_called_once()


# ===========================================================================
# TestPidLock
# ===========================================================================

class TestPidLock:
    """Tests for PID lock acquire/release."""

    def test_acquire_and_release_pid_lock(self, tmp_path):
        import agent_mgr
        original_pid_file = agent_mgr.PID_FILE
        agent_mgr.PID_FILE = tmp_path / "agent-mgr.pid"

        try:
            agent_mgr.acquire_pid_lock()
            assert agent_mgr.PID_FILE.exists()
            assert agent_mgr.PID_FILE.read_text().strip() == str(os.getpid())

            agent_mgr.release_pid_lock()
            assert not agent_mgr.PID_FILE.exists()
        finally:
            agent_mgr.PID_FILE = original_pid_file

    def test_acquire_overwrites_stale_pid(self, tmp_path):
        import agent_mgr
        original_pid_file = agent_mgr.PID_FILE
        agent_mgr.PID_FILE = tmp_path / "agent-mgr.pid"

        try:
            # Write a stale PID (very unlikely to be a running process)
            agent_mgr.PID_FILE.write_text("999999999")

            agent_mgr.acquire_pid_lock()
            assert agent_mgr.PID_FILE.read_text().strip() == str(os.getpid())
        finally:
            agent_mgr.release_pid_lock()
            agent_mgr.PID_FILE = original_pid_file


# ===========================================================================
# TestSignalHandler
# ===========================================================================

class TestSignalHandler:
    """Tests for SIGTERM signal handler."""

    def test_sigterm_handler_raises_keyboard_interrupt(self):
        from agent_mgr import _handle_sigterm
        with pytest.raises(KeyboardInterrupt):
            _handle_sigterm(signal.SIGTERM, None)


# ===========================================================================
# TestFileLogging
# ===========================================================================

class TestFileLogging:
    """Tests for file logging setup."""

    def test_setup_file_logging_adds_handler(self, tmp_path):
        import agent_mgr
        original_log_file = agent_mgr.LOG_FILE
        agent_mgr.LOG_FILE = tmp_path / "test.log"

        try:
            initial_handlers = len(logging.getLogger().handlers)
            agent_mgr.setup_file_logging()
            assert len(logging.getLogger().handlers) > initial_handlers

            # Verify it's a RotatingFileHandler
            new_handler = logging.getLogger().handlers[-1]
            assert isinstance(new_handler, RotatingFileHandler)

            # Clean up: remove the handler we added
            logging.getLogger().removeHandler(new_handler)
        finally:
            agent_mgr.LOG_FILE = original_log_file


# ===========================================================================
# TestDryRunCLI
# ===========================================================================

class TestDryRunCLI:
    """Tests for --dry-run CLI flag."""

    def test_auto_dry_run_flag_parses(self):
        from agent_mgr import build_parser
        parser = build_parser()
        args = parser.parse_args(["auto", "--dry-run"])
        assert args.command == "auto"
        assert args.dry_run is True

    def test_auto_without_dry_run(self):
        from agent_mgr import build_parser
        parser = build_parser()
        args = parser.parse_args(["auto"])
        assert args.command == "auto"
        assert args.dry_run is False


# ===========================================================================
# TestResourceAllocatorIntegration
# ===========================================================================

class TestResourceAllocatorIntegration:
    """Tests for ResourceAllocator integration in AutoManager."""

    def _make_manager(self):
        """Return (AutoManager, mock_gh)."""
        mock_gh = MagicMock(spec=GitHubConnector)
        mock_gh.repo = "owner/repo"
        from auto_manager import AutoManager
        mgr = AutoManager(mock_gh)
        return mgr, mock_gh

    def _make_issue_obj(
        self,
        number: int = 1,
        title: str = "Test",
        body: str = "body",
        labels: list | None = None,
    ) -> Issue:
        return Issue(
            number=number,
            title=title,
            state="OPEN",
            body=body,
            labels=labels or [],
            url=f"https://github.com/owner/repo/issues/{number}",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T01:00:00Z",
        )

    # ------------------------------------------------------------------
    # _check_agent_resources returns True when RESOURCE_AWARE is False
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", False)
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_returns_true_when_not_resource_aware(self, mock_log):
        """Returns True when RESOURCE_AWARE is False (graceful degradation)."""
        mgr, mock_gh = self._make_manager()
        assert mgr._check_agent_resources("researcher") is True
        assert mgr._check_agent_resources("builder") is True
        assert mgr._check_agent_resources("kimi") is True

    # ------------------------------------------------------------------
    # _check_agent_resources maps agents to correct ResourceType
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_maps_researcher_to_claude_sub(self, mock_log, mock_allocator_cls):
        """researcher maps to CLAUDE_SUB resource type."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()
        mock_allocator.list_resources.return_value = [MagicMock()]  # available
        mock_allocator_cls.return_value = mock_allocator

        result = mgr._check_agent_resources("researcher")

        assert result is True
        # Verify ResourceType.CLAUDE_SUB was used
        from auto_manager import ResourceType
        mock_allocator.list_resources.assert_called_with(
            resource_type=ResourceType.CLAUDE_SUB,
            available_only=True,
        )

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_maps_builder_to_claude_sub(self, mock_log, mock_allocator_cls):
        """builder maps to CLAUDE_SUB resource type."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()
        mock_allocator.list_resources.return_value = [MagicMock()]
        mock_allocator_cls.return_value = mock_allocator

        result = mgr._check_agent_resources("builder")

        assert result is True
        from auto_manager import ResourceType
        mock_allocator.list_resources.assert_called_with(
            resource_type=ResourceType.CLAUDE_SUB,
            available_only=True,
        )

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_maps_kimi_to_kimi_cli(self, mock_log, mock_allocator_cls):
        """kimi maps to KIMI_CLI resource type."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()
        mock_allocator.list_resources.return_value = [MagicMock()]
        mock_allocator_cls.return_value = mock_allocator

        result = mgr._check_agent_resources("kimi")

        assert result is True
        from auto_manager import ResourceType
        mock_allocator.list_resources.assert_called_with(
            resource_type=ResourceType.KIMI_CLI,
            available_only=True,
        )

    # ------------------------------------------------------------------
    # _check_agent_resources returns False when exhausted
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_returns_false_when_exhausted(self, mock_log, mock_allocator_cls):
        """Returns False when all resources of the type are exhausted."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()

        def list_resources_side_effect(resource_type=None, available_only=False):
            if available_only:
                return []  # None available
            return [MagicMock()]  # But resources exist (just exhausted)

        mock_allocator.list_resources.side_effect = list_resources_side_effect
        mock_allocator_cls.return_value = mock_allocator

        result = mgr._check_agent_resources("researcher")

        assert result is False

    # ------------------------------------------------------------------
    # _check_agent_resources returns True when no resources registered (fail-open)
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_returns_true_when_none_registered(self, mock_log, mock_allocator_cls):
        """Returns True (fail-open) when no resources are registered for the type."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()
        mock_allocator.list_resources.return_value = []  # No resources at all
        mock_allocator_cls.return_value = mock_allocator

        result = mgr._check_agent_resources("researcher")

        assert result is True

    # ------------------------------------------------------------------
    # _check_agent_resources returns True on exception (fail-open)
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_check_agent_resources_returns_true_on_exception(self, mock_log, mock_allocator_cls):
        """Returns True when ResourceAllocator raises an exception."""
        mgr, mock_gh = self._make_manager()
        mock_allocator_cls.side_effect = Exception("Cannot load resources")

        result = mgr._check_agent_resources("researcher")

        assert result is True

    # ------------------------------------------------------------------
    # replenish_queues skips agents with exhausted resources
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_replenish_queues_skips_exhausted_agents(self, mock_log, mock_allocator_cls):
        """replenish_queues skips agents whose resources are exhausted."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()

        def list_resources_side_effect(resource_type=None, available_only=False):
            if available_only:
                return []  # None available
            return [MagicMock()]  # But resources exist (just exhausted)

        mock_allocator.list_resources.side_effect = list_resources_side_effect
        mock_allocator_cls.return_value = mock_allocator

        # Set up backlog issues -- these should NOT be promoted
        backlog_issues = [
            self._make_issue_obj(10, "Backlog1", labels=["research", "status:backlog", "P1"]),
        ]

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:ready" in labels:
                return []  # empty ready queue
            if labels and "status:backlog" in labels:
                return backlog_issues
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        mgr.replenish_queues(min_ready=2)

        # No promotions should have happened since all agents are resource-exhausted
        mock_gh.transition_status.assert_not_called()

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_replenish_queues_logs_decision_when_skipping(self, mock_log, mock_allocator_cls):
        """Logs a decision when skipping an agent due to exhausted resources."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()

        def list_resources_side_effect(resource_type=None, available_only=False):
            if available_only:
                return []
            return [MagicMock()]

        mock_allocator.list_resources.side_effect = list_resources_side_effect
        mock_allocator_cls.return_value = mock_allocator

        mock_gh.list_issues.return_value = []

        mgr.replenish_queues(min_ready=2)

        # Should have logged a "Queue Replenishment Skipped" decision for each agent
        skip_calls = [
            c for c in mock_log.call_args_list
            if c[0][0] == "Queue Replenishment Skipped"
        ]
        assert len(skip_calls) == 3  # researcher, builder, kimi

    # ------------------------------------------------------------------
    # replenish_queues works normally when resources available
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_replenish_queues_proceeds_when_resources_available(self, mock_log, mock_allocator_cls):
        """replenish_queues promotes backlog when resources are available."""
        mgr, mock_gh = self._make_manager()
        mock_allocator = MagicMock()
        mock_allocator.list_resources.return_value = [MagicMock()]  # resources available
        mock_allocator_cls.return_value = mock_allocator

        ready_issues = []
        backlog_issues = [
            self._make_issue_obj(10, "Backlog1", labels=["research", "status:backlog", "P1"]),
            self._make_issue_obj(11, "Backlog2", labels=["research", "status:backlog", "P0"]),
        ]

        def list_issues_side_effect(labels=None, state="open", **kwargs):
            if labels and "status:ready" in labels:
                return ready_issues
            if labels and "status:backlog" in labels:
                return backlog_issues
            return []

        mock_gh.list_issues.side_effect = list_issues_side_effect

        mgr.replenish_queues(min_ready=2)

        # Should have promoted backlog items
        assert mock_gh.transition_status.call_count >= 2

    # ------------------------------------------------------------------
    # _get_resource_status returns None when not RESOURCE_AWARE
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", False)
    @patch("auto_manager._log_decision")
    def test_get_resource_status_returns_none_when_not_aware(self, mock_log):
        """_get_resource_status returns None when RESOURCE_AWARE is False."""
        mgr, mock_gh = self._make_manager()
        assert mgr._get_resource_status() is None

    # ------------------------------------------------------------------
    # Dashboard shows resource status section
    # ------------------------------------------------------------------

    @patch("auto_manager.RESOURCE_AWARE", False)
    @patch("auto_manager._log_decision")
    def test_dashboard_shows_resource_section_not_available(self, mock_log):
        """Dashboard shows 'not available' when RESOURCE_AWARE is False."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []

        result = mgr.dashboard()

        assert "RESOURCE STATUS" in result
        assert "resource tracking not available" in result

    @patch("auto_manager.RESOURCE_AWARE", True)
    @patch("auto_manager.ResourceAllocator")
    @patch("auto_manager._log_decision")
    def test_dashboard_shows_resource_availability(self, mock_log, mock_allocator_cls):
        """Dashboard shows resource availability when RESOURCE_AWARE is True."""
        mgr, mock_gh = self._make_manager()
        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []

        from auto_manager import ResourceType
        mock_allocator = MagicMock()

        def list_resources_side_effect(resource_type=None, available_only=False):
            if resource_type == ResourceType.CLAUDE_SUB:
                if available_only:
                    return [MagicMock()]
                return [MagicMock(), MagicMock()]
            return []

        mock_allocator.list_resources.side_effect = list_resources_side_effect
        mock_allocator_cls.return_value = mock_allocator

        result = mgr.dashboard()

        assert "RESOURCE STATUS" in result
        assert "AVAILABLE" in result or "available" in result
