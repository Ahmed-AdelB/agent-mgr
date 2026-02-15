"""
Comprehensive unit tests for agent-mgr system.

Covers:
- TestGitHubConnector: subprocess.run mocking, error handling, security
- TestAutoManager: GitHubConnector mocking, decision logic
- TestCLI: argparse validation

Author: Ahmed Adel Bakr Alderai
"""
from __future__ import annotations

import json
import subprocess
import sys
import os
from datetime import datetime, timezone, timedelta
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

    @patch("auto_manager.time.sleep")
    @patch("auto_manager._log_decision")
    def test_run_auto_loop_cycles_through_all_checks(self, mock_log, mock_sleep):
        mgr, mock_gh = self._make_manager()

        mock_gh.list_issues.return_value = []
        mock_gh.get_comments.return_value = []

        # Make sleep raise KeyboardInterrupt to exit the loop after one cycle
        mock_sleep.side_effect = KeyboardInterrupt()

        mgr.run_auto_loop(interval=1)

        # Verify that list_issues was called (health_check, replenish, dependencies, nudge all use it)
        assert mock_gh.list_issues.call_count > 0


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
