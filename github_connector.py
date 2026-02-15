"""GitHub Connector — wraps the ``gh`` CLI to provide a typed Python API.

This module is the backbone of the agent management system.  Every interaction
with GitHub Issues and Labels goes through :class:`GitHubConnector`, which
shells out to ``gh`` via :func:`subprocess.run` and returns parsed JSON.

No external dependencies are required — only the Python 3.11+ standard library
and a working ``gh`` CLI installation that is already authenticated.

Author: Ahmed Adel Bakr Alderai
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GitHubConnectorError(Exception):
    """Base exception for all connector errors."""

    def __init__(self, message: str, *, cmd: str = "", stderr: str = "") -> None:
        self.cmd = cmd
        self.stderr = stderr
        super().__init__(message)


class CLINotFoundError(GitHubConnectorError):
    """Raised when the ``gh`` binary is not found on PATH."""


class CLIAuthError(GitHubConnectorError):
    """Raised when ``gh`` reports an authentication problem."""


class CLIExecutionError(GitHubConnectorError):
    """Raised when ``gh`` exits with a non-zero return code."""


class JSONParseError(GitHubConnectorError):
    """Raised when the CLI output cannot be decoded as JSON."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Issue:
    """Lightweight representation of a GitHub issue."""

    number: int
    title: str
    state: str
    body: str = ""
    labels: list[str] = field(default_factory=list)
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    comments_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Issue:
        """Build an ``Issue`` from a dict returned by ``gh``."""
        labels_raw = data.get("labels") or []
        if labels_raw and isinstance(labels_raw[0], dict):
            labels = [lb.get("name", "") for lb in labels_raw]
        else:
            labels = [str(lb) for lb in labels_raw]

        return cls(
            number=int(data.get("number", 0)),
            title=data.get("title", ""),
            state=data.get("state", ""),
            body=data.get("body", ""),
            labels=labels,
            url=data.get("url", ""),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
            comments_count=int(data.get("comments", {}).get("totalCount", 0))
            if isinstance(data.get("comments"), dict)
            else int(data.get("commentsCount", 0)),
        )


@dataclass(frozen=True, slots=True)
class Label:
    """Lightweight representation of a GitHub label."""

    name: str
    color: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Label:
        return cls(
            name=data.get("name", ""),
            color=data.get("color", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class Comment:
    """Lightweight representation of an issue comment."""

    id: str
    body: str
    author: str
    created_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Comment:
        author_raw = data.get("author") or {}
        if isinstance(author_raw, dict):
            author = author_raw.get("login", "")
        else:
            author = str(author_raw)

        return cls(
            id=str(data.get("id", "")),
            body=data.get("body", ""),
            author=author,
            created_at=data.get("createdAt", ""),
        )


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

_ISSUE_FIELDS = (
    "number,title,state,body,labels,url,createdAt,updatedAt,comments"
)

_COMMENT_FIELDS = "id,body,author,createdAt"


class GitHubConnector:
    """Typed Python wrapper around the ``gh`` CLI.

    Parameters
    ----------
    repo:
        The *owner/name* slug of the repository (e.g. ``"Ahmed-AdelB/ummro"``).
    timeout:
        Maximum number of seconds to wait for each ``gh`` invocation.
    """

    def __init__(
        self,
        repo: str = "Ahmed-AdelB/ummro",
        *,
        timeout: int = 30,
    ) -> None:
        self.repo = repo
        self.timeout = timeout
        self._verify_cli()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_cli(self) -> None:
        """Assert that ``gh`` is installed and authenticated."""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise CLINotFoundError(
                    "gh CLI returned a non-zero exit code on --version",
                    cmd="gh --version",
                    stderr=result.stderr.strip(),
                )
        except FileNotFoundError as exc:
            raise CLINotFoundError(
                "gh CLI is not installed or not on PATH. "
                "Install it from https://cli.github.com/"
            ) from exc

        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise CLIAuthError(
                "gh CLI is not authenticated. Run `gh auth login` first.",
                cmd="gh auth status",
                stderr=result.stderr.strip(),
            )

    def _run(self, args: list[str]) -> str:
        """Execute a ``gh`` command and return its stdout.

        Uses list-mode ``subprocess.run`` (no shell) for safety — each
        argument is passed directly to the process without shell
        interpretation, preventing injection.

        Raises
        ------
        CLIExecutionError
            When ``gh`` exits with a non-zero return code.
        """
        full_args: list[str] = ["gh"] + args
        # Log with shlex.quote for readability, but pass raw list to subprocess
        cmd_str = " ".join(shlex.quote(a) for a in full_args)
        logger.debug("gh exec: %s", cmd_str)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    full_args,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise CLIExecutionError(
                    f"gh command timed out after {self.timeout}s",
                    cmd=cmd_str,
                ) from exc

            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Detect rate limiting and retry with backoff
                if ("rate limit" in stderr.lower()
                        or "api rate" in stderr.lower()
                        or "403" in stderr):
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)  # 2, 4 seconds
                        logger.warning(
                            "Rate limited (attempt %d/%d), retrying in %ds: %s",
                            attempt + 1, max_retries, wait, stderr[:100],
                        )
                        import time
                        time.sleep(wait)
                        continue
                raise CLIExecutionError(
                    f"gh command failed (rc={result.returncode}): {stderr}",
                    cmd=cmd_str,
                    stderr=stderr,
                )

            return result.stdout

        # Should not reach here, but satisfy type checker
        raise CLIExecutionError("Exhausted retries", cmd=cmd_str)

    def _run_json(self, args: list[str]) -> Any:
        """Execute a ``gh`` command and parse its JSON output."""
        raw = self._run(args)
        if not raw.strip():
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JSONParseError(
                f"Failed to parse gh output as JSON: {exc}",
                cmd=" ".join(args),
            ) from exc

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def list_issues(
        self,
        *,
        labels: list[str] | None = None,
        state: str = "open",
        limit: int = 100,
    ) -> list[Issue]:
        """List issues in the repository.

        Parameters
        ----------
        labels:
            Optional list of label names to filter by.
        state:
            One of ``"open"``, ``"closed"``, or ``"all"``.
        limit:
            Maximum number of issues to return.

        Returns
        -------
        list[Issue]
            Parsed issue objects sorted by most recently updated first.
        """
        args = [
            "issue", "list",
            "--repo", self.repo,
            "--state", state,
            "--limit", str(limit),
            "--json", _ISSUE_FIELDS,
        ]
        if labels:
            args.extend(["--label", ",".join(labels)])

        data = self._run_json(args)
        return [Issue.from_dict(item) for item in data]

    def get_issue(self, number: int) -> dict[str, Any]:
        """Get full details for a single issue, including comments.

        Returns a dict with keys ``"issue"`` (:class:`Issue`) and
        ``"comments"`` (list of :class:`Comment`).
        """
        issue_data = self._run_json([
            "issue", "view", str(number),
            "--repo", self.repo,
            "--json", _ISSUE_FIELDS,
        ])
        issue = Issue.from_dict(issue_data)

        comments_data = self._run_json([
            "issue", "view", str(number),
            "--repo", self.repo,
            "--json", "comments",
        ])
        raw_comments = comments_data.get("comments", [])
        comments = [Comment.from_dict(c) for c in raw_comments]

        return {"issue": issue, "comments": comments}

    def create_issue(
        self,
        title: str,
        body: str,
        *,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create a new issue and return it.

        Parameters
        ----------
        title:
            Issue title.
        body:
            Markdown body.
        labels:
            Optional list of label names to attach.

        Returns
        -------
        Issue
            The newly created issue.
        """
        args = [
            "issue", "create",
            "--repo", self.repo,
            "--title", title,
            "--body", body,
        ]
        if labels:
            for label in labels:
                args.extend(["--label", label])

        # gh issue create outputs the URL; we parse the issue number from it.
        raw_output = self._run(args).strip()
        logger.info("Created issue: %s", raw_output)

        # The last line is typically the URL like
        # https://github.com/owner/repo/issues/42
        url_line = raw_output.splitlines()[-1].strip()
        try:
            issue_number = int(url_line.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            # Fallback: try to fetch the latest issue by title.
            logger.warning(
                "Could not parse issue number from URL: %s. "
                "Falling back to search.",
                url_line,
            )
            results = self.search_issues(f"{title} in:title")
            if results:
                return results[0]
            raise GitHubConnectorError(
                f"Created issue but could not determine its number. "
                f"gh output: {raw_output}"
            )

        full = self.get_issue(issue_number)
        return full["issue"]

    def close_issue(self, number: int) -> None:
        """Close an issue."""
        self._run([
            "issue", "close", str(number),
            "--repo", self.repo,
        ])
        logger.info("Closed issue #%d in %s", number, self.repo)

    def reopen_issue(self, number: int) -> None:
        """Reopen a previously closed issue."""
        self._run([
            "issue", "reopen", str(number),
            "--repo", self.repo,
        ])
        logger.info("Reopened issue #%d in %s", number, self.repo)

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    def add_comment(self, number: int, body: str) -> None:
        """Add a comment to an issue.

        Parameters
        ----------
        number:
            Issue number.
        body:
            Markdown comment body.
        """
        self._run([
            "issue", "comment", str(number),
            "--repo", self.repo,
            "--body", body,
        ])
        logger.info("Commented on issue #%d in %s", number, self.repo)

    def get_comments(self, number: int) -> list[Comment]:
        """Return all comments on an issue."""
        data = self._run_json([
            "issue", "view", str(number),
            "--repo", self.repo,
            "--json", "comments",
        ])
        return [Comment.from_dict(c) for c in data.get("comments", [])]

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def add_label(self, number: int, label: str) -> None:
        """Add a label to an issue."""
        self._run([
            "issue", "edit", str(number),
            "--repo", self.repo,
            "--add-label", label,
        ])
        logger.info(
            "Added label %r to issue #%d in %s", label, number, self.repo
        )

    def remove_label(self, number: int, label: str) -> None:
        """Remove a label from an issue."""
        self._run([
            "issue", "edit", str(number),
            "--repo", self.repo,
            "--remove-label", label,
        ])
        logger.info(
            "Removed label %r from issue #%d in %s", label, number, self.repo
        )

    def transition_status(self, number: int, new_status: str) -> None:
        """Atomically transition an issue to *new_status*.

        Removes any existing ``status:*`` labels before adding the new one,
        preventing multiple status labels from coexisting (BUG-3 fix).

        Parameters
        ----------
        number:
            Issue number.
        new_status:
            The target status label (e.g. ``"status:ready"``).
        """
        if not new_status.startswith("status:"):
            new_status = f"status:{new_status}"

        # Get current labels to find existing status labels
        issue_data = self._run_json([
            "issue", "view", str(number),
            "--repo", self.repo,
            "--json", "labels",
        ])
        current_labels = issue_data.get("labels", [])
        old_statuses = [
            lb["name"] for lb in current_labels
            if isinstance(lb, dict) and lb.get("name", "").startswith("status:")
            and lb["name"] != new_status
        ]

        # Remove old status labels
        for old in old_statuses:
            try:
                self.remove_label(number, old)
            except CLIExecutionError:
                logger.debug("Could not remove label %r from #%d", old, number)

        # Add new status
        self.add_label(number, new_status)
        logger.info(
            "Transitioned #%d to %s (removed: %s)",
            number, new_status, old_statuses or "none",
        )

    def list_labels(self) -> list[Label]:
        """List all labels defined in the repository."""
        data = self._run_json([
            "label", "list",
            "--repo", self.repo,
            "--json", "name,color,description",
            "--limit", "200",
        ])
        return [Label.from_dict(item) for item in data]

    def create_label(
        self,
        name: str,
        color: str,
        description: str = "",
    ) -> None:
        """Create a new label in the repository.

        Parameters
        ----------
        name:
            Label name.
        color:
            Hex color **without** the ``#`` prefix (e.g. ``"0075ca"``).
        description:
            Optional human-readable description.
        """
        args = [
            "label", "create", name,
            "--repo", self.repo,
            "--color", color,
        ]
        if description:
            args.extend(["--description", description])

        self._run(args)
        logger.info("Created label %r in %s", name, self.repo)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_issues(
        self,
        query: str,
        *,
        limit: int = 30,
    ) -> list[Issue]:
        """Search issues in the repository using GitHub search syntax.

        Parameters
        ----------
        query:
            A GitHub search query string (e.g. ``"bug label:urgent"``).
        limit:
            Maximum results to return.

        Returns
        -------
        list[Issue]
            Matching issues.
        """
        full_query = f"repo:{self.repo} is:issue {query}"
        data = self._run_json([
            "search", "issues",
            "--json", _ISSUE_FIELDS,
            "--limit", str(limit),
            "--", full_query,
        ])
        return [Issue.from_dict(item) for item in data]

    # ------------------------------------------------------------------
    # Recently closed issues & close detection
    # ------------------------------------------------------------------

    def list_recently_closed(
        self,
        *,
        hours: int = 24,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List issues closed within the last *hours* hours.

        Fetches closed issues and filters by ``closedAt`` timestamp.

        Parameters
        ----------
        hours:
            Look-back window in hours (default 24).
        limit:
            Maximum number of closed issues to fetch from the API.

        Returns
        -------
        list[dict[str, Any]]
            Recently closed issue dicts with keys: ``number``, ``title``,
            ``labels`` (list[str]), ``closedAt`` (ISO string).
        """
        data = self._run_json([
            "issue", "list",
            "--repo", self.repo,
            "--state", "closed",
            "--json", "number,title,labels,closedAt",
            "--limit", str(limit),
        ])

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result: list[dict[str, Any]] = []

        for item in data:
            closed_at = item.get("closedAt", "")
            if not closed_at:
                continue

            try:
                closed_dt = datetime.fromisoformat(
                    closed_at.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                logger.warning("Unparseable closedAt timestamp: %r", closed_at)
                continue

            if closed_dt >= cutoff:
                labels_raw = item.get("labels") or []
                if labels_raw and isinstance(labels_raw[0], dict):
                    labels = [lb.get("name", "") for lb in labels_raw]
                else:
                    labels = [str(lb) for lb in labels_raw]

                result.append({
                    "number": int(item.get("number", 0)),
                    "title": item.get("title", ""),
                    "labels": labels,
                    "closedAt": closed_at,
                })

        return result

    def get_issue_closer(self, number: int) -> str | None:
        """Return the GitHub username of the user who most recently closed
        issue *number*.

        Queries the issue timeline events API and extracts the actor login
        from the last ``closed`` event.

        Parameters
        ----------
        number:
            Issue number.

        Returns
        -------
        str | None
            The login of the closer, or ``None`` if it cannot be determined.
        """
        try:
            raw = self._run([
                "api",
                f"repos/{self.repo}/issues/{number}/events",
                "--jq",
                '[.[] | select(.event == "closed")] | last | .actor.login',
            ])
            login = raw.strip().strip('"')
            if login and login != "null":
                return login
            return None
        except CLIExecutionError:
            logger.warning(
                "Could not determine closer for issue #%d", number
            )
            return None

    # ------------------------------------------------------------------
    # Health / monitoring helpers
    # ------------------------------------------------------------------

    def get_latest_comment_time(
        self,
        label: str,
    ) -> datetime | None:
        """Return the timestamp of the most recent comment on any issue
        carrying *label*.

        This is useful for health-check monitoring: if no agent has commented
        on its labeled issues for a while, something is wrong.

        Returns
        -------
        datetime | None
            The UTC datetime of the latest comment, or ``None`` if no
            matching issues or comments exist.
        """
        issues = self.list_issues(labels=[label], limit=50)
        if not issues:
            return None

        latest: datetime | None = None

        for issue in issues:
            comments = self.get_comments(issue.number)
            for comment in comments:
                if not comment.created_at:
                    continue
                try:
                    ts = datetime.fromisoformat(
                        comment.created_at.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(
                        "Unparseable timestamp %r on issue #%d",
                        comment.created_at,
                        issue.number,
                    )
                    continue
                if latest is None or ts > latest:
                    latest = ts

        return latest

    def get_health_summary(self, label: str) -> dict[str, Any]:
        """Return a health summary for issues tagged with *label*.

        Returns a dict with:
        - ``total``: total issue count
        - ``open``: open issue count
        - ``latest_comment``: ISO timestamp of most recent comment or None
        - ``staleness_minutes``: minutes since the latest comment or None
        """
        issues = self.list_issues(labels=[label], state="all", limit=200)
        open_count = sum(1 for i in issues if i.state.upper() == "OPEN")
        latest = self.get_latest_comment_time(label)

        staleness: float | None = None
        if latest is not None:
            delta = datetime.now(timezone.utc) - latest
            staleness = delta.total_seconds() / 60.0

        return {
            "label": label,
            "total": len(issues),
            "open": open_count,
            "latest_comment": latest.isoformat() if latest else None,
            "staleness_minutes": round(staleness, 1) if staleness is not None else None,
        }

    # ------------------------------------------------------------------
    # Convenience methods (used by agent_mgr.py CLI)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agent_config() -> dict[str, dict[str, str]]:
        """Load agent config from config.py, mapping keys for CLI compatibility."""
        from config import AGENTS, STALE_HOURS
        return AGENTS

    @staticmethod
    def _get_stale_hours() -> float:
        from config import STALE_HOURS
        return float(STALE_HOURS)

    def add_labels(self, number: int, labels: list[str]) -> None:
        """Add multiple labels to an issue."""
        for label in labels:
            self.add_label(number, label)

    def post_comment(self, number: int, body: str) -> None:
        """Alias for :meth:`add_comment`."""
        self.add_comment(number, body)

    def create_or_update_label(
        self,
        name: str,
        color: str,
        description: str = "",
    ) -> None:
        """Create a label, or update it if it already exists."""
        existing = {lb.name: lb for lb in self.list_labels()}
        if name in existing:
            # gh label edit
            args = [
                "label", "edit", name,
                "--repo", self.repo,
                "--color", color,
            ]
            if description:
                args.extend(["--description", description])
            try:
                self._run(args)
                logger.info("Updated label %r in %s", name, self.repo)
            except CLIExecutionError:
                logger.debug("Label %r already up to date", name)
        else:
            self.create_label(name, color, description)

    def agent_health(self, agent: str) -> dict[str, Any]:
        """Return health info for *agent*.

        Returns a dict with keys:
        - ``last_activity``: ``datetime | None``
        - ``healthy``: ``bool``
        - ``stale_hours``: ``float``
        - ``current_issue``: ``dict | None``
        """
        cfg = self._get_agent_config().get(agent)
        if cfg is None:
            return {
                "last_activity": None,
                "healthy": False,
                "stale_hours": 0,
                "current_issue": None,
            }

        # Find current in-progress issue
        in_progress = self.list_issues(
            labels=[cfg["label"], "status:in-progress"],
            limit=5,
        )
        current_issue: dict[str, Any] | None = None
        if in_progress:
            iss = in_progress[0]
            current_issue = {
                "number": iss.number,
                "title": iss.title,
            }

        # Find latest comment time
        latest = self.get_latest_comment_time(cfg["label"])
        stale_hours: float = 0.0
        if latest is not None:
            delta = datetime.now(timezone.utc) - latest
            stale_hours = delta.total_seconds() / 3600.0

        return {
            "last_activity": latest,
            "healthy": stale_hours < self._get_stale_hours() if latest else False,
            "stale_hours": round(stale_hours, 1),
            "current_issue": current_issue,
        }

    def agent_queue(self, agent: str) -> dict[str, int]:
        """Return queue stats for *agent*.

        Returns ``{"ready": N, "in_progress": N, "total": N}``.
        """
        cfg = self._get_agent_config().get(agent)
        if cfg is None:
            return {"ready": 0, "in_progress": 0, "total": 0}

        all_issues = self.list_issues(labels=[cfg["label"]], limit=100)
        ready = sum(
            1
            for i in all_issues
            if "status:ready" in i.labels
        )
        in_progress = sum(
            1
            for i in all_issues
            if "status:in-progress" in i.labels
        )
        return {
            "ready": ready,
            "in_progress": in_progress,
            "total": len(all_issues),
        }

    def agent_queue_issues(self, agent: str) -> list[dict[str, Any]]:
        """Return all open issues for *agent* as plain dicts."""
        cfg = self._get_agent_config().get(agent)
        if cfg is None:
            return []

        issues = self.list_issues(labels=[cfg["label"]], limit=100)
        return [
            {
                "number": iss.number,
                "title": iss.title,
                "labels": [{"name": lb} for lb in iss.labels],
            }
            for iss in issues
        ]

    def needs_review_issues(self) -> list[dict[str, Any]]:
        """Return all open issues with ``status:needs-review`` as dicts."""
        issues = self.list_issues(labels=["status:needs-review"], limit=100)
        return [
            {
                "number": iss.number,
                "title": iss.title,
                "labels": [{"name": lb} for lb in iss.labels],
            }
            for iss in issues
        ]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo = sys.argv[1] if len(sys.argv) > 1 else "Ahmed-AdelB/ummro"
    print(f"Connecting to repo: {repo}\n")

    try:
        gh = GitHubConnector(repo=repo)
    except GitHubConnectorError as exc:
        print(f"FATAL: {exc}")
        sys.exit(1)

    # -- Labels --
    print("=== Labels ===")
    labels = gh.list_labels()
    for lb in labels[:10]:
        print(f"  [{lb.color}] {lb.name}: {lb.description}")
    if len(labels) > 10:
        print(f"  ... and {len(labels) - 10} more")

    # -- Open issues --
    print("\n=== Open Issues (up to 10) ===")
    issues = gh.list_issues(limit=10)
    for iss in issues:
        tags = ", ".join(iss.labels) if iss.labels else "none"
        print(f"  #{iss.number}  {iss.title}  [{tags}]")

    if not issues:
        print("  (no open issues)")

    # -- Search demo --
    print("\n=== Search: 'bug' ===")
    results = gh.search_issues("bug", limit=5)
    for iss in results:
        print(f"  #{iss.number}  {iss.title}")
    if not results:
        print("  (no results)")

    print("\nDone.")
