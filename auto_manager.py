"""
auto_manager.py - Smart agent management logic for the command center.

Orchestrates researcher, builder, and kimi agents via GitHub Issues.
Handles task assignment, queue management, health monitoring, dependency
resolution, and automated quality review.

Author: Ahmed Adel Bakr Alderai
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    AGENTS,
    COMMAND_CENTER,
    PRIORITY_LABELS,
    REPO,
    STATUS_LABELS,
    DECISIONS_FILE,
    VALIDATE_SCRIPT,
    STALE_HOURS,
    DEAD_HOURS,
    MIN_READY_TASKS,
    AUTO_LOOP_INTERVAL,
    MANAGER_GITHUB_USER,
)
from github_connector import GitHubConnector
from process_controller import ProcessController

DECISIONS_PATH: Path = DECISIONS_FILE

# ---------------------------------------------------------------------------
# ANSI helpers (no external deps)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_WHITE = "\033[37m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"

logger = logging.getLogger("auto_manager")


def _color(text: str, *codes: str) -> str:
    """Wrap *text* with ANSI escape codes."""
    return "".join(codes) + text + _RESET


def _relative_time(iso_timestamp: str | None) -> str:
    """Return a human-readable relative time string from an ISO-8601 timestamp."""
    if not iso_timestamp:
        return "never"
    try:
        # Handle both 'Z' suffix and '+00:00' offset formats
        ts = iso_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        delta = datetime.now(timezone.utc) - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "just now"
        if total_seconds < 60:
            return f"{total_seconds}s ago"
        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except (ValueError, TypeError):
        return "unknown"


def _hours_since(iso_timestamp: str | None) -> float:
    """Return the number of hours elapsed since *iso_timestamp*."""
    if not iso_timestamp:
        return float("inf")
    try:
        ts = iso_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        delta = datetime.now(timezone.utc) - dt
        return delta.total_seconds() / 3600.0
    except (ValueError, TypeError):
        return float("inf")


def _extract_priority(labels: list[str]) -> int:
    """Return a numeric priority from a list of label names (0 = highest)."""
    for i, plabel in enumerate(PRIORITY_LABELS):
        if plabel in labels:
            return i
    return len(PRIORITY_LABELS)  # no priority label -> lowest


def _extract_status(labels: list[str]) -> str | None:
    """Return the first matching status label, or None."""
    for lbl in labels:
        if lbl in STATUS_LABELS:
            return lbl
    return None


def _priority_label(labels: list[str]) -> str:
    """Return the priority label string (e.g. 'P0') or 'P?' if missing."""
    for plabel in PRIORITY_LABELS:
        if plabel in labels:
            return plabel
    return "P?"


# ---------------------------------------------------------------------------
# Decision logger
# ---------------------------------------------------------------------------


def _log_decision(
    decision_type: str,
    decision: str,
    rationale: str = "",
) -> None:
    """Append a decision entry to DECISIONS.md."""
    try:
        DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Determine next decision number by scanning existing entries
        number = 1
        if DECISIONS_PATH.exists():
            content = DECISIONS_PATH.read_text(encoding="utf-8")
            matches = re.findall(r"## Decision #(\d+)", content)
            if matches:
                number = max(int(m) for m in matches) + 1

        entry = (
            f"\n## Decision #{number} -- {now}\n"
            f"**Type:** {decision_type}\n"
            f"**Decision:** {decision}\n"
        )
        if rationale:
            entry += f"**Rationale:** {rationale}\n"
        entry += f"**Source:** auto_manager.py (automated)\n"

        with DECISIONS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError as exc:
        logger.warning("Failed to write decision log: %s", exc)


# ===========================================================================
# AutoManager
# ===========================================================================


class AutoManager:
    """Smart management layer that orchestrates agents via GitHub Issues.

    All GitHub operations are delegated to the injected *GitHubConnector*
    instance, keeping this class focused on decision logic.
    """

    def __init__(self, gh: GitHubConnector) -> None:
        self.gh = gh
        self.repo = REPO

    # ------------------------------------------------------------------
    # 1. Dashboard
    # ------------------------------------------------------------------

    def dashboard(self) -> str:
        """Return a formatted ANSI dashboard string showing agent status.

        Displays:
        - Each agent's current in-progress task
        - Queue depth (status:ready count) per agent
        - Last activity timestamp per agent
        - Issues currently awaiting review
        """
        lines: list[str] = []
        lines.append("")
        lines.append(
            _color(
                "  COMMAND CENTER DASHBOARD  ",
                _BOLD,
                _BG_GREEN,
                _WHITE,
            )
        )
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(_color(f"  Updated: {now_str}", _DIM))
        lines.append("")

        # --- Agent rows ---
        header = (
            f"  {'Agent':<14} {'Status':<14} {'Current Task':<50} "
            f"{'Queue':<8} {'Last Activity'}"
        )
        lines.append(_color(header, _BOLD, _CYAN))
        lines.append(_color("  " + "-" * 110, _DIM))

        for agent_name, cfg in AGENTS.items():
            try:
                in_progress = self._get_agent_issues(
                    agent_name, status="status:in-progress"
                )
                ready_issues = self._get_agent_issues(
                    agent_name, status="status:ready"
                )
                queue_depth = len(ready_issues)

                # Current task
                if in_progress:
                    issue = in_progress[0]
                    task_str = f"#{issue['number']} {issue['title'][:44]}"
                    status_str = _color("ACTIVE", _GREEN, _BOLD)
                else:
                    task_str = _color("-- idle --", _DIM)
                    status_str = _color("IDLE", _YELLOW)

                # Last activity: latest comment by the agent
                last_ts = self._get_agent_last_comment_ts(agent_name)
                activity_str = _relative_time(last_ts)

                # Queue coloring
                if queue_depth == 0:
                    queue_str = _color(str(queue_depth), _RED)
                elif queue_depth < 2:
                    queue_str = _color(str(queue_depth), _YELLOW)
                else:
                    queue_str = _color(str(queue_depth), _GREEN)

                lines.append(
                    f"  {agent_name:<14} {status_str:<24} {task_str:<50} "
                    f"{queue_str:<18} {activity_str}"
                )
            except Exception as exc:
                lines.append(
                    f"  {agent_name:<14} {_color('ERROR', _RED):<24} "
                    f"{str(exc)[:50]}"
                )

        # --- Review queue ---
        lines.append("")
        lines.append(_color("  REVIEW QUEUE", _BOLD, _MAGENTA))
        lines.append(_color("  " + "-" * 80, _DIM))

        try:
            review_items = self.review_queue()
            if review_items:
                for item in review_items:
                    prio = item.get("priority", "P?")
                    prio_color = _RED if prio == "P0" else _YELLOW
                    lines.append(
                        f"  #{item['number']:<6} "
                        f"{_color(prio, prio_color):<18} "
                        f"{item['title'][:60]}"
                    )
            else:
                lines.append(_color("  (empty)", _DIM))
        except Exception as exc:
            lines.append(f"  {_color('Error fetching review queue: ' + str(exc), _RED)}")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2. Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, dict[str, str]]:
        """Check agent health based on their last GitHub comment.

        Returns a dict keyed by agent name:
            {
                "researcher": {"last_comment": "2h ago", "status": "healthy"},
                ...
            }

        Status thresholds:
        - healthy: last comment < 4 hours ago
        - stale:   last comment 4-8 hours ago
        - dead:    last comment > 8 hours ago
        """
        result: dict[str, dict[str, str]] = {}

        for agent_name in AGENTS:
            try:
                last_ts = self._get_agent_last_comment_ts(agent_name)
                hours = _hours_since(last_ts)
                relative = _relative_time(last_ts)

                if hours > 8:
                    status = "dead"
                elif hours > 4:
                    status = "stale"
                else:
                    status = "healthy"

                result[agent_name] = {
                    "last_comment": relative,
                    "hours_ago": round(hours, 1),
                    "status": status,
                }

                if status in ("stale", "dead"):
                    logger.warning(
                        "Agent %s is %s (last comment %s)",
                        agent_name,
                        status,
                        relative,
                    )
            except Exception as exc:
                logger.error("Health check failed for %s: %s", agent_name, exc)
                result[agent_name] = {
                    "last_comment": "error",
                    "hours_ago": -1,
                    "status": "error",
                }

        return result

    # ------------------------------------------------------------------
    # 3. Queue
    # ------------------------------------------------------------------

    def queue(self, agent: str) -> list[dict[str, Any]]:
        """Get the prioritized task queue for *agent*.

        Returns a list of dicts sorted by priority (P0 first):
            [{"number": 123, "title": "...", "priority": "P0", "status": "status:ready"}, ...]
        """
        self._validate_agent(agent)

        issues = self._get_agent_issues(agent)
        items: list[dict[str, Any]] = []

        for issue in issues:
            labels = [lbl["name"] if isinstance(lbl, dict) else lbl for lbl in issue.get("labels", [])]
            items.append(
                {
                    "number": issue["number"],
                    "title": issue["title"],
                    "priority": _priority_label(labels),
                    "status": _extract_status(labels) or "unknown",
                    "_priority_rank": _extract_priority(labels),
                }
            )

        items.sort(key=lambda x: x["_priority_rank"])

        # Strip internal sort key before returning
        for item in items:
            del item["_priority_rank"]

        return items

    # ------------------------------------------------------------------
    # 4. Assign
    # ------------------------------------------------------------------

    def assign(
        self,
        agent: str,
        issue_number: int,
        message: str | None = None,
    ) -> None:
        """Assign *issue_number* to *agent*.

        Adds the agent's role label and status:ready, then posts a comment.
        """
        self._validate_agent(agent)
        cfg = AGENTS[agent]

        # Add labels
        self.gh.add_labels(
            issue_number,
            [cfg["role_label"], "status:ready"],
        )

        # Comment
        body = f"Assigned to **{agent}**."
        if message:
            body += f" {message}"
        self.gh.add_comment(issue_number, body)

        _log_decision(
            "Task Assignment",
            f"Assigned #{issue_number} to {agent}",
            message or "",
        )
        logger.info("Assigned #%d to %s", issue_number, agent)

    # ------------------------------------------------------------------
    # 5. Directive
    # ------------------------------------------------------------------

    def directive(self, agent: str, message: str) -> None:
        """Post a manager directive to *agent*.

        If the agent has a current in-progress issue, the directive is
        posted as a comment on that issue.  Otherwise a new issue is
        created to carry the directive.
        """
        self._validate_agent(agent)
        cfg = AGENTS[agent]
        directive_body = f"[MANAGER DIRECTIVE] {message}"

        in_progress = self._get_agent_issues(agent, status="status:in-progress")

        if in_progress:
            target = in_progress[0]
            self.gh.add_comment(target["number"], directive_body)
            logger.info(
                "Posted directive to %s on #%d", agent, target["number"]
            )
        else:
            # No in-progress issue -- create one.
            # Use only the role_label (avoids duplicate since label and
            # role_label are now identical).
            new_issue = self.gh.create_issue(
                title=f"[Directive] {agent}: {message[:80]}",
                body=(
                    f"## Manager Directive\n\n"
                    f"{directive_body}\n\n"
                    f"---\n"
                    f"*Auto-created by auto_manager because {agent} had no "
                    f"in-progress task.*"
                ),
                labels=[cfg["role_label"], "status:ready"],
            )
            logger.info(
                "Created directive issue #%s for %s",
                new_issue.number if hasattr(new_issue, "number") else "?",
                agent,
            )

        _log_decision(
            "Directive",
            f"Directive to {agent}: {message[:120]}",
        )

    # ------------------------------------------------------------------
    # 6. Review Queue
    # ------------------------------------------------------------------

    def review_queue(self) -> list[dict[str, Any]]:
        """List all issues with status:needs-review, with summary info.

        Returns a list of dicts:
            [{"number": N, "title": "...", "priority": "P1", "agent": "builder", "body_preview": "..."}, ...]
        """
        raw_issues = self.gh.list_issues(
            labels=["status:needs-review"],
            state="open",
        )

        items: list[dict[str, Any]] = []
        for iss in raw_issues:
            labels = iss.labels if hasattr(iss, "labels") else []
            agent = self._agent_from_labels(labels)
            body = iss.body if hasattr(iss, "body") else ""
            items.append(
                {
                    "number": iss.number if hasattr(iss, "number") else iss.get("number", 0),
                    "title": iss.title if hasattr(iss, "title") else iss.get("title", ""),
                    "priority": _priority_label(labels),
                    "agent": agent,
                    "body_preview": (body or "")[:200].replace("\n", " "),
                }
            )
        return items

    # ------------------------------------------------------------------
    # 7. Auto Review
    # ------------------------------------------------------------------

    def auto_review(self, issue_number: int) -> str:
        """Run automated quality checks on *issue_number*.

        Steps:
        1. Fetch issue details and latest comments
        2. If code files are involved, run validate-quality.sh
        3. Return "qa:passed" or "qa:failed" with reasons

        Returns a string like "qa:passed" or "qa:failed: <reasons>".
        """
        issue_data = self.gh.get_issue(issue_number)
        issue = issue_data["issue"]
        comment_objs = issue_data["comments"]

        body = issue.body or ""
        labels = issue.labels if hasattr(issue, "labels") else []

        reasons: list[str] = []

        # Check for code file references (.py, .ts, .js, .sh, etc.)
        code_pattern = re.compile(
            r"[\w/.-]+\.(?:py|ts|tsx|js|jsx|sh|go|rs|java|yaml|yml|toml|json)\b"
        )
        code_files_mentioned = code_pattern.findall(body)
        for comment in comment_objs:
            cbody = comment.body if hasattr(comment, "body") else ""
            code_files_mentioned.extend(code_pattern.findall(cbody))

        # Basic content checks
        if not body.strip():
            reasons.append("Issue body is empty")

        if not comment_objs:
            reasons.append("No comments or progress updates found")

        # If code files referenced, attempt to run validate-quality.sh
        if code_files_mentioned:
            qa_result = self._run_quality_script(code_files_mentioned)
            if qa_result:
                reasons.append(qa_result)

        # Check that at least one substantive comment exists (>50 chars)
        substantive = [
            c
            for c in comment_objs
            if len((c.body if hasattr(c, "body") else "").strip()) > 50
        ]
        if not substantive:
            reasons.append("No substantive progress comment (>50 chars)")

        if reasons:
            verdict = "qa:failed: " + "; ".join(reasons)
            self.gh.add_comment(
                issue_number,
                f"[AUTO-REVIEW] **QA FAILED**\n\n"
                + "\n".join(f"- {r}" for r in reasons),
            )
        else:
            verdict = "qa:passed"
            self.gh.add_comment(
                issue_number,
                "[AUTO-REVIEW] **QA PASSED** -- Automated checks cleared.",
            )

        _log_decision(
            "Auto Review",
            f"#{issue_number} -> {verdict}",
        )
        logger.info("Auto review #%d: %s", issue_number, verdict)
        return verdict

    # ------------------------------------------------------------------
    # 8. Replenish Queues
    # ------------------------------------------------------------------

    def replenish_queues(self, min_ready: int = 2) -> None:
        """Ensure each agent has at least *min_ready* status:ready issues.

        If an agent's ready count is below the threshold, backlog issues
        are promoted in priority order (P0 first).
        """
        for agent_name, cfg in AGENTS.items():
            try:
                ready = self._get_agent_issues(agent_name, status="status:ready")
                deficit = min_ready - len(ready)

                if deficit <= 0:
                    continue

                backlog = self._get_agent_issues(
                    agent_name, status="status:backlog"
                )

                # Sort backlog by priority
                def _sort_key(issue: dict) -> int:
                    labels = [
                        lbl["name"] if isinstance(lbl, dict) else lbl
                        for lbl in issue.get("labels", [])
                    ]
                    return _extract_priority(labels)

                backlog.sort(key=_sort_key)

                promoted = 0
                for issue in backlog[:deficit]:
                    try:
                        self.gh.transition_status(
                            issue["number"], "status:ready"
                        )
                        self.gh.add_comment(
                            issue["number"],
                            "Auto-promoted to ready queue.",
                        )
                        promoted += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to promote #%d for %s: %s",
                            issue["number"],
                            agent_name,
                            exc,
                        )

                if promoted > 0:
                    _log_decision(
                        "Queue Replenishment",
                        f"Promoted {promoted} issue(s) to ready for {agent_name}",
                        f"Ready count was {len(ready)}, minimum is {min_ready}",
                    )
                    logger.info(
                        "Replenished %s queue: promoted %d issues",
                        agent_name,
                        promoted,
                    )
            except Exception as exc:
                logger.error(
                    "Replenish failed for %s: %s", agent_name, exc
                )

    # ------------------------------------------------------------------
    # 9. Check Dependencies
    # ------------------------------------------------------------------

    def check_dependencies(self) -> None:
        """Resolve dependency chains among approved issues.

        For every issue with status:approved, parse its body for
        "Depends on #N" references.  If all dependencies are also
        approved (or closed), unblock dependent issues that are
        currently status:blocked.
        """
        approved_issues_raw = self.gh.list_issues(
            labels=["status:approved"],
            state="open",
        )
        approved_issues = [
            {"number": iss.number, "body": iss.body, "labels": iss.labels}
            if hasattr(iss, "number")
            else iss
            for iss in approved_issues_raw
        ]
        approved_numbers: set[int] = {iss["number"] for iss in approved_issues}

        # Also consider closed issues as resolved dependencies
        closed_issues_raw = self.gh.list_issues(
            state="closed",
        )
        closed_issues = [
            {"number": iss.number}
            if hasattr(iss, "number")
            else iss
            for iss in closed_issues_raw
        ]
        resolved_numbers: set[int] = approved_numbers | {
            iss["number"] for iss in closed_issues
        }

        blocked_issues_raw = self.gh.list_issues(
            labels=["status:blocked"],
            state="open",
        )
        blocked_issues = [
            {"number": iss.number, "body": iss.body, "labels": iss.labels}
            if hasattr(iss, "number")
            else iss
            for iss in blocked_issues_raw
        ]

        dep_pattern = re.compile(
            r"[Dd]epends?\s+on\s+#(\d+)", re.IGNORECASE
        )

        for issue in blocked_issues:
            body = issue.get("body", "") or ""
            deps = [int(m) for m in dep_pattern.findall(body)]

            if not deps:
                continue

            unresolved = [d for d in deps if d not in resolved_numbers]

            if not unresolved:
                # All dependencies satisfied -- unblock
                try:
                    self.gh.transition_status(
                        issue["number"], "status:ready"
                    )
                    dep_list = ", ".join(f"#{d}" for d in deps)
                    self.gh.add_comment(
                        issue["number"],
                        f"Dependency {dep_list} complete. You may proceed.",
                    )
                    _log_decision(
                        "Dependency Resolution",
                        f"Unblocked #{issue['number']} (deps: {dep_list})",
                    )
                    logger.info(
                        "Unblocked #%d -- dependencies resolved: %s",
                        issue["number"],
                        dep_list,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to unblock #%d: %s",
                        issue["number"],
                        exc,
                    )

    # ------------------------------------------------------------------
    # 10. Nudge Stale
    # ------------------------------------------------------------------

    def nudge_stale(self, hours: int = 4) -> None:
        """Nudge agents whose in-progress issues have gone silent.

        For each issue with status:in-progress, check the timestamp of
        the latest comment.  If it exceeds *hours*, post a nudge.
        """
        in_progress_raw = self.gh.list_issues(
            labels=["status:in-progress"],
            state="open",
        )
        in_progress = [
            {"number": iss.number, "updated_at": iss.updated_at}
            if hasattr(iss, "number")
            else iss
            for iss in in_progress_raw
        ]

        for issue in in_progress:
            try:
                comments = self.gh.get_comments(issue["number"])

                if comments:
                    last_comment = comments[-1]
                    last_ts = last_comment.created_at
                else:
                    # Fall back to issue updated_at
                    last_ts = issue.get("updated_at")

                elapsed = _hours_since(last_ts)

                if elapsed > hours:
                    # Avoid duplicate nudges: check if last comment is
                    # already a nudge from us
                    if comments:
                        last_body = comments[-1].body or ""
                        if "[MANAGER]" in last_body and "Status update" in last_body:
                            continue

                    self.gh.add_comment(
                        issue["number"],
                        "[MANAGER] Status update required. Are you blocked?",
                    )
                    logger.info(
                        "Nudged #%d (silent for %.1fh)",
                        issue["number"],
                        elapsed,
                    )
            except Exception as exc:
                logger.error(
                    "Failed to nudge #%d: %s", issue["number"], exc
                )

    # ------------------------------------------------------------------
    # 11. Prevent Unauthorized Closes
    # ------------------------------------------------------------------

    def prevent_unauthorized_closes(self, hours: int = 24) -> None:
        """Reopen issues that were closed by agents without manager approval.

        Scans issues closed in the last *hours* hours that carry agent role
        labels (role:builder, role:researcher, role:kimi).  For each such
        issue, queries the GitHub events API to determine who closed it.
        If the closer is not the manager, the issue is reopened with a
        warning comment and the action is logged to DECISIONS.md.

        Parameters
        ----------
        hours:
            Look-back window in hours (default 24).
        """
        agent_role_labels = {
            cfg["role_label"] for cfg in AGENTS.values()
        }

        try:
            recently_closed = self.gh.list_recently_closed(hours=hours)
        except Exception as exc:
            logger.error("Failed to fetch recently closed issues: %s", exc)
            return

        for issue in recently_closed:
            issue_labels = set(issue.get("labels", []))

            # Only care about issues assigned to agents
            matching_roles = issue_labels & agent_role_labels
            if not matching_roles:
                continue

            issue_number = issue["number"]

            # Determine who actually closed the issue via events API
            try:
                closer = self.gh.get_issue_closer(issue_number)
            except Exception as exc:
                logger.error(
                    "Failed to determine closer for #%d: %s",
                    issue_number,
                    exc,
                )
                continue

            # If closer cannot be determined, skip (fail-open)
            if closer is None:
                logger.warning(
                    "Could not determine who closed #%d; skipping.",
                    issue_number,
                )
                continue

            # If the manager closed it, that is authorized
            if closer == MANAGER_GITHUB_USER:
                continue

            # Unauthorized close -- reopen and warn
            try:
                self.gh.reopen_issue(issue_number)
                self.gh.add_comment(
                    issue_number,
                    "Issues can only be closed by the Manager. "
                    "Reopened automatically.",
                )
                _log_decision(
                    "Unauthorized Close Prevention",
                    f"Reopened #{issue_number} (closed by {closer})",
                    f"Only {MANAGER_GITHUB_USER} may close agent issues. "
                    f"Roles: {', '.join(matching_roles)}",
                )
                logger.warning(
                    "Reopened #%d -- unauthorized close by %s",
                    issue_number,
                    closer,
                )
            except Exception as exc:
                logger.error(
                    "Failed to reopen #%d after unauthorized close: %s",
                    issue_number,
                    exc,
                )

    # ------------------------------------------------------------------
    # 12. Sync GitHub to INBOX
    # ------------------------------------------------------------------

    def sync_github_to_inbox(self) -> None:
        """Sync top GitHub tasks to each agent's INBOX.md.

        For each agent, gets their ``status:ready`` and ``status:in-progress``
        issues from GitHub, then writes/updates the agent's INBOX.md file at
        ``~/.claude/command-center/{agent}/INBOX.md``.

        This bridges the GitHub-based agent-mgr with the file-based
        command center that agents actually read.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        for agent_name, cfg in AGENTS.items():
            try:
                in_progress = self._get_agent_issues(
                    agent_name, status="status:in-progress"
                )
                ready_issues = self._get_agent_issues(
                    agent_name, status="status:ready"
                )

                # Build INBOX.md content
                lines: list[str] = []
                lines.append(f"# {agent_name.capitalize()} -- Task Inbox")
                lines.append(f"**Updated:** {now_str}")
                lines.append("")
                lines.append("---")
                lines.append("")

                # Current assignment section
                if in_progress:
                    lines.append("## CURRENT ASSIGNMENT")
                    lines.append("")
                    lines.append("| # | Task | GitHub Issue | Status |")
                    lines.append("|---|------|-------------|--------|")
                    for issue in in_progress:
                        number = issue["number"]
                        title = issue["title"][:60]
                        lines.append(
                            f"| {number} | {title} | #{number} | in-progress |"
                        )
                    lines.append("")

                # Ready queue section
                if ready_issues:
                    lines.append("## READY QUEUE")
                    lines.append("")
                    lines.append("| # | Task | GitHub Issue | Priority |")
                    lines.append("|---|------|-------------|----------|")
                    for issue in ready_issues:
                        number = issue["number"]
                        title = issue["title"][:60]
                        labels = [
                            lbl["name"] if isinstance(lbl, dict) else lbl
                            for lbl in issue.get("labels", [])
                        ]
                        priority = _priority_label(labels)
                        lines.append(
                            f"| {number} | {title} | #{number} | {priority} |"
                        )
                    lines.append("")

                if not in_progress and not ready_issues:
                    lines.append("## NO TASKS")
                    lines.append("")
                    lines.append("No tasks currently assigned. Awaiting new assignments.")
                    lines.append("")

                # GitHub-first protocol section
                lines.append("---")
                lines.append("")
                lines.append("## GITHUB-FIRST PROTOCOL")
                lines.append("")
                lines.append(
                    f"1. Check issues: `gh issue list --repo {REPO} "
                    f"--label {cfg['role_label']} --state open`"
                )
                lines.append("2. Comment on issue when starting")
                lines.append("3. Reference issue # in commits")
                lines.append("4. Comment with results when done")
                lines.append("")

                # Write the INBOX.md file
                inbox_path = COMMAND_CENTER / agent_name / "INBOX.md"
                inbox_path.parent.mkdir(parents=True, exist_ok=True)
                inbox_path.write_text("\n".join(lines), encoding="utf-8")

                logger.info(
                    "Synced INBOX.md for %s (%d in-progress, %d ready)",
                    agent_name,
                    len(in_progress),
                    len(ready_issues),
                )

            except Exception as exc:
                logger.error(
                    "Failed to sync INBOX.md for %s: %s", agent_name, exc
                )

    # ------------------------------------------------------------------
    # 13. Run Auto Loop
    # ------------------------------------------------------------------

    def run_auto_loop(self, interval: int = 300) -> None:
        """Main autonomous management loop.

        Runs indefinitely, executing all management tasks on each cycle:
        1. Health check (log alerts for stale/dead agents)
        2. Process controller check (Hetzner connectivity and Kimi sessions)
        3. Sync GitHub tasks to INBOX.md files
        4. Prevent unauthorized closes
        5. Replenish queues
        6. Check dependencies
        7. Nudge stale issues
        8. Auto-review all needs-review items

        Sleeps *interval* seconds between cycles.  Handles rate-limit and
        network errors gracefully by logging and continuing.

        KeyboardInterrupt is caught both during work steps and during
        sleep, ensuring a clean shutdown message in all cases.
        """
        cycle = 0
        logger.info(
            "Starting auto-management loop (interval=%ds)", interval
        )

        while True:
            try:
                cycle += 1
                cycle_start = time.monotonic()
                logger.info("--- Auto-loop cycle #%d ---", cycle)

                # 1. Health check
                try:
                    health = self.health_check()
                    for agent_name, info in health.items():
                        if info["status"] in ("stale", "dead"):
                            logger.warning(
                                "ALERT: %s is %s (last: %s)",
                                agent_name,
                                info["status"],
                                info["last_comment"],
                            )
                except Exception as exc:
                    logger.error("Health check cycle error: %s", exc)

                # 2. Process controller check
                try:
                    pc = ProcessController()
                    reachable = pc.check_hetzner_connectivity()
                    if not reachable:
                        logger.warning(
                            "ALERT: Hetzner server is unreachable"
                        )
                    else:
                        kimi_count = pc.count_kimi_sessions()
                        if kimi_count == 0:
                            logger.warning(
                                "ALERT: No active Kimi sessions on Hetzner"
                            )
                        else:
                            logger.info(
                                "Hetzner OK: %d active Kimi session(s)",
                                kimi_count,
                            )
                except Exception as exc:
                    logger.error("Process controller cycle error: %s", exc)

                # 3. Sync GitHub to INBOX.md
                try:
                    self.sync_github_to_inbox()
                except Exception as exc:
                    logger.error("INBOX sync cycle error: %s", exc)

                # 4. Prevent unauthorized closes
                try:
                    self.prevent_unauthorized_closes()
                except Exception as exc:
                    logger.error("Close prevention cycle error: %s", exc)

                # 5. Replenish queues
                try:
                    self.replenish_queues()
                except Exception as exc:
                    logger.error("Replenish cycle error: %s", exc)

                # 6. Check dependencies
                try:
                    self.check_dependencies()
                except Exception as exc:
                    logger.error("Dependency check cycle error: %s", exc)

                # 7. Nudge stale
                try:
                    self.nudge_stale()
                except Exception as exc:
                    logger.error("Nudge cycle error: %s", exc)

                # 8. Auto-review needs-review items
                try:
                    review_items = self.review_queue()
                    for item in review_items:
                        try:
                            self.auto_review(item["number"])
                        except Exception as exc:
                            logger.error(
                                "Auto-review failed for #%d: %s",
                                item["number"],
                                exc,
                            )
                except Exception as exc:
                    logger.error("Review queue cycle error: %s", exc)

                elapsed = time.monotonic() - cycle_start
                logger.info(
                    "Cycle #%d completed in %.1fs. Sleeping %ds...",
                    cycle,
                    elapsed,
                    interval,
                )

                try:
                    time.sleep(interval)
                except KeyboardInterrupt:
                    logger.info("Auto-loop interrupted by user. Exiting.")
                    break

            except KeyboardInterrupt:
                logger.info("Auto-loop interrupted during cycle #%d. Exiting cleanly.", cycle)
                break

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _validate_agent(self, agent: str) -> None:
        """Raise ValueError if *agent* is not a known agent name."""
        if agent not in AGENTS:
            raise ValueError(
                f"Unknown agent '{agent}'. Valid agents: "
                + ", ".join(AGENTS.keys())
            )

    def _get_agent_issues(
        self,
        agent: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch open issues for *agent*, optionally filtered by *status*.

        Uses the agent's primary label (e.g. 'role:researcher') and
        an optional status label for filtering.
        """
        cfg = AGENTS[agent]
        labels = [cfg["label"]]
        if status:
            labels.append(status)
        issues = self.gh.list_issues(labels=labels, state="open")
        # Convert Issue dataclass objects to dicts for internal use
        return [
            {
                "number": iss.number,
                "title": iss.title,
                "body": iss.body,
                "labels": [{"name": lb} for lb in iss.labels],
            }
            if hasattr(iss, "number")
            else iss
            for iss in issues
        ]

    def _get_agent_last_comment_ts(self, agent: str) -> str | None:
        """Get the ISO timestamp of the agent's most recent comment.

        Looks at the agent's in-progress issue first, then falls back to
        any issue carrying the agent's label.
        """
        # Try in-progress first
        in_progress = self._get_agent_issues(
            agent, status="status:in-progress"
        )
        issues_to_check = in_progress or self._get_agent_issues(agent)

        latest_ts: str | None = None

        for issue in issues_to_check[:5]:  # Cap API calls
            try:
                comments = self.gh.get_comments(issue["number"])
                if comments:
                    ts = comments[-1].created_at
                    if ts and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts
            except Exception:
                continue

        return latest_ts

    def _agent_from_labels(self, labels: list[str]) -> str:
        """Infer which agent owns an issue based on its labels."""
        for agent_name, cfg in AGENTS.items():
            if cfg["role_label"] in labels or cfg["label"] in labels:
                return agent_name
        return "unknown"

    def _run_quality_script(self, files: list[str]) -> str | None:
        """Run validate-quality.sh if it exists.  Returns failure reason or None."""
        script_path = VALIDATE_SCRIPT
        if not script_path.exists():
            return None

        # SEC-1 fix: Sanitize file paths — reject traversal and absolute paths
        safe_files: list[str] = []
        for f in files[:10]:
            # Reject paths containing ".." or starting with "/"
            if ".." in f or f.startswith("/"):
                logger.warning("Rejected unsafe file path: %s", f)
                continue
            # Only allow alphanumeric, hyphens, underscores, dots, and forward slashes
            if not re.match(r'^[\w./-]+$', f):
                logger.warning("Rejected file path with special chars: %s", f)
                continue
            safe_files.append(f)

        if not safe_files:
            return None

        try:
            result = subprocess.run(
                [str(script_path)] + safe_files,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.home(),
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[:300]
                stdout = result.stdout.strip()[:300]
                return f"validate-quality.sh failed (rc={result.returncode}): {stderr or stdout}"
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return "validate-quality.sh timed out (>60s)"
        except OSError as exc:
            logger.warning("Quality script error: %s", exc)
            return None

        return None
