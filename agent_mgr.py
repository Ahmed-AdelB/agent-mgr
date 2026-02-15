#!/usr/bin/env python3
"""
Agent Management System — CLI Entry Point

Provides a unified CLI for managing GitHub-based AI agents:
status, health, queue, review, assign, directive, auto, labels-setup.

Author: Ahmed Adel Bakr Alderai
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from config import (
    AGENTS as AGENTS_CONFIG,
    ALL_LABELS,
    REPO,
    AUTO_LOOP_INTERVAL,
)
from github_connector import GitHubConnector
from auto_manager import AutoManager

# ── Derived configuration ────────────────────────────────────────────────────

DEFAULT_REPO: str = REPO

# Agent names list (for CLI choices)
AGENTS: list[str] = list(AGENTS_CONFIG.keys())

LABELS: list[tuple[str, str, str]] = ALL_LABELS

AUTO_LOOP_INTERVAL_SECONDS: int = AUTO_LOOP_INTERVAL

# ── ANSI colour helpers ──────────────────────────────────────────────────────

_USE_COLOR: bool = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence when stdout is a TTY."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return _c("32", text)


def red(text: str) -> str:
    return _c("31", text)


def yellow(text: str) -> str:
    return _c("33", text)


def cyan(text: str) -> str:
    return _c("36", text)


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


# ── Formatting utilities ─────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _hours_ago(dt: datetime | None) -> str:
    """Return a human-readable '2h ago' string from a datetime."""
    if dt is None:
        return "never"
    delta = datetime.now(timezone.utc) - dt
    total_minutes = int(delta.total_seconds() / 60)
    if total_minutes < 1:
        return "just now"
    if total_minutes < 60:
        return f"{total_minutes}m ago"
    hours = total_minutes // 60
    return f"{hours}h ago"


def _health_badge(healthy: bool) -> str:
    if healthy:
        return green("[HEALTHY]")
    return red("[STALE!] ")


def _priority_sort_key(issue: dict[str, Any]) -> int:
    """Sort issues so P0 comes first, then P1, P2, etc."""
    labels: list[str] = [lbl.get("name", "") for lbl in issue.get("labels", [])]
    for lbl in labels:
        if lbl.startswith("P") and len(lbl) == 2 and lbl[1].isdigit():
            return int(lbl[1])
    return 9  # no priority label = lowest


def _agent_display_name(agent: str) -> str:
    names: dict[str, str] = {
        "researcher": "Researcher",
        "builder": "Builder",
        "kimi": "Kimi",
    }
    return names.get(agent, agent)


def _separator() -> str:
    return bold("=" * 47)


# ── Command implementations ──────────────────────────────────────────────────

def cmd_status(gh: GitHubConnector) -> None:
    """Full dashboard: agents, queues, review queue."""
    print()
    print(_separator())
    print(bold("  AGENT MANAGEMENT SYSTEM -- Dashboard"))
    print(f"  Repo: {gh.repo}")
    print(f"  Time: {_now_iso()}")
    print(_separator())

    # ── Agents section ────────────────────────────────────────────────────
    print()
    print(bold("AGENTS:"))
    for agent in AGENTS:
        health = gh.agent_health(agent)
        last_dt: datetime | None = health.get("last_activity")
        healthy: bool = health.get("healthy", False)
        current_issue: dict[str, Any] | None = health.get("current_issue")

        working_on = ""
        if current_issue:
            number = current_issue.get("number", "?")
            title = current_issue.get("title", "")
            working_on = f"Working on: #{number} {title}"

        badge = _health_badge(healthy)
        last_str = _hours_ago(last_dt)
        name = _agent_display_name(agent)

        print(f"  {name:<12}{badge}  {working_on}  {dim(f'(last: {last_str})')}")

    # ── Queues section ────────────────────────────────────────────────────
    print()
    print(bold("QUEUES:"))
    for agent in AGENTS:
        queue = gh.agent_queue(agent)
        ready: int = queue.get("ready", 0)
        in_progress: int = queue.get("in_progress", 0)
        total: int = queue.get("total", 0)
        name = _agent_display_name(agent)
        warning = ""
        if ready == 0 and in_progress <= 1:
            warning = yellow("  !! NEEDS TASKS")
        print(
            f"  {name + ':':<13}{ready} ready, "
            f"{in_progress} in-progress, {total} total{warning}"
        )

    # ── Needs review section ──────────────────────────────────────────────
    print()
    review_issues = gh.needs_review_issues()
    print(bold(f"NEEDS REVIEW: {len(review_issues)} issues"))
    for issue in review_issues:
        number = issue.get("number", "?")
        title = issue.get("title", "")
        role = _extract_role(issue)
        role_str = f"  ({role})" if role else ""
        print(f"  #{number:<5} {title}{dim(role_str)}")
    print()


def cmd_health(gh: GitHubConnector) -> None:
    """Agent health check — last activity and flags."""
    print()
    print(bold("Agent Health Check"))
    print(bold("-" * 40))
    for agent in AGENTS:
        health = gh.agent_health(agent)
        last_dt: datetime | None = health.get("last_activity")
        healthy: bool = health.get("healthy", False)
        current_issue: dict[str, Any] | None = health.get("current_issue")

        name = _agent_display_name(agent)
        badge = _health_badge(healthy)
        last_str = _hours_ago(last_dt)

        print(f"  {name:<12} {badge}  Last activity: {last_str}")

        if current_issue:
            number = current_issue.get("number", "?")
            title = current_issue.get("title", "")
            print(f"{'':>15}Current issue: #{number} {title}")

        if not healthy:
            stale_hours = health.get("stale_hours", 0)
            print(
                f"{'':>15}"
                f"{red(f'WARNING: No activity for {stale_hours}h — may need nudge')}"
            )
    print()


def cmd_queue(gh: GitHubConnector, agent: str | None) -> None:
    """Task queue for one or all agents, ordered by priority."""
    agents_to_show = [agent] if agent else AGENTS
    print()
    for ag in agents_to_show:
        name = _agent_display_name(ag)
        print(bold(f"Queue: {name}"))
        print(bold("-" * 40))

        issues = gh.agent_queue_issues(ag)
        issues.sort(key=_priority_sort_key)

        if not issues:
            print(dim("  (empty)"))
        for issue in issues:
            number = issue.get("number", "?")
            title = issue.get("title", "")
            status = _extract_status(issue)
            priority = _extract_priority(issue)

            status_colored = _colorize_status(status)
            pri_str = f"[{priority}]" if priority else ""

            print(f"  #{number:<5} {pri_str:<5} {status_colored:<22} {title}")
        print()


def cmd_review(gh: GitHubConnector) -> None:
    """List all needs-review issues."""
    print()
    issues = gh.needs_review_issues()
    print(bold(f"Issues Needing Review ({len(issues)})"))
    print(bold("-" * 50))
    if not issues:
        print(dim("  No issues currently need review."))
    for issue in issues:
        number = issue.get("number", "?")
        title = issue.get("title", "")
        role = _extract_role(issue)
        role_str = f"  [{role}]" if role else ""
        print(f"  #{number:<5} {title}{dim(role_str)}")
    print()


def cmd_assign(gh: GitHubConnector, agent: str, issue_number: int) -> None:
    """Assign an issue to an agent: add role label, status label, and comment."""
    role_label = _role_label_for_agent(agent)
    if role_label is None:
        print(red(f"Error: Unknown agent '{agent}'. Known agents: {', '.join(AGENTS)}"))
        sys.exit(1)

    print(f"Assigning #{issue_number} to {_agent_display_name(agent)}...")
    gh.add_labels(issue_number, [role_label, "status:ready"])
    gh.post_comment(
        issue_number,
        f"@{agent} — You have been assigned this issue. "
        f"Please begin work and update status to `status:in-progress`.",
    )
    print(green(f"Done. #{issue_number} assigned to {agent} with labels [{role_label}, status:ready]."))


def cmd_directive(gh: GitHubConnector, agent: str, message: str) -> None:
    """Post a manager directive to an agent's current issue."""
    health = gh.agent_health(agent)
    current_issue: dict[str, Any] | None = health.get("current_issue")
    if current_issue is None:
        print(red(f"Error: {agent} has no current in-progress issue."))
        sys.exit(1)

    issue_number: int = current_issue["number"]
    directive_body = (
        f"**Manager Directive** to @{agent}:\n\n"
        f"> {message}\n\n"
        f"Please acknowledge and act on this directive."
    )
    gh.post_comment(issue_number, directive_body)
    print(green(f"Directive posted to #{issue_number} for {_agent_display_name(agent)}."))


def cmd_auto(gh: GitHubConnector) -> None:
    """Run the autonomous management loop (5-min interval).

    Delegates to AutoManager.run_auto_loop() which has full per-step
    error handling, rate-limit awareness, and clean shutdown.
    """
    manager = AutoManager(gh)

    print(f"[{_now_ts()}] Starting autonomous management loop...")
    print(f"[{_now_ts()}] Repo: {gh.repo}")
    print(f"[{_now_ts()}] Interval: {AUTO_LOOP_INTERVAL_SECONDS}s")
    print()

    manager.run_auto_loop(interval=AUTO_LOOP_INTERVAL_SECONDS)


def cmd_labels_setup(gh: GitHubConnector) -> None:
    """Create all standard labels in the repository."""
    print(f"Setting up labels in {gh.repo}...")
    print()
    for name, color, description in LABELS:
        try:
            gh.create_or_update_label(name, color, description)
            print(f"  {green('OK')}  {name}  ({dim(description)})")
        except Exception as exc:
            print(f"  {red('FAIL')}  {name}  — {exc}")
    print()
    print(green("Label setup complete."))


# ── Label extraction helpers ─────────────────────────────────────────────────

def _extract_role(issue: dict[str, Any]) -> str:
    """Return the agent name from role:* labels, or empty string."""
    role_map: dict[str, str] = {
        "role:builder": "builder",
        "role:researcher": "researcher",
        "role:kimi": "kimi",
    }
    for lbl in issue.get("labels", []):
        name = lbl.get("name", "")
        if name in role_map:
            return role_map[name]
    return ""


def _extract_status(issue: dict[str, Any]) -> str:
    """Return the status:* label value, or 'unknown'."""
    for lbl in issue.get("labels", []):
        name: str = lbl.get("name", "")
        if name.startswith("status:"):
            return name.split(":", 1)[1]
    return "unknown"


def _extract_priority(issue: dict[str, Any]) -> str:
    """Return the priority label (P0, P1, ...), or empty string."""
    for lbl in issue.get("labels", []):
        name: str = lbl.get("name", "")
        if len(name) == 2 and name[0] == "P" and name[1].isdigit():
            return name
    return ""


def _colorize_status(status: str) -> str:
    colors: dict[str, str] = {
        "ready": green("ready"),
        "in-progress": cyan("in-progress"),
        "blocked": red("blocked"),
        "needs-review": yellow("needs-review"),
        "backlog": dim("backlog"),
        "approved": green("approved"),
        "rejected": red("rejected"),
    }
    return colors.get(status, status)


def _role_label_for_agent(agent: str) -> str | None:
    mapping: dict[str, str] = {
        "researcher": "role:researcher",
        "builder": "role:builder",
        "kimi": "role:kimi",
    }
    return mapping.get(agent)


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_mgr",
        description="Agent Management System — manage GitHub-based AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 agent_mgr.py status\n"
            "  python3 agent_mgr.py health\n"
            "  python3 agent_mgr.py queue researcher\n"
            "  python3 agent_mgr.py assign builder 489\n"
            '  python3 agent_mgr.py directive kimi "Prioritize Arabic papers"\n'
            "  python3 agent_mgr.py auto\n"
            "  python3 agent_mgr.py labels-setup\n"
        ),
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub repository (default: {DEFAULT_REPO})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Full dashboard: agents, queues, review")

    # health
    subparsers.add_parser("health", help="Agent health check (last activity time)")

    # queue
    queue_parser = subparsers.add_parser("queue", help="Task queue (all or specific agent)")
    queue_parser.add_argument(
        "agent",
        nargs="?",
        default=None,
        choices=AGENTS,
        help="Agent name (omit for all agents)",
    )

    # review
    subparsers.add_parser("review", help="List issues needing review")

    # assign
    assign_parser = subparsers.add_parser("assign", help="Assign issue to agent")
    assign_parser.add_argument("agent", choices=AGENTS, help="Agent to assign to")
    assign_parser.add_argument("issue", type=int, help="Issue number (e.g. 489)")

    # directive
    directive_parser = subparsers.add_parser("directive", help="Post directive to agent")
    directive_parser.add_argument("agent", choices=AGENTS, help="Target agent")
    directive_parser.add_argument("message", help="Directive message text")

    # auto
    subparsers.add_parser("auto", help="Run autonomous management loop (5-min interval)")

    # labels-setup
    subparsers.add_parser("labels-setup", help="Create standard labels in repo")

    return parser


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    gh = GitHubConnector(repo=args.repo)

    dispatch: dict[str, Any] = {
        "status": lambda: cmd_status(gh),
        "health": lambda: cmd_health(gh),
        "queue": lambda: cmd_queue(gh, args.agent),
        "review": lambda: cmd_review(gh),
        "assign": lambda: cmd_assign(gh, args.agent, args.issue),
        "directive": lambda: cmd_directive(gh, args.agent, args.message),
        "auto": lambda: cmd_auto(gh),
        "labels-setup": lambda: cmd_labels_setup(gh),
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler()
    except KeyboardInterrupt:
        print()
        print(yellow("Interrupted."))
        sys.exit(130)
    except Exception as exc:
        print(red(f"Error: {exc}"))
        sys.exit(1)


if __name__ == "__main__":
    main()
