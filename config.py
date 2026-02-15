"""
Agent Management System — Configuration.

Defines agent roles, GitHub labels, repo settings, and management parameters.

Author: Ahmed Adel Bakr Alderai
"""

from __future__ import annotations

from pathlib import Path

# GitHub repository
REPO = "Ahmed-AdelB/ummro"

# Manager GitHub username (only this user may close agent issues)
MANAGER_GITHUB_USER = "Ahmed-AdelB"

# Agent definitions
#
# "label" is the primary label used to filter issues belonging to this agent.
# It MUST match an actual label that exists in the repository.  The
# labels-setup command creates "role:researcher", "role:builder", "role:kimi"
# via ALL_LABELS below, so we use those as the filter labels too.
#
# "role_label" is kept as a semantic alias for assignment operations
# (currently identical to "label").
AGENTS: dict[str, dict] = {
    "researcher": {
        "label": "role:researcher",
        "role_label": "role:researcher",
        "description": "Deep research, web searches, paper analysis, competition analysis",
        "tools": ["Claude Code", "Gemini CLI", "NVIDIA NIM", "WebSearch", "Kimi CLI"],
    },
    "builder": {
        "label": "role:builder",
        "role_label": "role:builder",
        "description": "Implementation, coding, testing, PR creation",
        "tools": ["Claude Code", "Gemini CLI", "NVIDIA NIM", "GitHub CLI"],
    },
    "kimi": {
        "label": "role:kimi",
        "role_label": "role:kimi",
        "description": "Strategic research via Kimi CLI ($199 Moonshot plan)",
        "tools": ["Kimi CLI", "NVIDIA NIM"],
    },
}

# Priority labels (highest first)
PRIORITY_LABELS = ["P0", "P1", "P2", "P3"]

# Status labels (lifecycle order)
STATUS_LABELS = [
    "status:backlog",
    "status:ready",
    "status:in-progress",
    "status:blocked",
    "status:needs-review",
    "status:approved",
    "status:rejected",
]

# Role labels
ROLE_LABELS = [
    ("role:builder", "5319E7", "Assigned to Builder agent"),
    ("role:researcher", "0ABFBC", "Assigned to Researcher agent"),
    ("role:kimi", "00BCD4", "Assigned to Kimi agent"),
]

# All standard labels to create
ALL_LABELS = [
    ("status:backlog", "E4E669", "Task planned but not active"),
    ("status:ready", "0E8A16", "Ready for agent to pick up"),
    ("status:in-progress", "1D76DB", "Agent actively working"),
    ("status:blocked", "D93F0B", "Agent blocked, needs help"),
    ("status:needs-review", "FBCA04", "Work done, review needed"),
    ("status:approved", "0E8A16", "Approved by Manager"),
    ("status:rejected", "B60205", "Rejected, needs revision"),
    ("role:builder", "5319E7", "Assigned to Builder agent"),
    ("role:researcher", "0ABFBC", "Assigned to Researcher agent"),
    ("role:kimi", "00BCD4", "Assigned to Kimi agent"),
    ("qa:passed", "2EA44F", "Automated QA passed"),
    ("qa:failed", "CB2431", "Automated QA failed"),
]

# Paths
COMMAND_CENTER = Path.home() / ".claude" / "command-center"
DECISIONS_FILE = COMMAND_CENTER / "DECISIONS.md"
VALIDATE_SCRIPT = Path.home() / "projects" / "PM" / "scripts" / "validate-quality.sh"

# Auto-management parameters
STALE_HOURS = 4       # Hours before agent is flagged as stale
DEAD_HOURS = 8        # Hours before agent is flagged as dead
MIN_READY_TASKS = 2   # Minimum ready tasks per agent queue
AUTO_LOOP_INTERVAL = 300  # Seconds between auto-management cycles (5 min)

# ANSI colors for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
