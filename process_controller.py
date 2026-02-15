"""Process Controller — controls physical agent processes via SSH or local shell.

Provides methods to check, run, and manage Kimi CLI sessions on a remote
Hetzner server over SSH.  All SSH commands use ``subprocess.run`` with
timeouts and ``shlex.quote`` for safe parameter handling.

No external dependencies are required — only the Python 3.11+ standard library.

Author: Ahmed Adel Bakr Alderai
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProcessController
# ---------------------------------------------------------------------------


class ProcessController:
    """Controls physical agent processes via SSH or local shell.

    Parameters
    ----------
    hetzner_host:
        SSH host alias or address for the Hetzner server (must be configured
        in ``~/.ssh/config`` or reachable directly).
    timeout:
        Default timeout in seconds for SSH commands.
    """

    def __init__(
        self,
        hetzner_host: str = "hetzner",
        *,
        timeout: int = 30,
    ) -> None:
        self.hetzner_host = hetzner_host
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ssh_run(
        self,
        remote_cmd: str,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess | None:
        """Execute a command on Hetzner via SSH.

        Uses list-mode ``subprocess.run`` (no shell) for safety.  The
        *remote_cmd* is passed as a single string argument to ``ssh``,
        which is the standard way to run a remote command.

        Returns the ``CompletedProcess`` on success, or ``None`` if the
        command fails or times out.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        full_args = ["ssh", self.hetzner_host, remote_cmd]
        cmd_str = " ".join(shlex.quote(a) for a in full_args)
        logger.debug("ssh exec: %s", cmd_str)

        try:
            result = subprocess.run(
                full_args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return result
        except subprocess.TimeoutExpired:
            logger.warning(
                "SSH command timed out after %ds: %s",
                effective_timeout,
                cmd_str,
            )
            return None
        except FileNotFoundError:
            logger.error("ssh binary not found on PATH")
            return None
        except OSError as exc:
            logger.error("SSH command failed with OS error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def check_hetzner_connectivity(self) -> bool:
        """Quick SSH check -- returns True if Hetzner is reachable.

        Runs ``ssh hetzner echo ok`` with a short timeout.
        """
        logger.info("Checking Hetzner connectivity...")
        result = self._ssh_run("echo ok", timeout=10)

        if result is None:
            logger.warning("Hetzner is unreachable (SSH timed out or failed)")
            return False

        if result.returncode != 0:
            logger.warning(
                "Hetzner SSH check failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return False

        if "ok" in result.stdout:
            logger.info("Hetzner is reachable")
            return True

        logger.warning(
            "Hetzner SSH check returned unexpected output: %s",
            result.stdout.strip()[:200],
        )
        return False

    # ------------------------------------------------------------------
    # Kimi process management
    # ------------------------------------------------------------------

    def check_kimi_processes(self) -> list[dict[str, Any]]:
        """SSH to Hetzner, find running kimi processes.

        Returns a list of dicts with keys ``pid``, ``cmd``, and ``uptime``
        parsed from ``ps aux`` output.  Returns an empty list on failure.
        """
        logger.info("Checking kimi processes on Hetzner...")
        result = self._ssh_run("ps aux | grep '[k]imi'")

        if result is None:
            return []

        if result.returncode != 0:
            # grep returns rc=1 when no matches found -- that is normal
            if result.returncode == 1 and not result.stderr.strip():
                logger.info("No kimi processes found on Hetzner")
                return []
            logger.warning(
                "Failed to check kimi processes (rc=%d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return []

        processes: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            processes.append({
                "pid": int(parts[1]) if parts[1].isdigit() else 0,
                "cmd": parts[10],
                "uptime": parts[9],
            })

        logger.info("Found %d kimi process(es) on Hetzner", len(processes))
        return processes

    def run_kimi_task(
        self,
        prompt: str,
        model: str = "kimi-k2",
        working_dir: str = "/home/aadel/11/",
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Run a Kimi CLI task on Hetzner and return the result.

        Uses ``ssh hetzner "kimi --yolo -m MODEL -w WORKDIR -p 'PROMPT'"``
        with single quotes in the prompt safely escaped via ``shlex.quote``.

        Parameters
        ----------
        prompt:
            The task prompt to send to Kimi CLI.
        model:
            Kimi model name (default ``"kimi-k2"``).
        working_dir:
            Remote working directory.
        timeout:
            Maximum seconds to wait for the task to complete.

        Returns
        -------
        dict[str, Any]
            A dict with keys ``success`` (bool), ``stdout`` (str),
            and ``stderr`` (str).
        """
        safe_prompt = shlex.quote(prompt)
        safe_model = shlex.quote(model)
        safe_dir = shlex.quote(working_dir)

        remote_cmd = (
            f"kimi --yolo -m {safe_model} -w {safe_dir} -p {safe_prompt}"
        )
        logger.info(
            "Running kimi task on Hetzner (model=%s, timeout=%ds)",
            model,
            timeout,
        )

        result = self._ssh_run(remote_cmd, timeout=timeout)

        if result is None:
            return {"success": False, "stdout": "", "stderr": "SSH command failed or timed out"}

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def kill_stale_kimi(self, pid: int) -> bool:
        """Kill a specific kimi process on Hetzner by PID.

        Parameters
        ----------
        pid:
            The process ID to kill on the remote server.

        Returns
        -------
        bool
            True if the kill command succeeded, False otherwise.
        """
        if not isinstance(pid, int) or pid <= 0 or pid > 4194304:
            logger.error("Invalid PID: %s (must be integer between 1-4194304)", pid)
            return False

        logger.info("Killing kimi process PID=%d on Hetzner", pid)
        result = self._ssh_run(f"kill {pid}")

        if result is None:
            return False

        if result.returncode != 0:
            logger.warning(
                "Failed to kill PID %d (rc=%d): %s",
                pid,
                result.returncode,
                result.stderr.strip()[:200],
            )
            return False

        logger.info("Successfully killed kimi PID=%d", pid)
        return True

    def count_kimi_sessions(self) -> int:
        """Count active kimi sessions on Hetzner.

        Returns the number of running kimi processes, or 0 on failure.
        """
        processes = self.check_kimi_processes()
        return len(processes)

    # ------------------------------------------------------------------
    # Resource monitoring
    # ------------------------------------------------------------------

    def check_hetzner_resources(self) -> dict[str, Any] | None:
        """Check RAM/CPU usage on Hetzner.

        Runs ``free -g`` and ``uptime`` on the remote server and parses
        the output.

        Returns
        -------
        dict[str, Any] | None
            A dict with keys ``ram_total_gb``, ``ram_used_gb``,
            ``ram_pct``, ``cpu_pct``, and ``load_avg``.  Returns
            ``None`` if the command fails.
        """
        logger.info("Checking Hetzner resource usage...")
        result = self._ssh_run("free -g && uptime")

        if result is None:
            return None

        if result.returncode != 0:
            logger.warning(
                "Failed to check Hetzner resources (rc=%d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
            return None

        output = result.stdout

        # Parse free -g output
        ram_total_gb: float = 0.0
        ram_used_gb: float = 0.0
        ram_pct: float = 0.0

        for line in output.splitlines():
            if line.lower().startswith("mem:"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ram_total_gb = float(parts[1])
                        ram_used_gb = float(parts[2])
                        if ram_total_gb > 0:
                            ram_pct = (ram_used_gb / ram_total_gb) * 100.0
                    except (ValueError, IndexError):
                        logger.warning("Could not parse free output: %s", line)

        # Parse uptime output for load average
        load_avg: str = ""
        cpu_pct: float = 0.0

        for line in output.splitlines():
            if "load average" in line.lower():
                match = re.search(
                    r"load average:\s*([\d.]+)", line
                )
                if match:
                    load_avg = match.group(1)
                    try:
                        cpu_pct = float(load_avg) * 100.0
                    except ValueError:
                        pass

        resource_info = {
            "ram_total_gb": round(ram_total_gb, 1),
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_pct": round(ram_pct, 1),
            "cpu_pct": round(cpu_pct, 1),
            "load_avg": load_avg,
        }
        logger.info(
            "Hetzner resources: RAM %.1f/%.1fGB (%.0f%%), load %s",
            ram_used_gb,
            ram_total_gb,
            ram_pct,
            load_avg,
        )
        return resource_info
