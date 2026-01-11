"""
Daemon lifecycle management: start, stop, query.

Uses file locking to prevent race conditions when multiple processes
try to start the daemon simultaneously.
"""

import hashlib
import json
import logging
import os
import socket
import sys
import time
import platform
import tempfile
from pathlib import Path

# Conditional imports
if os.name == "nt":
    import msvcrt
else:
    import fcntl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import TLDRDaemon

logger = logging.getLogger(__name__)


def _get_lock_path(project: Path) -> Path:
    """Get lock file path for daemon startup synchronization."""
    hash_val = hashlib.md5(str(project).encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"tldr-{hash_val}.lock"


def _is_daemon_alive(project: Path, retries: int = 3, delay: float = 0.1) -> bool:
    """Check if daemon is alive and responding to ping.

    This is used during startup to avoid spawning duplicate daemons.
    Uses retry logic to handle daemons that just started and may not
    be fully ready yet.

    Args:
        project: Project path
        retries: Number of attempts (default 3)
        delay: Seconds between attempts (default 0.1)

    Returns:
        True if daemon responds to ping, False otherwise
    """
    for attempt in range(retries):
        try:
            result = query_daemon(project, {"cmd": "ping"})
            if result.get("status") == "ok":
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return False


def _create_client_socket(daemon: "TLDRDaemon") -> socket.socket:
    """Create appropriate client socket for platform.

    Args:
        daemon: TLDRDaemon instance to get connection info from

    Returns:
        Connected socket ready for communication
    """
    addr, port = daemon._get_connection_info()

    if port is not None:
        # TCP socket for Windows
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((addr, port))
    else:
        # Unix socket for Linux/macOS
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(addr)

    return client


def start_daemon(project_path: str | Path, foreground: bool = False):
    """
    Start the TLDR daemon for a project.

    Uses file locking to prevent race conditions when multiple processes
    try to start the daemon simultaneously.

    Args:
        project_path: Path to the project root
        foreground: If True, run in foreground; otherwise daemonize
    """
    from .core import TLDRDaemon
    from ..tldrignore import ensure_tldrignore

    project = Path(project_path).resolve()

    # Early check: if daemon is already running, exit immediately
    # This prevents zombie processes when multiple hooks spawn daemons in parallel
    if _is_daemon_alive(project):
        print("Daemon already running")
        return

    # Ensure .tldrignore exists (create with defaults if not)
    created, message = ensure_tldrignore(project)
    if created:
        print(f"\n\033[33m{message}\033[0m\n")  # Yellow warning

    daemon = TLDRDaemon(project)

    if foreground:
        daemon.run()
    else:
        if sys.platform == "win32":
            # Windows: Use subprocess to run in background
            import subprocess

            # Get the connection info for display
            addr, port = daemon._get_connection_info()

            # Start detached process on Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            proc = subprocess.Popen(
                [sys.executable, "-m", "tldr.daemon", str(project), "--foreground"],
                startupinfo=startupinfo,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
            )
            print(f"Daemon started with PID {proc.pid}")
            print(f"Listening on {addr}:{port}")
        else:
            # Unix: Fork and run in background with file locking
            # This prevents race conditions when multiple agents start simultaneously
            lock_path = _get_lock_path(project)
            lock_path.touch(exist_ok=True)

            with open(lock_path, 'r') as lock_file:
                # Acquire exclusive lock - blocks until available
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                try:
                    # Re-check if daemon is running after acquiring lock
                    # Another process may have started it while we were waiting
                    # Use more aggressive retries here since this is the critical path
                    if _is_daemon_alive(project, retries=5, delay=0.2):
                        print("Daemon already running")
                        return

                    # Don't delete socket here - the bind logic in _create_server_socket()
                    # handles stale sockets properly. Deleting here causes race conditions
                    # when another process just started a daemon that isn't ready yet.

                    # Fork daemon process
                    pid = os.fork()
                    if pid == 0:
                        # Child process - run daemon
                        # NOTE: Don't release lock here! Parent holds it until daemon is ready.
                        # The lock is shared between parent/child after fork, so parent
                        # releasing it after daemon is confirmed ready is sufficient.
                        os.setsid()
                        daemon.run()
                        sys.exit(0)  # Should not reach here
                    else:
                        # Parent process - wait for daemon to be ready before releasing lock
                        start_time = time.time()
                        timeout = 10.0
                        while time.time() - start_time < timeout:
                            if _is_daemon_alive(project):
                                print(f"Daemon started with PID {pid}")
                                print(f"Socket: {daemon.socket_path}")
                                return
                            time.sleep(0.1)

                        # Daemon started but not responding - warn but don't fail
                        print(f"Warning: Daemon started (PID {pid}) but not responding within {timeout}s")
                        print(f"Socket: {daemon.socket_path}")
                finally:
                    # Release lock (parent only - child exits via daemon.run())
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass  # Lock may already be released by child


def stop_daemon(project_path: str | Path) -> bool:
    """
    Stop the TLDR daemon for a project.

    Args:
        project_path: Path to the project root

    Returns:
        True if daemon was stopped, False if not running
    """
    from .core import TLDRDaemon

    project = Path(project_path).resolve()
    daemon = TLDRDaemon(project)

    try:
        client = _create_client_socket(daemon)
        client.sendall(json.dumps({"cmd": "shutdown"}).encode() + b"\n")
        response = client.recv(4096)
        client.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def query_daemon(project_path: str | Path, command: dict) -> dict:
    """
    Send a command to the daemon and get the response.

    Args:
        project_path: Path to the project root
        command: Command dict to send

    Returns:
        Response dict from daemon
    """
    from .core import TLDRDaemon

    project = Path(project_path).resolve()
    daemon = TLDRDaemon(project)

    client = _create_client_socket(daemon)
    try:
        client.sendall(json.dumps(command).encode() + b"\n")
        response = client.recv(65536)
        return json.loads(response.decode())
    finally:
        client.close()


def main():
    """CLI entry point for daemon management."""
    import argparse

    parser = argparse.ArgumentParser(description="TLDR Daemon")
    parser.add_argument("project", help="Project path")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Get daemon status")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.stop:
        if stop_daemon(args.project):
            print("Daemon stopped")
        else:
            print("Daemon not running")
    elif args.status:
        try:
            result = query_daemon(args.project, {"cmd": "status"})
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Daemon not running: {e}")
    else:
        start_daemon(args.project, foreground=args.foreground)


if __name__ == "__main__":
    main()
