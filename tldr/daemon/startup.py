"""
Daemon lifecycle management: start, stop, query.

Uses file locking on PID file as the primary synchronization mechanism.
The lock is held for the daemon's entire lifetime, preventing duplicates.
Cross-platform: fcntl.flock() on Unix, msvcrt.locking() on Windows.
"""

import hashlib
import json
import logging
import os
import socket
import sys
import tempfile
import time

from pathlib import Path
from typing import TYPE_CHECKING, Optional, IO

# Platform-specific imports for file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from .core import TLDRDaemon

logger = logging.getLogger(__name__)


def _get_lock_path(project: Path) -> Path:
    """Get lock file path for daemon startup synchronization."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{hash_val}.lock"


def _get_pid_path(project: Path) -> Path:
    """Get PID file path for daemon process tracking."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{hash_val}.pid"


def _get_socket_path(project: Path) -> Path:
    """Get socket path for daemon communication."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{hash_val}.sock"


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    if sys.platform == "win32":
        # Windows: use tasklist or ctypes
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)  # Signal 0 = check if process exists
            return True
        except (OSError, ProcessLookupError):
            return False


def _try_acquire_pidfile_lock(pid_path: Path) -> Optional[IO]:
    """Try to acquire exclusive lock on PID file.

    Returns:
        File handle if lock acquired (caller must keep it open!), None if locked by another process.
    """
    try:
        # Open in append mode to create if not exists, don't truncate
        pidfile = open(pid_path, "a+")

        if sys.platform == "win32":
            # Windows: msvcrt.locking with LK_NBLCK (non-blocking)
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_NBLCK, 1)
                return pidfile
            except (IOError, OSError):
                # Lock held by another process
                pidfile.close()
                return None
        else:
            # Unix: fcntl.flock with LOCK_NB (non-blocking)
            try:
                fcntl.flock(pidfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return pidfile
            except (IOError, BlockingIOError):
                pidfile.close()
                return None
    except PermissionError:
        # Windows: file locked by another process prevents open
        logger.debug(f"PID file locked by another process: {pid_path}")
        return None
    except FileNotFoundError:
        # File doesn't exist - no daemon running, but we can't create it here
        # Return a special sentinel to distinguish from "locked"
        logger.debug(f"PID file not found: {pid_path}")
        return None
    except Exception as e:
        logger.debug(f"Failed to open PID file: {e}")
        return None


def _write_pid_to_locked_file(pidfile: IO, pid: int) -> None:
    """Write PID to an already-locked file."""
    pidfile.seek(0)
    pidfile.truncate()
    pidfile.write(str(pid))
    pidfile.flush()


def _is_socket_connectable(project: Path, timeout: float = 1.0) -> bool:
    """Check if daemon socket exists and accepts connections.

    This is more robust than ping-based check because it doesn't
    depend on response format - just whether a daemon is listening.
    """
    socket_path = _get_socket_path(project)
    if not socket_path.exists():
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(str(socket_path))
        sock.close()
        return True
    except (socket.error, OSError):
        return False



def _is_daemon_alive(project: Path, retries: int = 3, delay: float = 0.1) -> bool:
    """Check if daemon is alive using file lock on PID file.

    This is the authoritative check - if we can't acquire the lock,
    another daemon is holding it and is therefore alive. No socket
    connectivity check needed (avoids race conditions with slow daemons).

    Args:
        project: Project path
        retries: Number of attempts (default 3) - used for brief retries
        delay: Seconds between attempts (default 0.1)

    Returns:
        True if daemon is alive (lock held by another process), False otherwise
    """
    pid_path = _get_pid_path(project)

    for attempt in range(retries):
        # Try to acquire lock - if we can't, daemon is running
        pidfile = _try_acquire_pidfile_lock(pid_path)
        if pidfile is None:
            # Lock held by another process = daemon is alive
            return True

        # We got the lock - check if there's a stale PID
        pidfile.seek(0)
        content = pidfile.read().strip()
        if content:
            try:
                pid = int(content)
                if _is_process_running(pid):
                    # Process exists but we got the lock? Shouldn't happen normally.
                    # Could be a daemon that crashed after writing PID but before locking.
                    # Release lock and report alive (process still running).
                    if sys.platform == "win32":
                        msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
                    pidfile.close()
                    return True
            except ValueError:
                pass  # Corrupt PID, ignore

        # Release lock - no daemon running
        if sys.platform == "win32":
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        pidfile.close()

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


def _recv_all(sock: socket.socket) -> bytes:
    """Receive complete response from socket until newline delimiter.

    Socket recv() returns only data currently available, not the full
    requested buffer size. This function loops until we receive the
    newline delimiter that marks the end of the message.

    Args:
        sock: Connected socket to read from

    Returns:
        Complete response data (with newline stripped)
    """
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    return data.rstrip(b"\n")


def start_daemon(project_path: str | Path, foreground: bool = False):
    """
    Start the TLDR daemon for a project.

    Uses file locking on the PID file as the primary synchronization mechanism.
    The lock is held for the daemon's entire lifetime, preventing duplicates.

    Args:
        project_path: Path to the project root
        foreground: If True, run in foreground; otherwise daemonize
    """
    from .core import TLDRDaemon
    from ..tldrignore import ensure_tldrignore

    project = Path(project_path).resolve()
    pid_path = _get_pid_path(project)

    # Try to acquire exclusive lock on PID file
    # If we can't, another daemon is running
    pidfile = _try_acquire_pidfile_lock(pid_path)
    if pidfile is None:
        print("Daemon already running")
        return

    # We have the lock - we're the only one starting a daemon
    # Ensure .tldrignore exists (create with defaults if not)
    created, message = ensure_tldrignore(project)
    if created:
        print(f"\n\033[33m{message}\033[0m\n")  # Yellow warning

    daemon = TLDRDaemon(project)

    if foreground:
        # Write PID and run - pidfile stays open (lock held)
        _write_pid_to_locked_file(pidfile, os.getpid())
        daemon._pidfile = pidfile  # Daemon keeps reference to hold lock
        daemon.run()
    else:
        if sys.platform == "win32":
            # Windows: Use subprocess to run in background
            # Release our lock - the subprocess will acquire its own
            import subprocess
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            pidfile.close()

            # Acquire lock to prevent race conditions
            lock_path = _get_lock_path(project)
            # Ensure lock file exists
            if not lock_path.exists():
                lock_path.touch()

            try:
                with open(lock_path, "w") as lock_file:
                    # Windows locking: try to acquire lock
                    # msvcrt.locking raises OSError if locked when using LK_NBLCK, 
                    # or blocks 10s with LK_RLCK. We want to wait until acquired.
                    start_lock = time.time()
                    while True:
                        try:
                            # Lock the first byte
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.time() - start_lock > 10.0:
                                print("Timeout waiting for daemon lock")
                                return
                            time.sleep(0.1)
                    
                    try:
                        # Re-check if daemon is alive (race condition handling)
                        if _is_daemon_alive(project):
                            print("Daemon already running")
                            return

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
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        print(f"Daemon started with PID {proc.pid}")

                        # Verify daemon is listening
                        start_wait = time.time()
                        connected = False
                        while time.time() - start_wait < 5.0:
                            try:
                                with socket.create_connection((addr, port), timeout=0.5):
                                    connected = True
                                    break
                            except (OSError, ConnectionRefusedError):
                                time.sleep(0.1)

                        if connected:
                            print(f"Listening on {addr}:{port}")
                        else:
                            logger.error("Daemon started but failed to accept connections")
                            # Should we kill it? Maybe not strictly required but logging is good.

                    finally:
                        # Release lock
                        try:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
            except Exception:
                logger.exception("Error starting daemon")

        else:
            # Unix: Fork and run in background
            # Child inherits the lock, parent releases it

            # Fork daemon process
            pid = os.fork()
            if pid == 0:
                # Child process - we inherit the lock
                os.setsid()
                # Write our PID to the locked file
                _write_pid_to_locked_file(pidfile, os.getpid())
                daemon._pidfile = pidfile  # Keep reference to hold lock
                daemon.run()
                sys.exit(0)  # Should not reach here
            else:
                # Parent process - release lock and wait for daemon
                try:
                    fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                pidfile.close()

                # Wait for daemon to be ready (socket exists)
                start_time = time.time()
                timeout = 10.0
                socket_path = _get_socket_path(project)
                while time.time() - start_time < timeout:
                    if socket_path.exists() and _is_socket_connectable(project, timeout=0.5):
                        print(f"Daemon started with PID {pid}")
                        print(f"Socket: {daemon.socket_path}")
                        return
                    time.sleep(0.1)

                # Daemon started but socket not ready - warn but don't fail
                print(f"Warning: Daemon (PID {pid}) socket not ready within {timeout}s")
                print(f"Socket: {daemon.socket_path}")


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
        _recv_all(client)  # Consume response (we don't need the content)
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
        response = _recv_all(client)
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
