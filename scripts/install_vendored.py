#!/usr/bin/env python3
"""Install vendored dependencies based on current platform."""

import platform
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent.parent / "vendor"


def get_platform_tag():
    """Get the platform tag for wheel matching."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine == "arm64":
            return "macosx_11_0_arm64"
        else:
            return "macosx_10_9_x86_64"
    elif system == "linux":
        if machine == "x86_64":
            return "manylinux_2_17_x86_64"
        elif machine == "aarch64":
            return "manylinux_2_17_aarch64"
    elif system == "windows":
        if machine == "amd64":
            return "win_amd64"

    return None


def install_vendored():
    """Install vendored wheels for current platform."""
    platform_tag = get_platform_tag()

    if not platform_tag:
        print(f"Unknown platform: {platform.system()} {platform.machine()}")
        print("You may need to build from source. See vendor/README.md")
        return 1

    print(f"Platform: {platform_tag}")

    # Find matching wheels
    installed = 0
    for wheel in VENDOR_DIR.glob("*.whl"):
        if platform_tag in wheel.name or "abi3" in wheel.name:
            # abi3 wheels are compatible across Python versions
            if platform_tag.split("_")[0] in wheel.name or "macosx" in wheel.name and "darwin" in platform.system().lower():
                print(f"Installing: {wheel.name}")
                # Try uv first, fall back to pip
                result = subprocess.run(
                    ["uv", "pip", "install", "--force-reinstall", str(wheel)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    # Fall back to pip
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)],
                        capture_output=True,
                        text=True
                    )
                if result.returncode == 0:
                    print(f"  Installed successfully")
                    installed += 1
                else:
                    print(f"  Failed: {result.stderr}")

    if installed == 0:
        print(f"No wheels found for platform {platform_tag}")
        print("See vendor/README.md for build instructions")
        return 1

    print(f"\nInstalled {installed} vendored package(s)")
    return 0


if __name__ == "__main__":
    sys.exit(install_vendored())
