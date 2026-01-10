#!/usr/bin/env python3
"""
Install Swift support for TLDR.

Usage:
    python -m tldr.install_swift

Swift requires a separate installation step because the upstream PyPI package
(tree-sitter-swift) is broken. This script installs a pre-built wheel for
macOS ARM64, or provides instructions for other platforms.
"""

import platform
import subprocess
import sys
from pathlib import Path


def get_vendor_dir() -> Path:
    """Get the vendor directory path."""
    return Path(__file__).parent.parent / "vendor"


def install_swift():
    """Install tree-sitter-swift for Swift language support."""
    print("Installing Swift support for TLDR...")
    print()

    system = platform.system().lower()
    machine = platform.machine().lower()

    vendor_dir = get_vendor_dir()

    # Check for macOS ARM64 wheel
    if system == "darwin" and machine == "arm64":
        wheel = vendor_dir / "tree_sitter_swift-0.0.1-cp38-abi3-macosx_11_0_arm64.whl"
        if wheel.exists():
            print(f"Found pre-built wheel for macOS ARM64")
            print(f"Installing: {wheel.name}")
            print()

            # Try uv first, fall back to pip
            result = subprocess.run(
                ["uv", "pip", "install", "--force-reinstall", str(wheel)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                print("Swift support installed successfully!")
                print()
                print("Test it with:")
                print("  tldr extract yourfile.swift")
                return 0
            else:
                print(f"Installation failed: {result.stderr}")
                return 1
        else:
            print(f"Wheel not found at {wheel}")
            print("Please reinstall tldr-code or build from source (see below)")

    # Other platforms - provide build instructions
    print(f"Platform: {system} {machine}")
    print()
    print("No pre-built wheel available for your platform.")
    print("Please build from source:")
    print()
    print("  # Install tree-sitter CLI (requires Node.js)")
    print("  npm install -g tree-sitter-cli")
    print()
    print("  # Clone and build")
    print("  git clone https://github.com/alex-pinkus/tree-sitter-swift.git")
    print("  cd tree-sitter-swift")
    print("  tree-sitter generate")
    print("  pip install .")
    print()
    print("After building, Swift support will be available in TLDR.")

    return 1


if __name__ == "__main__":
    sys.exit(install_swift())
