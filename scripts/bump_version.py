#!/usr/bin/env python3
"""
Version bump script for Hammy.

Usage:
    python scripts/bump_version.py major    # Bump major version (1.0.0 → 2.0.0)
    python scripts/bump_version.py minor    # Bump minor version (1.0.1 → 1.1.0)
    python scripts/bump_version.py patch    # Bump patch version (1.0.0 → 1.0.1)
    python scripts/bump_version.py 1.2.3    # Set specific version
    python scripts/bump_version.py 1.0.0rc1 # Set pre-release version
"""

import re
import sys
from pathlib import Path


def parse_version(version_str: str) -> tuple:
    """Parse version string into components."""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)(.*)', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type."""
    major, minor, patch, suffix = parse_version(current)
    
    if bump_type == "major":
        return f"{major + 1}.0.0{suffix}"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0{suffix}"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}{suffix}"
    else:
        # Assume it's a specific version
        return bump_type


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    bump_type = sys.argv[1]
    version_file = Path("src/hammy/__version__.py")
    
    if not version_file.exists():
        print(f"Error: Version file not found: {version_file}")
        sys.exit(1)
    
    # Read current version
    content = version_file.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    if not match:
        print("Error: Could not find __version__ in file")
        sys.exit(1)
    
    current_version = match.group(1)
    new_version = bump_version(current_version, bump_type)
    
    # Update version file
    new_content = content.replace(f'__version__ = "{current_version}"', 
                                   f'__version__ = "{new_version}"')
    version_file.write_text(new_content)
    
    print(f"Version bumped: {current_version} → {new_version}")
    print(f"File updated: {version_file}")


if __name__ == "__main__":
    main()
