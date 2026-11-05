#!/usr/bin/env python3
"""
Update version in documentation files before build.

This script reads the version from hatch-vcs and updates:
- README.md badges
- docs/index.md version references
"""

import re
import importlib.metadata
from pathlib import Path


def get_version() -> str:
    """Get version from hatch-vcs."""
    try:
        version = importlib.metadata.version("quatham")
        # Clean up dev versions
        if "+" in version or ".d" in version:
            base = version.split("+")[0] if "+" in version else version.split(".d")[0]
            return f"{base}-dev"
        return version
    except Exception:
        return "0.0.0-dev"


def update_readme(version: str) -> None:
    """Update README.md with current version."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return
    
    content = readme_path.read_text()
    
    # Update PyPI badge version (optional - usually PyPI shows latest)
    # We'll leave this as-is since it links to PyPI which shows the published version
    
    print(f"README.md: Version badge links to PyPI (auto-updated)")


def update_docs_index(version: str) -> None:
    """Update docs/index.md with current version."""
    index_path = Path("docs/index.md")
    if not index_path.exists():
        return
    
    content = index_path.read_text()
    
    # Look for version placeholder or add it
    if "{{ version }}" in content:
        content = content.replace("{{ version }}", version)
        index_path.write_text(content)
        print(f"docs/index.md: Updated version to {version}")
    else:
        print(f"docs/index.md: No version placeholder found")


def main():
    version = get_version()
    print(f"Current version: {version}")
    
    update_readme(version)
    update_docs_index(version)
    
    # Also update mkdocs.yml site_name
    mkdocs_path = Path("mkdocs.yml")
    if mkdocs_path.exists():
        content = mkdocs_path.read_text()
        # Replace site_name if it has a version
        if "site_name: Hammy v" in content:
            content = re.sub(
                r'site_name: Hammy v[\d\.a-z\-+]+',
                f'site_name: Hammy v{version}',
                content
            )
            mkdocs_path.write_text(content)
            print(f"mkdocs.yml: Updated site_name to Hammy v{version}")


if __name__ == "__main__":
    main()
