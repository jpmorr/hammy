"""
MkDocs hooks for dynamic version injection.

These hooks are called by MkDocs/Zensical during the build process
and allow us to inject the package version into the documentation.
"""

import importlib.metadata


def get_package_version() -> str:
    """Get the current package version from hatch-vcs."""
    try:
        version = importlib.metadata.version("hammy")
        # Clean up development versions for display
        # e.g., "0.0.post19+g4e7ffc23a.d20261105" → "0.0.0-dev"
        if "+" in version or ".d" in version:
            # This is a dev build
            base_version = version.split("+")[0] if "+" in version else version.split(".d")[0]
            return f"{base_version}-dev"
        return version
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0-dev"


def on_config(config, **kwargs):
    """
    MkDocs hook: Called after config is loaded.
    
    Injects the package version into the site name.
    """
    version = get_package_version()
    
    # Add version to site name for browser title
    config["site_name"] = f"Hammy v{version} (dev)" if "dev" in version else f"Hammy v{version}"
    
    return config
