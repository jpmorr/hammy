# Version Information

The current version of Hammy is dynamically determined from Git tags using `hatch-vcs`.

## Checking Version

### In Python

```python
import hammy
print(hammy.__version__)
```

### From Command Line

```bash
python -c "import hammy; print(hammy.__version__)"
```

### From Package Metadata

```bash
uv run python -m pip show hammy
```

## Version Scheme

Hammy uses semantic versioning with automatic version generation from Git:

- **Tagged releases**: `1.0.0`, `1.0.1`, `2.0.0`, etc.
- **Development versions**: `0.0.post19+g4e7ffc23a.d20261105`

Development versions indicate:
- Number of commits since last tag
- Git commit hash
- Date of build

## Creating a Release

To create a new release version:

```bash
# Create and push a tag
git tag -a 1.0.0 -m "Release version 1.0.0"
git push origin 1.0.0

# Build and publish
uv build
uv publish
```

The version will automatically be `1.0.0` for tagged commits.

## For Users

- **PyPI releases**: Use `pip install hammy` for stable releases
- **Development versions**: Install from GitHub for latest features

```bash
# Latest development version
pip install git+https://github.com/jpmorr/hammy.git
```
