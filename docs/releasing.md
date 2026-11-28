# Releasing

For maintainers. Releases go to PyPI as
[`nomem-ai`](https://pypi.org/project/nomem-ai/) through GitHub Actions with
PyPI trusted publishing, so no API token is stored anywhere.

## One-time setup

1. Sign in to [pypi.org](https://pypi.org) (create an account and turn on
   two-factor authentication if needed).
2. Go to [Your account, Publishing](https://pypi.org/manage/account/publishing/)
   and add a **pending publisher**:
    - PyPI project name: `nomem-ai`
    - Owner: `saswattulo`
    - Repository: `NoMem`
    - Workflow: `publish.yml`
    - Environment: `pypi`
3. In the GitHub repository, open Settings, Environments, and create an
   environment named `pypi` (optionally require your approval for it).

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `nomem/__init__.py`, and add a section to `CHANGELOG.md`.
2. Check locally:

    ```bash
    pip install -e ".[dev]"
    pytest
    python -m build
    twine check --strict dist/*
    ```

3. Merge to `main`, then create a GitHub release with a tag such as
   `v0.3.0`. The `publish to PyPI` workflow builds the sdist and wheel and
   uploads them.
4. Check `pip install nomem-ai==0.3.0` in a fresh environment.

To try the upload first, add a pending publisher on
[test.pypi.org](https://test.pypi.org) the same way and point a copy of the
workflow at it with `repository-url: https://test.pypi.org/legacy/`.

## Manual upload (fallback)

```bash
python -m build
twine upload dist/*     # asks for a PyPI API token
```

## Docs site

The site is built from `docs/` with MkDocs Material and deployed by the
`docs` workflow on every push to `main`. One-time setup: in the repository's
Settings, Pages, set **Source** to **GitHub Actions**. Preview locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```
