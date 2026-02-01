# PyPI Release Checklist

Before publishing to production PyPI, verify ALL items:

## Code Quality

- [ ] All tests passing (`pytest tests/`)
- [ ] No lint errors (`ruff check axis_core/`)
- [ ] Type checking clean (`mypy axis_core/ --strict`)
- [ ] Code coverage adequate (>80%)

## Documentation

- [ ] README.md updated
- [ ] CHANGELOG.md entry added
- [ ] API docs complete for new features
- [ ] Examples work and are tested

## Version & Metadata

- [ ] Version bumped correctly (semantic versioning)
- [ ] pyproject.toml classifiers accurate
- [ ] Development Status reflects reality:
  - `3 - Alpha` for 0.0.x - 0.9.x
  - `4 - Beta` for 0.10.x - 0.99.x
  - `5 - Production/Stable` for 1.0.0+

## Testing

- [ ] Published to TestPyPI
- [ ] Installed from TestPyPI in fresh venv
- [ ] Manual smoke tests passed
- [ ] No critical bugs known
- [ ] Breaking changes documented

## Release Notes

- [ ] CHANGELOG.md has clear entry
- [ ] Migration guide if breaking changes
- [ ] Known issues documented

## Git

- [ ] All changes committed
- [ ] Branch is `main` (not dev branch)
- [ ] Ready to tag after publish

## Final Question

**Is this version ready for users to depend on in production?**

- If NO → TestPyPI only
- If YES → Proceed to PyPI

---

## After Publishing to PyPI

- [ ] Tag release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push tags: `git push origin vX.Y.Z`
- [ ] Create GitHub release with notes
- [ ] Announce on Twitter/Discord/etc (if applicable)
- [ ] Update documentation site (if applicable)
