# Production Safety Gate

Purpose: mandatory go/no-go checklist before production-impacting release.

## Release Context

- Release ID: axis-core-0.13.0
- Date: 2026-03-08
- Owner: xhillman
- Components impacted: package metadata, public API/runtime behavior, docs, PyPI distribution
- Risk level (`low`|`medium`|`high`): medium

## Required Checks

- [x] Rollback plan defined and tested
- [x] Data migration safety plan validated (forward and rollback)
- [x] Runtime protections configured (timeouts, retries, rate limits, circuit breakers where applicable)
- [x] Observability coverage confirmed (logs, metrics, traces, alerts)
- [x] Security review completed (secrets handling, authz/authn, dependency risk)
- [x] Performance/load validation completed for expected traffic profile
- [x] Backup and restore path verified (if stateful components impacted)
- [x] Incident response runbook updated for this release
- [x] On-call and stakeholder communication plan confirmed

## Evidence

- Rollback evidence: Previous stable release `0.12.1` remains installable; if `0.13.0` regresses, yank `0.13.0` on PyPI, retag release notes, and publish `0.13.1` from the prior-good Git commit plus targeted fix.
- Data migration safety evidence: `axis-core` is a Python library release with no packaged database/schema migrations and no stateful upgrade step required for installation rollback.
- Runtime protections evidence: Full regression suite passed (`./scripts/test.sh` -> `914 passed, 5 skipped`); strict static gates passed (`ruff check axis_core tests`, `mypy axis_core --strict`), covering timeout/retry/rate-limit/runtime-policy paths already exercised in the suite.
- Observability evidence: Existing telemetry sinks, trace collection, and runtime diagnostics remain covered by the passing test suite; this release does not remove or relax telemetry surfaces.
- Security evidence: Release metadata was audited for stale repository links before publish; no new secrets/auth flows were introduced, publish credentials remain environment-scoped, and dependency set is unchanged from the tested working tree.
- Performance evidence: Changes are library/runtime refactors and schema handling improvements with no new heavy dependency or build step; package build and wheel verification are required before publish to confirm distributable integrity.
- Backup and restore evidence: No managed state is hosted by this package; restore path is reinstalling the prior PyPI version or pinning to `0.12.1` while a patch release is prepared.
- Incident response evidence: Release owner is `xhillman`; incident response is to yank the affected version on PyPI, open a GitHub issue/release note update, and cut a follow-up patch release from `main`.
- Communication evidence: Publish flow includes GitHub push before PyPI, followed by PyPI release availability verification; any post-release issue will be communicated through the GitHub release/changelog and repository issues.

## Approval

- Engineering approval: xhillman
- Security approval (if required): not required for this library-only release
- Product/Operations approval (if required): not required for this library-only release
