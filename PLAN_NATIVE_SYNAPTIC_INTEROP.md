# Axis-Core Plan: Native Synaptic Integration with Strict Package Decoupling

## Objective
Keep `axis-core` fully operational without `synaptic-core`, and make Synaptic integration first-class by owning the adapter layer entirely in Axis.

## Success Criteria
1. `axis-core` runs normally with built-in memory backends when Synaptic is not installed.
2. With Synaptic installed, Axis integrates natively through Axis-maintained adapter code.
3. Axis does not require any Axis-specific module inside `synaptic-core`.
4. Interop is enforced by contract tests and version compatibility checks.

## Boundary Rule (Hard Constraint)
1. `synaptic-core` remains framework-agnostic and contains no Axis integration code.
2. Axis owns all Synaptic adapter semantics in `axis-core` (or Axis-maintained plugin package).
3. Axis must never rely on Synaptic private fields.
4. Axis must never require Synaptic entrypoint declarations from the Synaptic package.

## Canonical Interop Contract
`/Users/xavierhillman/blackbox/code/axis-core/docs/contracts/axis-memory-provider-v1.md`

## Target End-State
1. Axis Synaptic integration is provided by `axis_core/adapters/memory/synaptic.py` against unified `synaptic_core.SynapticMemory` APIs.
2. Axis memory registry remains plugin-capable for other providers, but Synaptic support does not depend on Synaptic-provided entrypoints.
3. Axis docs and CLI make optional Synaptic installation paths explicit.

## Workstreams

### WS1: Lock and Enforce Provider Contract
1. Maintain `AXIS-MEMORY-PROVIDER-V1` as the source of truth.
2. Add adapter conformance fixtures for KV/session/capability semantics.
3. Add compatibility matrix (Axis version x Synaptic version).

Acceptance checks:
1. Axis interop tests verify behavior against contract semantics, not provider class names.

### WS2: Refactor Axis Synaptic Adapter
1. Update `axis_core/adapters/memory/synaptic.py` to target `synaptic_core.SynapticMemory` as primary integration surface.
2. Remove dependency on `synaptic_core.axis.SynapticAxisMemory` path assumptions.
3. Keep temporary compatibility fallback only as long as migration window requires.
4. Preserve Axis-facing adapter interface stability.

Acceptance checks:
1. Adapter works with Synaptic unified API with no Synaptic Axis module required.
2. Adapter surface remains backward-compatible for Axis users.

### WS3: Optional Dependency and UX
1. Keep Synaptic optional in Axis packaging.
2. Improve missing-dependency diagnostics and installation instructions.
3. Validate `AXIS_SYNAPTIC_PATH` behavior under adapter initialization.

Acceptance checks:
1. Axis startup without Synaptic has no import failures.
2. Missing Synaptic produces clear install guidance.

### WS4: Runtime Hardening
1. Add provider capability negotiation and strict validation at adapter init time.
2. Fail fast on unsupported provider versions with actionable messaging.
3. Keep hot paths resilient to provider telemetry/storage failure modes.

Acceptance checks:
1. Version mismatch is caught before runtime corruption.
2. Compatible versions run without warnings/errors.

### WS5: Cross-Repo Interop CI
1. Add Axis CI jobs testing:
   - latest released Synaptic
   - current Synaptic main branch
2. Add minimum-version pin tests.
3. Add migration tests that verify transition away from legacy SynapticAxis paths.

Acceptance checks:
1. Interop regressions are caught pre-release.

### WS6: Docs and Migration
1. Update Axis docs/changelog with:
   - Synaptic is optional
   - Axis owns Synaptic adapter
   - supported Synaptic versions
2. Provide migration guidance for any legacy Axis+Synaptic setup relying on removed Synaptic Axis artifacts.

Acceptance checks:
1. Users can configure Axis with or without Synaptic from docs alone.

## Release Plan
1. Release N:
   - adapter supports unified Synaptic APIs
   - contract tests active in CI
2. Release N+1:
   - deprecation warnings for legacy fallback paths
3. Release N+2:
   - remove legacy fallback paths

## Risks and Mitigations
1. Risk: Synaptic API changes break Axis integration.
   - Mitigation: strict contract tests + version matrix CI.
2. Risk: user confusion around optional dependencies.
   - Mitigation: explicit docs and runtime diagnostics.
3. Risk: hidden coupling to removed Synaptic Axis artifacts.
   - Mitigation: static checks and targeted regression tests.

## Done Definition
1. Axis is fully functional standalone without Synaptic.
2. Axis integrates with Synaptic natively through Axis-owned adapter code only.
3. Interop is seamless, contract-verified, and independent of Synaptic Axis-specific artifacts.
