# Axis Memory Provider Contract v1

- Contract ID: `AXIS-MEMORY-PROVIDER-V1`
- Status: Draft (intended for implementation)
- Effective date: March 2, 2026
- Canonical location: `axis-core/docs/contracts/axis-memory-provider-v1.md`

## Purpose
Define a stable interoperability contract between `axis-core` and external memory providers (including `synaptic-core`) such that:
1. `axis-core` functions completely without any external provider installed.
2. Provider packages function completely without `axis-core` installed.
3. When both are installed, integration is native, predictable, and contract-tested.

## Non-Goals
1. This contract does not define internal graph/retrieval algorithms.
2. This contract does not require provider implementation details.
3. This contract does not require a specific database backend.

## Packaging and Discovery
1. Axis may integrate providers through:
   - Axis-owned built-in adapters, or
   - external plugin entry points in group `axis.memory`.
2. Provider registration must be install-time only; no hard import dependency from `axis-core` to a provider package.
3. For framework-agnostic provider cores (for example `synaptic-core`), framework adapter code should live outside the core provider package.
4. Entry point target (if used) may be either:
   - adapter class, or
   - factory returning adapter instance
   provided the resulting object satisfies this contract.

## Independence Requirements
### Axis-Core
1. `import axis_core` must succeed when provider packages are not installed.
2. Missing provider must produce actionable config/install errors, not import-time crashes.

### Provider Package
1. `import <provider_package>` must succeed when `axis-core` is not installed.
2. Base provider APIs must remain usable standalone.
3. Framework-specific adapter code should not be required for provider runtime use.
4. If framework adapters are published, they should be delivered as separate adapter packages or in the framework repository.

## Required Runtime Interface
A provider adapter loaded by `axis-core` must implement `MemoryAdapter` + `SessionStore` semantics equivalent to:

### `capabilities`
- Property: `set[MemoryCapability | str]`
- Supported values (strings or enums):
  - `semantic_search`
  - `keyword_search`
  - `ttl`
  - `namespaces`

### `store`
```python
async def store(
    key: str,
    value: Any,
    metadata: dict[str, Any] | None = None,
    ttl: int | None = None,
    namespace: str | None = None,
) -> None
```

### `retrieve`
```python
async def retrieve(
    key: str,
    namespace: str | None = None,
) -> Any | None
```

### `search`
```python
async def search(
    query: str,
    limit: int = 10,
    namespace: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[MemoryItem]
```

### `delete`
```python
async def delete(
    key: str,
    namespace: str | None = None,
) -> bool
```

### `clear`
```python
async def clear(
    namespace: str | None = None,
) -> int
```

### Session API
```python
async def store_session(session: SessionLike) -> SessionLike
async def retrieve_session(session_id: str) -> SessionLike | dict[str, Any] | None
async def update_session(session: SessionLike) -> SessionLike
```

## Behavioral Semantics
### Keys, Namespaces, and TTL
1. `key` must be non-empty after trimming.
2. `namespace=None` means provider default namespace.
3. `clear(namespace="*")` must clear all namespaces.
4. If TTL is supported and expiration has passed, `retrieve` must return `None`.
5. Expired records must not appear in `search` results.

### Search
1. `limit <= 0` should return an empty list or provider default bounded behavior; must not crash.
2. Result ordering must be deterministic for identical inputs and state.
3. `MemoryItem.score` may be `None`; if present, higher score should indicate better match.
4. `filters` applies to metadata equality semantics unless provider docs specify more.

### Delete/Clear
1. `delete` returns `True` if the item existed and was removed, else `False`.
2. `clear` returns count of removed items.

### Session Operations
1. `store_session` and `update_session` must enforce optimistic version checks.
2. Version conflict should raise an error that includes expected and actual version when available.
3. `retrieve_session` should return provider-native session object or serialized dict.

## Data Shape Normalization
To be Axis-compatible, adapters must normalize:
1. `MemoryItem.key` as `str`.
2. `MemoryItem.metadata` as `dict[str, Any]`.
3. `MemoryItem.namespace` as `str | None`.
4. `MemoryItem.created_at` / `expires_at` as `datetime | None` (timezone-aware preferred).
5. `capabilities` values into Axis `MemoryCapability` enum or equivalent string values.

## Error Contract
1. Invalid user input should raise `ValueError` with actionable message.
2. Missing keys are not errors for `retrieve`/`delete`.
3. Optional provider features that are unsupported should fail explicitly (for example: `NotImplementedError`) or via capability negotiation.
4. Unexpected provider backend failures may raise provider-specific exceptions; Axis adapter layer should map to clear runtime errors where practical.

## Compatibility and Versioning
1. This document defines provider contract version `v1`.
2. Providers should advertise supported contract version(s) in docs and release notes.
3. Axis should maintain a compatibility matrix by provider package/version.
4. Synaptic matrix canonical location:
   `axis-core/docs/contracts/synaptic-compatibility-matrix.md`.
5. Breaking changes to this contract require `v2` document and migration notes.

## Conformance Test Expectations
A provider integration is contract-compliant only if it passes:
1. Axis memory adapter protocol tests (store/retrieve/search/delete/clear).
2. Session store tests (store_session/retrieve_session/update_session with optimistic locking).
3. Namespace and TTL behavior tests.
4. Missing-dependency and plugin-discovery behavior tests.
5. Cross-repo interop CI job against supported version matrix.

## Synaptic Mapping Notes (Informative)
1. Synaptic should implement required memory/session primitives through canonical
   `synaptic_core.api.AsyncSynaptic` APIs (`set/get/find`, `remember/recall`) plus session
   persistence support.
2. Axis should own Synaptic adapter semantics in `axis-core`.
3. Synaptic package should remain framework-agnostic and not require Axis integration artifacts.

## Security and Operational Constraints
1. Provider must not execute arbitrary code from memory payload content.
2. Provider should enforce safe serialization boundaries.
3. Provider should avoid blocking hot paths with heavy maintenance operations.

## Migration Guidance (v1 Adoption)
1. Implement/retain contract-compatible adapter surface first.
2. Move framework-specific adapters to framework-owned code or dedicated bridge packages.
3. Deprecate legacy framework-coupled paths with warnings.
4. Remove legacy paths only after at least one documented migration window.
