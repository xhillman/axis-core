# Synaptic Compatibility Matrix

- Contract: `AXIS-MEMORY-PROVIDER-V1`
- Last verified: March 3, 2026
- Axis adapter surface: `axis_core.adapters.memory.synaptic.SynapticMemory`

## Supported Synaptic Versions

| axis-core | synaptic-core | Status | Evidence |
|---|---|---|---|
| `0.12.0b` | `0.3.x` (`>=0.3.0,<0.4.0`) | verified | `tests/adapters/memory/test_synaptic.py`, `tests/adapters/memory/test_synaptic_integration.py`, `tests/engine/test_registry.py`, local interop run on `0.3.0` |
| `0.12.0b` | `<0.3.0` | not_supported | Adapter init version gate fails fast with actionable upgrade guidance |
| `0.12.0b` | `>=0.4.0` | not_verified | Outside currently validated range; blocked by adapter init version gate |

## Notes

1. Axis integration targets canonical `synaptic_core.api.AsyncSynaptic` APIs (`set/get/find`,
   `remember/recall`) and validated session persistence paths.
2. Axis does not depend on `synaptic_core.axis.*` artifacts.
3. Synaptic remains optional; core Axis runtime works without Synaptic installed.
