# Kernels

This directory is the canonical public source for Intuition Labs Kernel
Workbench. The readable end-to-end procedure lives at
[intuitionlabs.tech/kernels](https://intuitionlabs.tech/kernels).

The current implemented translator is CUDA C++ to HIP C++ for AMD ROCm. The
bundled target profile is AMD Strix Halo (`gfx1151`). Future processor, memory,
and execution-topology values in the registry are vocabulary, not claims that
an adapter exists.

## Ownership and toolbox scope

Intuition Labs LLC owns and maintains Kernel Workbench: the procedure, work
orders, schemas, planner, adapters, tests, evidence contract, and integrations.

Kyuz0 publishes and maintains the external Strix Halo toolbox selected by the
default container profile. His role and authority stop at that toolbox. He does
not own or authorize Kernel Workbench or Intuition Labs kernel work. No Kyuz0
source is vendored here; the adapter pins and attributes an external image and
source repository.

## Source map

- `.codex-plugin/plugin.json` — Codex plugin metadata.
- `kernel-workbench/registry/` — source/target/toolchain/profile registry.
- `kernel-workbench/schemas/` — strict, content-free work-order schema.
- `kernel-workbench/scripts/kernel_workbench.py` — deterministic validator and
  planner.
- `skills/port-cuda-to-hip/assets/hipify-api-catalog.json` — the single full
  generated CUDA/HIP/ROCm API mapping object.
- `skills/port-cuda-to-hip/scripts/rocm_port.py` — bounded inventory and
  isolated translation utility.
- `skills/port-cuda-to-hip/scripts/real_smoke.py` — compile, AMD-device,
  reference-parity, and profiling witness driver.
- `skills/port-cuda-to-hip/references/official-porting-grounding.md` — pinned
  official AMD documentation and tool observations.
- `tests/` — planner, mapping-catalog, and porting regressions.

## Reproduce the public checks

Python 3.12 or newer is sufficient for the static planner and unit suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 kernel-workbench/scripts/kernel_workbench.py validate \
  kernel-workbench/examples/cuda-to-hip-strix-halo.v1.json
python3 kernel-workbench/scripts/kernel_workbench.py plan \
  kernel-workbench/examples/cuda-to-hip-strix-halo.v1.json
python3 skills/port-cuda-to-hip/scripts/hipify_api_catalog.py validate \
  skills/port-cuda-to-hip/assets/hipify-api-catalog.json
```

Inventory a CUDA-bearing source tree without modifying it:

```bash
python3 skills/port-cuda-to-hip/scripts/rocm_port.py inventory <source-root>
```

Create a separate HIP candidate only after reviewing the inventory:

```bash
python3 skills/port-cuda-to-hip/scripts/rocm_port.py translate \
  <source-root> <candidate-root> --tool hipify-clang
```

The real compile/device/parity witness additionally requires a compatible AMD
GPU, ROCm/HIP toolchain or the admitted digest-pinned container, and explicit
device access. It never follows from a successful static translation.

## Evidence law

`scanned`, `translated`, `manual-reviewed`, `compiled`, `linked`,
`executed-on-amd`, `reference-matched`, and `profiled` are separate claims.
The safe path is the default. Optimization is admitted only after compile,
target execution, and reference correspondence; every optimization returns
through the same counterexamples and reference check.

See [NOTICE.md](NOTICE.md) for attribution and source boundaries. Intuition
Labs-authored code in this directory is licensed under the repository's
AGPL-3.0-or-later license.
