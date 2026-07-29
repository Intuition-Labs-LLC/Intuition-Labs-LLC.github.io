# Kernel Workbench

> A CUDA → HIP port is not done when the file changes. It is done when it
> runs—and matches.

[![Kernels CI](https://github.com/Intuition-Labs-LLC/Intuition-Labs-LLC.github.io/actions/workflows/kernels.yml/badge.svg)](https://github.com/Intuition-Labs-LLC/Intuition-Labs-LLC.github.io/actions/workflows/kernels.yml)
[![Procedure](https://img.shields.io/badge/procedure-intuitionlabs.tech%2Fkernels-1f6f78)](https://intuitionlabs.tech/kernels)

This directory is the canonical public source for Intuition Labs Kernel
Workbench. The readable end-to-end procedure lives at
[intuitionlabs.tech/kernels](https://intuitionlabs.tech/kernels).

The current implemented translator is CUDA C++ to HIP C++ for AMD ROCm. The
bundled target profile is AMD Strix Halo (`gfx1151`). Future processor, memory,
and execution-topology values in the registry are vocabulary, not claims that
an adapter exists.

## What is real today

The bundled, bounded integer witness was catalog-inventoried and translated by
`hipify-clang`, compiled and linked by `hipcc` for `gfx1151`, executed on the
named AMD device, matched its frozen reference exactly, and produced a
ROCprofiler trace containing the named `add_bias` dispatch.

| Receipt | Observed result |
| --- | --- |
| HIPIFY | 15 converted, 0 unconverted references |
| Target | compiled, linked, and executed on AMD `gfx1151` |
| Correspondence | 257 integers; expected sum `553064`, observed sum `553064` |
| Profiling | named `add_bias(int const*, int*, int)` GPU dispatch captured |

This proves only the bundled smoke kernel and named toolchain path. It is not a
benchmark and does not certify another project, kernel, workload, or machine.

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

The same checks run automatically for pull requests and pushes that change
`kernels/`.

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
