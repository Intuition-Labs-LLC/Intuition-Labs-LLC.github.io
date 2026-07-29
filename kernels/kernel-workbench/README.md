# Kernel Workbench

Kernel Workbench is a public utility layer for describing and planning
source-kernel rewrites. It models source language, target language, processor
plane, memory topology, execution topology, publisher, runtime profile,
trajectory, optimization scope, and evidence as independent axes.

Intuition Labs LLC owns and maintains the Kernel Workbench procedure, schemas,
planner, adapters, tests, and evidence contract. Its canonical public source is
the [`kernels/` directory](https://github.com/Intuition-Labs-LLC/Intuition-Labs-LLC.github.io/tree/main/kernels)
in the Intuition Labs GitHub portal; this plugin is the installable Codex and
Maxwell distribution snapshot.

Only CUDA C++ to HIP C++ is implemented in this release. CPU, NPU, FPGA, ASIC,
hypervisor, multiplexer, and additional memory-topology values are vocabulary
for future work orders, not claims that adapters exist.

## Core objects

- `kernel-workbench-registry/v1` is the single capability/profile/toolchain
  registry.
- `kernel-work-order/v1` is a strict, content-free semantic request.
- `kernel-plan/v1` is the deterministic resolved path trace.
- `kernel-workbench-hold/v1` is a fail-closed resolution result.

The default profile is scoped by translator, target, and execution topology.
Today, `cuda-to-hip@amd-strix-halo-gfx1151@container` resolves to Kyuz0's
Strix Halo toolchain. Set `runtime_profile` to
`profile.operator.cuda-to-hip.native-rocm` and the target execution topology to
`native` to select an operator-managed ROCm installation without changing
planner code.

Kyuz0 is the external publisher and maintainer of the selected toolbox only.
That role grants no ownership or authority over Kernel Workbench, its procedure,
planner, schemas, integration, validation, or Intuition Labs kernel work.

## Commands

From the plugin root:

```bash
python3 kernel-workbench/scripts/kernel_workbench.py registry
python3 kernel-workbench/scripts/kernel_workbench.py validate \
  kernel-workbench/examples/cuda-to-hip-strix-halo.v1.json
python3 kernel-workbench/scripts/kernel_workbench.py plan \
  kernel-workbench/examples/cuda-to-hip-strix-halo.v1.json
```

`safe_reachable` is the default. A requested `leap_target` falls back to the
safe plan until compile, link, target execution, and reference-match evidence
are all present. The planner is pure: it neither translates nor executes code.
The existing `port-cuda-to-hip` skill owns those effects and their validation.

The public layer contains utilities, schemas, profile pointers, and evidence
gates. It intentionally contains no private optimization mechanisms, training
data, or internal runtime paths.
