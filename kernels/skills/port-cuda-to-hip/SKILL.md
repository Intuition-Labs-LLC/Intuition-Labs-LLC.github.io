---
name: port-cuda-to-hip
description: Inventory, translate, review, and validate CUDA kernels or CUDA-bearing C/C++ projects as HIP C++ candidates for AMD ROCm. Use when Codex is asked to port `.cu`/`.cuh` code, replace CUDA runtime or library APIs with HIP/ROCm equivalents, assess HIPIFY readiness, review a CUDA-to-HIP diff, make a CUDA project portable across AMD and NVIDIA backends, or plan/execute ROCm correctness and performance validation.
---

# Port CUDA to HIP

Treat a port as a staged correspondence, not a rename claim. Keep `scanned`,
`translated`, `compiled`, `linked`, `executed-on-amd`, `reference-matched`, and
`profiled` separate. Read
[references/official-porting-grounding.md](references/official-porting-grounding.md)
before choosing tools or making support claims.

## 0. Resolve the semantic work order

For a new port, encode the request as `KernelWorkOrderV1` and resolve it through
the public [Kernel Workbench](../../kernel-workbench/README.md) before invoking
translation tools:

```bash
plugin_root="$(cd "${skill_root}/../.." && pwd)"
python3 "${plugin_root}/kernel-workbench/scripts/kernel_workbench.py" plan \
  <kernel-work-order.json>
```

The work order keeps source/target dialects, processor plane, memory topology,
execution topology, runtime profile, trajectory, optimization scopes, evidence,
and revision pins as independent dimensions. It contains logical references and
digests, not source content or local absolute paths.

The current implemented SKU is CUDA C++ to HIP C++. The scoped
`cuda-to-hip@amd-strix-halo-gfx1151@container` default selects
`profile.kyuz0.cuda-to-hip.strix-halo-gfx1151`. Kyuz0 is the external
publisher and maintainer of that toolbox only. Intuition Labs LLC owns Kernel
Workbench, this procedure, its schemas, planner, adapters, tests, and evidence
contract. Set `runtime_profile` explicitly to switch profiles without changing
planner code. The operator-managed native option is
`profile.operator.cuda-to-hip.native-rocm`.

`safe_reachable` is the default trajectory. A requested `leap_target`
automatically resolves to a safe fallback until compile, link, target execution,
and reference parity evidence are present. The plan is a deterministic path
trace, not execution or correctness evidence.

## 1. Freeze the port contract

Inspect applicable repository instructions, dirty paths, source ownership, build
files, tests, and current failures before editing. Record:

- CUDA source revision and toolkit/API versions;
- target AMD product, `gfx` architecture, OS, driver, and ROCm component
  versions;
- whether the result must remain one portable AMD/NVIDIA source base or may use
  AMD-specific `roc*` libraries;
- the known-good NVIDIA/CPU reference corpus and numeric tolerances;
- owned output paths, protected dirty paths, and rollback;
- approval holds for dependency installation, external checkout, live device
  execution, publication, or deployment.

If the target or baseline is unknown, continue with a static inventory only and
leave the corresponding compile, parity, or performance claim at `HOLD`.

## 2. Inventory before translation

Resolve `skill_root` to this skill directory. Run the bounded, read-only helper:

```bash
python3 "${skill_root}/scripts/rocm_port.py" inventory <source-root>
```

Use `--output <report.json>` when a durable local report is authorized. The
report contains relative paths, digests, counts, risk identifiers, and line
numbers; it does not contain source lines. It includes `.cu`, `.cuh`, `.hip`,
and `.hip.*` candidates so the same bounded mechanism can review residual risks
after translation. `NO_GPU_SOURCE_CANDIDATES` is not a clean-port claim.

Review every high-risk finding. In particular, search for inline PTX,
hard-coded warp width or lane masks, `__CUDA_ARCH__`, warp intrinsics,
two-argument `__launch_bounds__`, CUDA libraries, driver/context APIs,
cooperative groups, divergent barriers, texture/surface code, graphs,
architecture flags, atomics, async-copy primitives, and NVIDIA-specific types.
An empty static finding set is not runtime proof.

Probe available tools without installing anything:

```bash
hipify-clang --version
hipify-clang --versions
hipcc --version
hipconfig --full
rocminfo
```

Compare the observed toolchain and source CUDA version against the live official
compatibility and HIPIFY documentation. A version outside the documented
translator range is `HOLD(unsupported_or_unverified_cuda_input)` unless the
user explicitly authorizes an experimental attempt.

### Mapping catalog primitive

Treat the bundled
[`assets/hipify-api-catalog.json`](assets/hipify-api-catalog.json) as the
versioned CUDA→HIP/ROC transpiler ontology for inventory and lookup. It is one
normalized object generated from HIPIFY's full joint CSV export. Each CUDA
symbol has typed HIP and ROC targets, added/changed/deprecated/removed metadata,
unsupported CUDA-version ranges, experimental markers, domain, section, source
digests, and the generating tool receipt.

The inventory helper loads this catalog by default and emits catalog-backed API
references. A CUDA-like identifier absent from the catalog is a high-risk
manual-review finding. Query an exact mapping without loading the whole object
into conversation:

```bash
python3 "${skill_root}/scripts/hipify_api_catalog.py" lookup \
  "${skill_root}/assets/hipify-api-catalog.json" cudaMalloc --backend hip
```

Regenerate the catalog only from the exact installed `hipify-clang` being
qualified:

```bash
mapping_dir="$(mktemp -d)"
hipify-clang --csv --doc-format=full --doc-roc=joint --o-dir="${mapping_dir}"
python3 "${skill_root}/scripts/hipify_api_catalog.py" normalize \
  "${mapping_dir}" <new-catalog.json> \
  --tool-version "$(hipify-clang --version | head -1)" \
  --generation-command \
  'hipify-clang --csv --doc-format=full --doc-roc=joint --o-dir=<output-directory>'
python3 "${skill_root}/scripts/hipify_api_catalog.py" validate \
  <new-catalog.json>
```

Review the complete catalog/source digest diff before replacing the bundled
object. Do not download the documentation and scrape rendered HTML when the
installed generator is available. A catalog mapping is static transpiler
capability evidence, not compile, semantic-equivalence, device-parity, or
performance evidence.

When `hipify-clang` is present, also run its live-help-verified non-output
statistics mode (for example, `--no-output --print-stats`) with the same compile
database, CUDA path, includes, and defines intended for translation. Reconcile
converted, unconverted, warning, and experimental counts. HIPIFY statistics are
scope evidence, not correctness evidence.

## 3. Produce an isolated candidate

Prefer `hipify-clang` with the real CUDA headers, include/define flags, and
`compile_commands.json`. Create a new output tree outside the source root:

```bash
python3 "${skill_root}/scripts/rocm_port.py" convert \
  <source-root> <new-output-root> \
  --tool hipify-clang \
  --compile-db <directory-containing-compile_commands.json> \
  --cuda-path <cuda-toolkit-root>
```

The helper refuses in-place work, existing output, path escape, missing tools,
bounds overflow, timeouts, and partial publication. It stages all translations
and exposes the candidate tree only after every selected file succeeds.
Treat the candidate and translator diagnostics with the same sensitivity and
retention policy as the input source; do not place either in operational logs.

Use `--tool hipify-perl` only when the user explicitly accepts a lower-confidence
textual fallback. Do not silently fall back from `hipify-clang`. Use `--roc`
only when AMD-optimized library mappings are desired and the contract does not
require backend-neutral HIP library calls.

Do not run HIPIFY `--inplace`. Do not install CUDA, ROCm, Clang, or HIPIFY
without approval. Do not call a nonempty candidate correct.

### Repeatable managed toolchain

Prefer a repository or operator-owned native ROCm installation when it matches
the target. On immutable hosts, use an approved, digest-pinned container rather
than layering packages onto the host. This plugin includes
[`Containerfile`](../../kernel-workbench/publishers/external/kyuz0/strix-halo-gfx1151/Containerfile)
for the observed `gfx1151` path. The runtime profile records Kyuz0 as the
external toolbox publisher for that toolbox only, pins the admitted ROCm/HIPIFY
base image, and adds CUDA 12.9 runtime headers because the observed HIPIFY
build reports CUDA input support only through 12.9. No Kyuz0 source is
vendored. Preserve attribution and hold redistribution until the current
upstream license and notice requirements are verified.

Build and install the local image from the plugin root only with dependency and
network approval:

```bash
podman build \
  -f kernel-workbench/publishers/external/kyuz0/strix-halo-gfx1151/Containerfile \
  -t localhost/kernel-workbench-cuda-hip:7.13-cuda12.9 .
```

Run the real bundled acceptance witness with the AMD device explicitly exposed:

```bash
podman run --rm \
  --network=none \
  --read-only \
  --workdir /tmp \
  --cap-drop=all \
  --security-opt no-new-privileges \
  --pids-limit=512 \
  --memory=8g \
  --cpus=8 \
  --device /dev/kfd \
  --device /dev/dri/card1 \
  --device /dev/dri/renderD128 \
  --group-add keep-groups \
  --security-opt label=disable \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g,mode=1777 \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${skill_root}:${skill_root}:ro" \
  localhost/kernel-workbench-cuda-hip:7.13-cuda12.9 \
  python3 "${skill_root}/scripts/real_smoke.py" --gfx gfx1151 --profile
```

The witness succeeds only after the bundled CUDA source is catalog-inventoried,
AST-translated by `hipify-clang`, compiled and linked by `hipcc` for the named
`gfx` target, executed through the observed AMD device, and exact integer
reference parity passes. It emits source/candidate/executable digests and
separate evidence labels. `profiled` is granted only when the bounded
ROCprofiler JSON contains the named `add_bias` kernel dispatch on a GPU agent.
A passing smoke proves only this small witness and toolchain path; it does not
prove another project or kernel.

Before reusing a container, record its immutable image digest, tool versions,
CUDA header version, device nodes, `rocminfo` target, and command. Do not
silently pull a floating image, mount credentials, expose unrelated host paths,
or reuse a background serving container for translation.

## 4. Reconcile semantics and integrate narrowly

Review the candidate against the original and HIPIFY diagnostics. Resolve each
inventory finding and each unsupported/experimental mapping against the current
API tables. Manually handle:

- library calls without an exact HIP/ROCm equivalent;
- PTX, architecture-specific intrinsics, and feature tests;
- warp/wave size, lane-mask width, synchronization, atomic scope, and memory
  ordering assumptions;
- `__launch_bounds__`, occupancy, register flags, and shared-memory layout;
- generated CUDA, macros/templates, nested headers, and build-system flags;
- driver/context/module differences, graphs, textures, and multi-GPU behavior;
- target-specific memory topology. Never delete copies merely because an APU
  may share memory.

Apply the reviewed changes through the repository's existing build/runtime
owner. Preserve unrelated changes and add the negative tests with the positive
path. Keep portable HIP and AMD-specific optimization as separate commits or
clearly separable diffs when both are required.

## 5. Climb the verification ladder

Use the owning project's commands; do not invent a parallel build. At minimum:

1. Re-run the inventory over the candidate and reconcile remaining
   CUDA/PTX/NVIDIA identifiers and HIP architecture-sensitive findings.
2. Compile every translated unit and link the real application with
   `HIP_PLATFORM=amd` and the actual `--offload-arch=<gfx...>` or equivalent
   CMake target.
3. Exercise zero, one, odd, large, non-workgroup-multiple, multidimensional,
   aliasing, stream, atomic, and relevant wave-size cases.
4. Compare against the frozen reference: integers exactly and floating results
   using algorithm-defined tolerances.
5. Capture the requested/observed AMD device, ROCm stack, executable revision,
   inputs, outputs, and test result.
6. Establish an AMD performance baseline only after correctness, then profile
   with `rocprofv3` and re-measure each optimization.

Use ROCgdb and the documented temporary serialization settings for diagnosis,
not as default production configuration. If the user supplies a compatible
PyTorch profiler trace and asks for agentic trace analysis, hand it to
the installed `AMD Skills:tracelens-analysis-orchestrator` capability; do not
represent TraceLens as a substitute for HIP compilation or AMD reference
parity.

## 6. Handoff with honest evidence

Report:

```text
source and target revisions
target GPU/gfx/OS/driver/ROCm
files scanned and translated
HIPIFY tool, version, warnings, unsupported and experimental mappings
manual changes and unresolved findings
build/link commands and outcomes
reference corpus, tolerances, NVIDIA/CPU baseline, and AMD outcomes
profile commands and measurements
evidence level reached
HOLD(...) items, rollback, and next smallest falsifier
```

Never generalize the AMD blog's example conversion percentage to another
codebase. Never claim that source portability means one binary runs on both
vendors. Never promote `translated` or `compiled` to `reference-matched`.
