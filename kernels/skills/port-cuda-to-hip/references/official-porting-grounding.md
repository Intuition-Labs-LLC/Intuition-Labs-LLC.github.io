# Official CUDA to HIP/ROCm grounding

Grounding snapshot: 2026-07-28. Re-check the live pages and local tool versions
for every real port because ROCm components, GPU support, operating systems,
CUDA inputs, and HIPIFY dependencies drift independently.

## Source index

- AMD, “From CUDA to AMD ROCm Software Without Starting Over,” 2026-07-23:
  <https://www.amd.com/en/blogs/2026/from-cuda-to-amd-rocm-software-without-starting-over.html>
- ROCm release notes:
  <https://rocm.docs.amd.com/en/latest/about/release-notes.html>
- ROCm compatibility matrix:
  <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- HIPIFY overview:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/>
- `hipify-clang` usage and commands:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-clang.html>
  and
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hipify-clang-cmd.html>
- `hipify-perl` usage and commands:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/how-to/hipify-perl.html>
  and
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hipify-perl-cmd.html>
- HIPIFY supported CUDA APIs:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/supported_apis.html>
- CUDA APIs supported by HIP:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hip_supported_apis.html>
- CUDA APIs supported by ROC:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/roc_supported_apis.html>
- CUDA APIs supported jointly by HIP and ROC:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hip_roc_supported_apis.html>
- Building `hipify-clang` on Linux:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/building/build-hipify-clang-linux.html>
- Building `hipify-perl`:
  <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/building/build-hipify-perl.html>
- HIP CUDA porting guide:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html>
- HIP kernel C++ support:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/kernel_language_cpp_support.html>
- HIP cooperative groups:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html>
- HIP performance guidance:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html>
- HIP debugging and ROCgdb:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html>
  and <https://rocm.docs.amd.com/projects/ROCgdb/en/latest/>
- ROCprofiler SDK `rocprofv3`:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
- HIPCC:
  <https://rocm.docs.amd.com/projects/HIPCC/en/latest/>
- NVIDIA CUDA Programming Guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- NVIDIA CUDA Best Practices Guide:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA CUDA Toolkit release notes:
  <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/>

## Stable distinctions

### Observed 2026-07-28 versions

The current AMD compatibility matrix identified ROCm Core SDK 7.14.0, dated
2026-07-16, with HIP 7.14, HIPIFY 7.14, and LLVM 23.0.0. The current NVIDIA
release notes identified CUDA Toolkit 13.3 Update 1. At the same time, the
HIPIFY `hipify-clang` dependency page documented CUDA input support only through
12.9.1 and recommended Clang 21.1.6. This is a documentation/tool-support
mismatch, not permission to assume CUDA 13.x support. Probe the installed tools
and hold any source/tool tuple outside the live documented range.

### Translation tools

`hipify-clang` parses CUDA with Clang and applies AST-aware mappings. Prefer it
for production sources. It requires correct CUDA input, CUDA headers, and the
real include paths, defines, and language options. A compilation database is
the safest way to preserve those inputs.

`hipify-perl` is a regex-based generated script. It can scan without CUDA or
Clang, but AMD documents limitations around macro expansion, namespaces,
templates, host/device call differentiation, header injection, and complicated
argument lists. Treat it as advisory or as an explicitly accepted fallback.

`hipcc` is a compiler driver, not a translator. `hipconfig` reports the selected
HIP platform and toolchain. HIP source portability still requires separate
vendor builds and validation.

HIPIFY maps many CUDA runtime, driver, device, RTC, and library APIs, but support
is API-by-API. A category in the mapping table is not a guarantee that every
entry has an exact, production-ready equivalent.

### Mapping ontology

HIPIFY itself owns the canonical mapping export. `hipify-clang --md
--doc-format=full --doc-roc=joint` generates the full joint Markdown
documentation; replacing `--md` with `--csv` generates the CSV form. The CSV
documents use repeated typed tables whose columns include CUDA/HIP/ROC symbols,
added/deprecated/changed/removed versions, unsupported CUDA-version ranges, and
experimental markers. The plugin normalizes all emitted CSV documents into one
deterministic object rather than treating rendered web pages as an executable
type system.

`--doc-roc=separate` produces HIP and ROC documentation separately.
`--doc-roc=joint` preserves both targets in the same generated table where the
domain has both mappings. An empty target is an explicit `unmapped` or
`not-emitted` state; it must never be promoted to identity or inferred support.

### Manual review obligations

- Do not translate NVIDIA PTX as if it were portable assembly. HIP inline
  assembly is AMD architecture-specific and still creates a portability burden.
- Do not assume CUDA warp width 32. Use `warpSize` and runtime/feature queries;
  verify lane-mask types and wave-sensitive algorithms on the target.
- Reconcile CUDA and HIP memory ordering, atomic scopes, barriers, divergent
  synchronization, cooperative groups, and independent-thread-scheduling
  assumptions.
- Convert two-argument `__launch_bounds__` semantically. HIP's second parameter
  is warps per execution unit, not CUDA blocks per SM, and differs by AMD
  execution mode. `amdclang++` does not accept `nvcc --maxregcount`.
- Replace `__CUDA_ARCH__` numeric thresholds with the required HIP feature
  query or an explicit target contract.
- Review driver/context/module behavior, nested headers, generated sources,
  build flags, textures/surfaces, graphs, async-copy features, and multi-GPU
  behavior.
- Select `hip*` versus `roc*` libraries deliberately. `roc*` libraries can be
  more AMD-specific and performant; they can also narrow backend neutrality.
- Treat unified-memory and copy removal as a later target-specific optimization.
  Shared-memory APUs do not make that rewrite universally semantics-preserving.

### Evidence ladder

Use these terms literally:

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| `scanned` | Bounded static inventory completed | Translator support |
| `translated` | HIPIFY emitted a candidate | Compile or correctness |
| `compiled` | Named units compiled for a target | Link or runtime behavior |
| `linked` | Named application linked | Device execution |
| `executed-on-amd` | Named binary ran on observed AMD hardware | Reference parity |
| `reference-matched` | Frozen cases met declared tolerances | Broad correctness |
| `profiled` | Named trace/counters were captured | Performance acceptance |

Capture the NVIDIA or CPU reference before changing semantics. Include edge
sizes, non-multiple workgroups, multidimensional launches, aliasing, streams,
atomics, and wave-sensitive cases. Compare integers exactly and floating values
with algorithm-derived tolerances; do not substitute bit equality or a generic
epsilon without justification.

Optimize only after the correctness corpus passes on supported AMD hardware.
Use `rocprofv3` to establish and compare AMD measurements. TraceLens is a
separate AMD PyTorch-profiler analysis workflow and applies only when its trace
input contract is satisfied.

## Snapshot-specific drift warning

Never hard-code the snapshot above as compatibility policy. Record local
`--version` and `--versions` output, then consult the live compatibility and
supported-API pages before translating a real project.
