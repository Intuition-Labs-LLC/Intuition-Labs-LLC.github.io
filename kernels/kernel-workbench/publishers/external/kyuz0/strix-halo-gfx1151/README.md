# Kyuz0 Strix Halo translator toolchain

This adapter makes Kyuz0's Strix Halo toolbox a selectable Kernel Workbench
toolchain. It does not vendor or relicense the upstream project. Verify the
current upstream license and notice requirements before redistribution.

- Toolbox publisher and maintainer: Kyuz0
- Toolbox source: <https://github.com/kyuz0/amd-strix-halo-toolboxes>
- Target scope: AMD Strix Halo, `gfx1151`, CUDA-to-HIP
- Profile: `profile.kyuz0.cuda-to-hip.strix-halo-gfx1151`
- Default scope: `cuda-to-hip@amd-strix-halo-gfx1151@container`
- Execution: digest-pinned container with only the named AMD device nodes
- Claim ceiling: the bundled small integer witness was translated, compiled,
  linked, executed, reference-matched, and profiled on the observed target.

Kyuz0's role is limited to his toolbox. Intuition Labs LLC owns Kernel
Workbench, the adapter, immutable selection pointer, procedure, schemas,
planner, tests, local acceptance receipt, and evidence boundary. Future profile
changes are registry edits; the planner contains no publisher-specific branch.
