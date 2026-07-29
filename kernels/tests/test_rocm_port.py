from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PLUGIN_ROOT / "skills" / "port-cuda-to-hip" / "scripts" / "rocm_port.py"
REAL_SMOKE = PLUGIN_ROOT / "skills" / "port-cuda-to-hip" / "scripts" / "real_smoke.py"
SCRIPTS_ROOT = REAL_SMOKE.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
REAL_SMOKE_SPEC = importlib.util.spec_from_file_location("cuda_hip_real_smoke", REAL_SMOKE)
assert REAL_SMOKE_SPEC is not None and REAL_SMOKE_SPEC.loader is not None
REAL_SMOKE_MODULE = importlib.util.module_from_spec(REAL_SMOKE_SPEC)
REAL_SMOKE_SPEC.loader.exec_module(REAL_SMOKE_MODULE)


class RocmPortTests(unittest.TestCase):
    def run_cli(
        self,
        *arguments: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
        )

    def make_cuda_tree(self, root: Path) -> Path:
        source = root / "cuda-src"
        source.mkdir()
        (source / "kernel.cu").write_text(
            """#include <cuda_runtime.h>
#include <cublas_v2.h>
__global__ __launch_bounds__(256, 2) void reduce(float *out) {
  unsigned lane = threadIdx.x & 31;
  float value = __shfl_sync(0xffffffff, out[threadIdx.x], lane);
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  atomicAdd(out, value);
  __syncthreads();
}
""",
            encoding="utf-8",
        )
        (source / "CMakeLists.txt").write_text(
            "set(CMAKE_CUDA_ARCHITECTURES 90)\n",
            encoding="utf-8",
        )
        (source / "toolchain.cmake").write_text(
            "enable_language(CUDA)\n",
            encoding="utf-8",
        )
        (source / "ordinary.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
        excluded = source / "node_modules"
        excluded.mkdir()
        (excluded / "ignored.cu").write_text("__global__ void ignored() {}\n", encoding="utf-8")
        return source

    def load_stdout(self, result: subprocess.CompletedProcess[str]) -> dict:
        self.assertTrue(result.stdout, result.stderr)
        return json.loads(result.stdout)

    def test_inventory_is_deterministic_and_reports_manual_review_risks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_cuda_tree(Path(directory))
            first = self.run_cli("inventory", str(source))
            second = self.run_cli("inventory", str(source))

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout, second.stdout)
            report = self.load_stdout(first)
            self.assertEqual(report["schema"], "cuda-hip-inventory/v1")
            self.assertEqual(report["evidence"], "scanned")
            self.assertEqual(report["summary"]["source_files"], 1)
            self.assertEqual(
                [item["path"] for item in report["files"]],
                ["CMakeLists.txt", "kernel.cu", "toolchain.cmake"],
            )
            risks = {finding["risk_id"] for finding in report["findings"]}
            self.assertIn("inline-ptx-or-assembly", risks)
            self.assertIn("warp-width-or-lane-mask", risks)
            self.assertIn("two-argument-launch-bounds", risks)
            self.assertIn("cuda-library", risks)
            self.assertIn("nvidia-build-flag", risks)
            self.assertNotIn("ignored.cu", {item["path"] for item in report["files"]})
            self.assertNotIn("float value", first.stdout)

    def test_scan_bounds_hold_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_cuda_tree(Path(directory))
            result = self.run_cli("inventory", str(source), "--max-files", "1")

            self.assertEqual(result.returncode, 2)
            report = self.load_stdout(result)
            self.assertEqual(report["disposition"], "HOLD")
            self.assertEqual(report["reason"], "scan_file_limit_exceeded")

    def test_inventory_includes_hip_candidates_for_residual_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "candidate.hip").write_text(
                """#include <hip/hip_runtime.h>
__global__ void candidate(float *value) {
  atomicAdd(value, 1.0f);
}
""",
                encoding="utf-8",
            )

            result = self.run_cli("inventory", str(source))

            self.assertEqual(result.returncode, 0, result.stderr)
            report = self.load_stdout(result)
            self.assertEqual(report["summary"]["source_files"], 1)
            self.assertEqual(report["summary"]["cuda_source_files"], 0)
            self.assertEqual(report["summary"]["hip_source_files"], 1)
            self.assertEqual(report["files"][0]["path"], "candidate.hip")
            self.assertEqual(report["files"][0]["dialect"], "hip")
            self.assertIn(
                "atomic-or-memory-order",
                {finding["risk_id"] for finding in report["findings"]},
            )

    def test_inventory_detects_cuda_api_only_cpp_and_header_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "runtime.cpp").write_text(
                "void allocate(void **value) { cudaMalloc(value, 4); }\n",
                encoding="utf-8",
            )
            (source / "driver.h").write_text(
                "inline void context() { cuCtxCreate(nullptr, 0, 0); }\n",
                encoding="utf-8",
            )

            result = self.run_cli("inventory", str(source))

            self.assertEqual(result.returncode, 0, result.stderr)
            report = self.load_stdout(result)
            self.assertEqual(report["summary"]["cuda_source_files"], 2)
            self.assertEqual(
                {item["path"] for item in report["files"]},
                {"driver.h", "runtime.cpp"},
            )
            mappings = {item["cuda"]: item for item in report["api_references"]}
            self.assertEqual(mappings["cudaMalloc"]["hip"]["names"], ["hipMalloc"])
            self.assertIn("mapped", mappings["cudaMalloc"]["hip"]["support"])
            self.assertIn("cuCtxCreate", mappings)
            self.assertEqual(report["api_catalog"]["schema"], "hipify-api-catalog/v1")

    def test_inventory_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            report_path = root / "inventory.json"
            report_path.write_text("preserve me\n", encoding="utf-8")

            result = self.run_cli(
                "inventory",
                str(source),
                "--output",
                str(report_path),
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(self.load_stdout(result)["reason"], "inventory_output_already_exists")
            self.assertEqual(report_path.read_text(encoding="utf-8"), "preserve me\n")

    def test_inventory_output_is_new_and_outside_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            report_path = root / "inventory.json"

            written = self.run_cli(
                "inventory",
                str(source),
                "--output",
                str(report_path),
            )
            nested = self.run_cli(
                "inventory",
                str(source),
                "--output",
                str(source / "inventory.json"),
            )

            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["evidence"], "scanned")
            self.assertEqual(nested.returncode, 2)
            self.assertEqual(
                self.load_stdout(nested)["reason"],
                "inventory_output_must_be_outside_source",
            )

    def test_escaping_symlink_holds_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            outside = root / "outside.cu"
            outside.write_text("__global__ void outside() {}\n", encoding="utf-8")
            try:
                (source / "escape.cu").symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")

            result = self.run_cli("inventory", str(source))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(self.load_stdout(result)["reason"], "source_symlink_escapes_root")

    def make_fake_translator(self, root: Path, name: str) -> Path:
        binary = root / name
        binary.write_text(
            """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "fake hipify 1.0"
  exit 0
fi
last=""
output=""
previous=""
for argument in "$@"; do
  if [ "$previous" = "-o" ]; then
    output="$argument"
    previous=""
    continue
  fi
  if [ "$argument" = "-o" ]; then
    previous="-o"
    continue
  fi
  last="$argument"
done
if [ -n "$RACE_OUTPUT" ]; then
  /bin/mkdir "$RACE_OUTPUT"
  /bin/printf 'preserve\\n' > "$RACE_OUTPUT/owner.txt"
fi
if [ -n "$output" ]; then
  {
    echo '#include <hip/hip_runtime.h>'
    /bin/sed 's/cudaMalloc/hipMalloc/g' "$last"
  } > "$output"
else
  echo '#include <hip/hip_runtime.h>'
  /bin/sed 's/cudaMalloc/hipMalloc/g' "$last"
fi
echo 'fake diagnostic' >&2
""",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def test_conversion_is_isolated_and_published_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            original = (source / "kernel.cu").read_bytes()
            (source / "existing.hip").write_text(
                "#include <hip/hip_runtime.h>\n__global__ void existing() {}\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-clang")
            output = root / "hip-output"
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)

            result = self.run_cli(
                "convert",
                str(source),
                str(output),
                "--tool",
                "hipify-clang",
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = self.load_stdout(result)
            self.assertEqual(report["evidence"], "translated")
            self.assertEqual((source / "kernel.cu").read_bytes(), original)
            self.assertTrue((output / "kernel.hip").is_file())
            self.assertFalse((output / "existing.hip").exists())
            self.assertTrue((output / "diagnostics" / "kernel.hip.stderr.txt").is_file())
            manifest = json.loads(
                (output / "cuda-hip-conversion-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["evidence"], "translated")
            self.assertEqual(manifest["files"][0]["input_path"], "kernel.cu")
            self.assertEqual(manifest["tool"]["version"], "fake hipify 1.0")
            self.assertNotIn("__global__", json.dumps(manifest))

    def test_conversion_holds_on_mixed_cuda_hip_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "mixed.cpp").write_text(
                "void mixed(void **p) { cudaMalloc(p, 4); hipFree(*p); }\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-clang")
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)
            output = root / "hip-output"

            result = self.run_cli(
                "convert",
                str(source),
                str(output),
                "--tool",
                "hipify-clang",
                env=environment,
            )

            self.assertEqual(result.returncode, 2)
            report = self.load_stdout(result)
            self.assertEqual(report["reason"], "mixed_cuda_hip_sources_require_manual_split")
            self.assertEqual(report["details"]["paths"], ["mixed.cpp"])
            self.assertFalse(output.exists())

    def test_conversion_preserves_cuda_header_name_for_local_includes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "kernel.cu").write_text(
                '#include "helpers.cuh"\n__global__ void kernel() { helper(); }\n',
                encoding="utf-8",
            )
            (source / "helpers.cuh").write_text(
                "__device__ inline void helper() {}\n",
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-clang")
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)
            output = root / "hip-output"

            result = self.run_cli(
                "convert",
                str(source),
                str(output),
                "--tool",
                "hipify-clang",
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "kernel.hip").is_file())
            self.assertTrue((output / "helpers.cuh").is_file())
            self.assertIn(
                '#include "helpers.cuh"',
                (output / "kernel.hip").read_text(encoding="utf-8"),
            )

    def test_conversion_refuses_concurrent_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-clang")
            output = root / "hip-output"
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)
            environment["RACE_OUTPUT"] = str(output)

            result = self.run_cli(
                "convert",
                str(source),
                str(output),
                "--tool",
                "hipify-clang",
                env=environment,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(self.load_stdout(result)["reason"], "output_root_already_exists")
            self.assertEqual((output / "owner.txt").read_text(encoding="utf-8"), "preserve\n")
            self.assertFalse((output / "kernel.hip").exists())

    def test_conversion_holds_on_unbounded_or_unidentified_translator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            excessive = fake_bin / "hipify-clang"
            excessive.write_text(
                """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "fake hipify 1.0"
  exit 0
fi
echo 'translation output exceeds the deliberately tiny bound'
""",
                encoding="utf-8",
            )
            excessive.chmod(excessive.stat().st_mode | stat.S_IXUSR)
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)

            bounded = self.run_cli(
                "convert",
                str(source),
                str(root / "bounded-output"),
                "--tool",
                "hipify-clang",
                "--max-output-bytes",
                "8",
                env=environment,
            )
            self.assertEqual(bounded.returncode, 2)
            self.assertEqual(
                self.load_stdout(bounded)["reason"],
                "translation_output_limit_exceeded",
            )
            self.assertFalse((root / "bounded-output").exists())

            excessive.write_text(
                """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "fake hipify 1.0"
  exit 0
fi
output=
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    output=$1
  fi
  shift
done
/usr/bin/dd if=/dev/zero of="$output" bs=1024 count=2 2>/dev/null
""",
                encoding="utf-8",
            )
            oversized_file = self.run_cli(
                "convert",
                str(source),
                str(root / "oversized-file"),
                "--tool",
                "hipify-clang",
                "--max-output-bytes",
                "8",
                env=environment,
            )
            self.assertEqual(oversized_file.returncode, 2)
            oversized_receipt = self.load_stdout(oversized_file)
            self.assertEqual(
                oversized_receipt["reason"],
                "translation_output_limit_exceeded",
            )
            self.assertEqual(
                oversized_receipt["details"]["stream"],
                "output_file",
            )
            self.assertFalse((root / "oversized-file").exists())

            excessive.write_text(
                """#!/bin/sh
if [ "$1" = "--version" ]; then
  exit 9
fi
echo 'candidate'
""",
                encoding="utf-8",
            )
            unidentified = self.run_cli(
                "convert",
                str(source),
                str(root / "unidentified"),
                "--tool",
                "hipify-clang",
                env=environment,
            )
            self.assertEqual(unidentified.returncode, 2)
            self.assertEqual(
                self.load_stdout(unidentified)["reason"],
                "translator_version_probe_failed",
            )
            self.assertFalse((root / "unidentified").exists())

    def test_auto_does_not_silently_fall_back_to_hipify_perl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-perl")
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)

            result = self.run_cli(
                "convert",
                str(source),
                str(root / "hip-output"),
                env=environment,
            )

            self.assertEqual(result.returncode, 2)
            report = self.load_stdout(result)
            self.assertEqual(report["reason"], "semantic_translator_unavailable")
            self.assertTrue(report["details"]["hipify_perl_present"])

    def test_existing_or_in_source_output_holds_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_cuda_tree(root)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self.make_fake_translator(fake_bin, "hipify-clang")
            environment = dict(os.environ)
            environment["PATH"] = str(fake_bin)

            existing = root / "existing"
            existing.mkdir()
            collision = self.run_cli(
                "convert",
                str(source),
                str(existing),
                "--tool",
                "hipify-clang",
                env=environment,
            )
            nested = self.run_cli(
                "convert",
                str(source),
                str(source / "hip-output"),
                "--tool",
                "hipify-clang",
                env=environment,
            )

            self.assertEqual(self.load_stdout(collision)["reason"], "output_root_already_exists")
            self.assertEqual(self.load_stdout(nested)["reason"], "output_root_must_be_outside_source")

    def test_real_smoke_holds_without_supported_cuda_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REAL_SMOKE),
                    "--cuda-path",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(
                json.loads(result.stdout)["reason"],
                "smoke_cuda_header_missing",
            )

    def test_profile_evidence_requires_named_gpu_kernel_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            valid_document = {
                "rocprofiler-sdk-tool": [
                    {
                        "agents": [
                            {
                                "gfx_target_version": 110501,
                                "id": {"handle": 42},
                                "name": "test AMD GPU",
                                "type": 2,
                                "vendor_id": 4098,
                            }
                        ],
                        "buffer_records": {
                            "hip_api": [{}],
                            "kernel_dispatch": [
                                {
                                    "dispatch_info": {
                                        "agent_id": {"handle": 42},
                                        "dispatch_id": 7,
                                        "kernel_id": 9,
                                    },
                                    "end_timestamp": 20,
                                    "start_timestamp": 10,
                                }
                            ],
                            "memory_copy": [],
                        },
                        "kernel_symbols": [
                            {
                                "formatted_kernel_name": "add_bias(int const*, int*, int)",
                                "kernel_id": 9,
                                "truncated_kernel_name": "add_bias",
                            }
                        ],
                    }
                ]
            }
            profile.write_text(json.dumps(valid_document), encoding="utf-8")

            receipt = REAL_SMOKE_MODULE.validate_profile_trace(profile)

            self.assertEqual(receipt["agent"]["handle"], 42)
            self.assertEqual(receipt["dispatch_id"], 7)
            self.assertEqual(receipt["kernel"]["name"], "add_bias(int const*, int*, int)")
            self.assertEqual(receipt["record_counts"]["kernel_dispatch"], 1)

            valid_document["rocprofiler-sdk-tool"][0]["buffer_records"][
                "kernel_dispatch"
            ] = []
            profile.write_text(json.dumps(valid_document), encoding="utf-8")
            with self.assertRaises(REAL_SMOKE_MODULE.SmokeHold) as context:
                REAL_SMOKE_MODULE.validate_profile_trace(profile)
            self.assertEqual(
                context.exception.reason,
                "smoke_profile_kernel_dispatch_missing",
            )


if __name__ == "__main__":
    unittest.main()
