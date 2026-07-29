from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PLUGIN_ROOT
    / "skills"
    / "port-cuda-to-hip"
    / "scripts"
    / "hipify_api_catalog.py"
)
BUNDLED_CATALOG = (
    PLUGIN_ROOT / "skills" / "port-cuda-to-hip" / "assets" / "hipify-api-catalog.json"
)


class HipifyApiCatalogTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def make_exports(self, root: Path) -> Path:
        exports = root / "exports"
        exports.mkdir()
        (exports / "CUDA_Runtime_API_functions_supported_by_HIP.csv").write_text(
            """<head>
<meta charset="UTF-8">
</head>

1. Memory Management

CUDA,A,D,C,R,HIP,A,D,C,R,U,E
cudaMalloc,,,,,hipMalloc,1.6.0,,,,,
cudaFuture,13.0,,,,,,,,,13.0,

2. Experimental

CUDA,A,D,C,R,HIP,A,D,C,R,U,E
cudaExperimental,12.0,,,,hipExperimental,7.0.0,,,,,yes
""",
            encoding="utf-8",
        )
        (exports / "CUBLAS_API_supported_by_HIP_and_ROC.csv").write_text(
            """CUBLAS API supported by HIP and ROC

1. Functions

CUDA,A,D,C,R,HIP,A,D,C,R,U,E,ROC,A,D,C,R,E
cublasCreate_v2,,,,,hipblasCreate,3.10.0,,,,,,rocblas_create_handle,3.8.0,,,,
cublasFuture,13.0,,,,,,,,,,,,,,,,
""",
            encoding="utf-8",
        )
        return exports

    def test_normalizes_full_joint_exports_into_one_typed_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = self.make_exports(root)
            output = root / "catalog.json"

            result = self.run_cli(
                "normalize",
                str(exports),
                str(output),
                "--tool-version",
                "AMD LLVM version 23.0.0git",
                "--completeness",
                "partial",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["evidence"], "partial-static-mapping")
            catalog = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(catalog["schema"], "hipify-api-catalog/v1")
            self.assertEqual(catalog["generation"]["completeness"], "partial")
            self.assertEqual(catalog["summary"]["cuda_symbols"], 5)
            self.assertEqual(catalog["summary"]["mapping_rows"], 5)
            malloc = catalog["apis"]["cudaMalloc"]["variants"][0]
            self.assertEqual(malloc["targets"]["hip"]["name"], "hipMalloc")
            self.assertEqual(malloc["targets"]["hip"]["support"], "mapped")
            self.assertEqual(malloc["targets"]["roc"]["support"], "not-emitted")
            future = catalog["apis"]["cudaFuture"]["variants"][0]
            self.assertEqual(future["targets"]["hip"]["support"], "unmapped")
            experimental = catalog["apis"]["cudaExperimental"]["variants"][0]
            self.assertEqual(experimental["targets"]["hip"]["support"], "experimental")
            blas = catalog["apis"]["cublasCreate_v2"]["variants"][0]
            self.assertEqual(blas["targets"]["roc"]["name"], "rocblas_create_handle")
            self.assertEqual(
                catalog["generation"]["command"],
                "hipify-clang --csv --doc-format=full --doc-roc=joint",
            )

    def test_lookup_can_project_one_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = self.make_exports(root)
            output = root / "catalog.json"
            normalized = self.run_cli(
                "normalize",
                str(exports),
                str(output),
                "--tool-version",
                "fake-version",
                "--completeness",
                "partial",
            )
            self.assertEqual(normalized.returncode, 0, normalized.stderr)

            result = self.run_cli(
                "lookup",
                str(output),
                "cublasCreate_v2",
                "--backend",
                "roc",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lookup = json.loads(result.stdout)
            self.assertEqual(
                lookup["api"]["variants"][0]["target"]["name"],
                "rocblas_create_handle",
            )
            self.assertNotIn("targets", lookup["api"]["variants"][0])

    def test_normalize_refuses_existing_output_and_unknown_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = self.make_exports(root)
            output = root / "catalog.json"
            output.write_text("preserve\n", encoding="utf-8")

            existing = self.run_cli(
                "normalize",
                str(exports),
                str(output),
                "--tool-version",
                "fake-version",
                "--completeness",
                "partial",
            )
            self.assertEqual(existing.returncode, 2)
            self.assertEqual(
                json.loads(existing.stdout)["reason"],
                "catalog_output_already_exists",
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve\n")

            invalid_exports = root / "invalid"
            invalid_exports.mkdir()
            (invalid_exports / "unsupported.csv").write_text(
                "source,target\ncudaMalloc,hipMalloc\n",
                encoding="utf-8",
            )
            invalid = self.run_cli(
                "normalize",
                str(invalid_exports),
                str(root / "invalid.json"),
                "--tool-version",
                "fake-version",
                "--completeness",
                "partial",
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(
                json.loads(invalid.stdout)["reason"],
                "catalog_csv_header_not_found",
            )

    def test_full_normalization_requires_complete_joint_source_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = self.make_exports(root)

            result = self.run_cli(
                "normalize",
                str(exports),
                str(root / "catalog.json"),
                "--tool-version",
                "fake-version",
            )

            self.assertEqual(result.returncode, 2)
            receipt = json.loads(result.stdout)
            self.assertEqual(
                receipt["reason"],
                "catalog_full_joint_source_set_mismatch",
            )
            self.assertIn("CUDA_Device_API_supported_by_HIP.csv", receipt["details"]["missing"])

    def test_normalize_rejects_malformed_mapping_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = root / "exports"
            exports.mkdir()
            (exports / "broken.csv").write_text(
                """CUDA,A,D,C,R,HIP,A,D,C,R,U,E
cudaMalloc,,,,,hipMalloc,1.6.0,,,,,
cudaBroken,too,few
""",
                encoding="utf-8",
            )

            result = self.run_cli(
                "normalize",
                str(exports),
                str(root / "catalog.json"),
                "--tool-version",
                "fake-version",
                "--completeness",
                "partial",
            )

            self.assertEqual(result.returncode, 2)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["reason"], "catalog_csv_malformed_mapping_row")
            self.assertEqual(receipt["details"]["line"], 3)

    def test_validate_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"schema":"wrong","apis":{}}\n', encoding="utf-8")

            result = self.run_cli("validate", str(path))

            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["reason"], "catalog_schema_invalid")

    def test_validate_recomputes_summary_and_nested_support_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = self.make_exports(root)
            generated = root / "generated.json"
            normalized = self.run_cli(
                "normalize",
                str(exports),
                str(generated),
                "--tool-version",
                "fake-version",
                "--completeness",
                "partial",
            )
            self.assertEqual(normalized.returncode, 0, normalized.stderr)
            catalog = json.loads(generated.read_text(encoding="utf-8"))

            catalog["summary"]["mapping_rows"] += 1
            summary_tampered = root / "summary-tampered.json"
            summary_tampered.write_text(json.dumps(catalog), encoding="utf-8")
            summary_result = self.run_cli("validate", str(summary_tampered))
            self.assertEqual(summary_result.returncode, 2)
            self.assertEqual(
                json.loads(summary_result.stdout)["reason"],
                "catalog_summary_inconsistent",
            )

            catalog = json.loads(generated.read_text(encoding="utf-8"))
            catalog["apis"]["cudaMalloc"]["variants"][0]["targets"]["hip"]["support"] = (
                "experimental"
            )
            nested_tampered = root / "nested-tampered.json"
            nested_tampered.write_text(json.dumps(catalog), encoding="utf-8")
            nested_result = self.run_cli("validate", str(nested_tampered))
            self.assertEqual(nested_result.returncode, 2)
            self.assertEqual(
                json.loads(nested_result.stdout)["reason"],
                "catalog_support_state_inconsistent",
            )

    def test_bundled_real_catalog_is_valid_and_maps_cuda_malloc(self) -> None:
        validated = self.run_cli("validate", str(BUNDLED_CATALOG))
        lookup = self.run_cli(
            "lookup",
            str(BUNDLED_CATALOG),
            "cudaMalloc",
            "--backend",
            "hip",
        )

        self.assertEqual(validated.returncode, 0, validated.stderr)
        summary = json.loads(validated.stdout)["summary"]
        self.assertGreater(summary["cuda_symbols"], 9000)
        self.assertGreater(summary["mapping_rows"], 9000)
        self.assertEqual(lookup.returncode, 0, lookup.stderr)
        self.assertEqual(
            json.loads(lookup.stdout)["api"]["variants"][0]["target"]["name"],
            "hipMalloc",
        )


if __name__ == "__main__":
    unittest.main()
