from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_ROOT = PLUGIN_ROOT / "kernel-workbench"
SCRIPT = WORKBENCH_ROOT / "scripts" / "kernel_workbench.py"
EXAMPLE = WORKBENCH_ROOT / "examples" / "cuda-to-hip-strix-halo.v1.json"
REGISTRY = WORKBENCH_ROOT / "registry" / "kernel-workbench-registry.v1.json"
SCHEMA = WORKBENCH_ROOT / "schemas" / "kernel-work-order.v1.schema.json"


class KernelWorkbenchTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def load_example(self) -> dict:
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def run_order(self, order: dict, command: str = "plan") -> tuple[subprocess.CompletedProcess[str], dict]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.json"
            path.write_text(json.dumps(order), encoding="utf-8")
            result = self.run_cli(command, str(path))
            return result, json.loads(result.stdout)

    def test_registry_schema_and_example_validate(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        result = self.run_cli("validate", str(EXAMPLE))
        report = json.loads(result.stdout)

        self.assertEqual(registry["schema"], "kernel-workbench-registry/v1")
        self.assertEqual(schema["title"], "KernelWorkOrderV1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["disposition"], "VALID")
        self.assertRegex(report["registry_digest"], r"^sha256:[0-9a-f]{64}$")

    def test_registry_and_plan_bytes_are_deterministic(self) -> None:
        registry_first = self.run_cli("registry")
        registry_second = self.run_cli("registry")
        plan_first = self.run_cli("plan", str(EXAMPLE))
        plan_second = self.run_cli("plan", str(EXAMPLE))

        self.assertEqual(registry_first.returncode, 0, registry_first.stderr)
        self.assertEqual(plan_first.returncode, 0, plan_first.stderr)
        self.assertEqual(registry_first.stdout, registry_second.stdout)
        self.assertEqual(plan_first.stdout, plan_second.stdout)
        self.assertRegex(
            json.loads(plan_first.stdout)["plan_digest"],
            r"^sha256:[0-9a-f]{64}$",
        )

    def test_intuition_owns_workbench_and_scoped_default_resolves_to_kyuz0_toolbox(self) -> None:
        result = self.run_cli("plan", str(EXAMPLE))
        plan = json.loads(result.stdout)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            plan["work_owner"],
            {
                "id": "owner.intuition-labs.kernel-workbench",
                "name": "Intuition Labs LLC",
                "role": "kernel-workbench-owner",
                "scope": "procedure-schemas-planner-adapters-tests-and-evidence-contract",
                "source_url": (
                    "https://github.com/Intuition-Labs-LLC/"
                    "Intuition-Labs-LLC.github.io/tree/main/kernels"
                ),
            },
        )
        self.assertEqual(
            plan["runtime_profile"]["id"],
            "profile.kyuz0.cuda-to-hip.strix-halo-gfx1151",
        )
        self.assertEqual(
            plan["runtime_profile"]["publisher"],
            {
                "id": "publisher.kyuz0",
                "name": "Kyuz0",
                "ownership_party": "external",
                "role": "external-toolbox-publisher",
                "scope": "kyuz0-strix-halo-toolboxes-only",
            },
        )
        self.assertNotEqual(
            plan["runtime_profile"]["publisher"]["role"],
            plan["work_owner"]["role"],
        )
        self.assertEqual(plan["trajectory"]["selected"], "safe_reachable")
        self.assertEqual(plan["disposition"], "READY")

    def test_explicit_operator_profile_switch_requires_no_code_change(self) -> None:
        order = self.load_example()
        order["runtime_profile"] = "profile.operator.cuda-to-hip.native-rocm"
        order["target"]["execution_topology"] = "native"
        result, plan = self.run_order(order)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            plan["runtime_profile"]["translator_toolchain"]["definition_ref"],
            "operator://installed-rocm",
        )
        self.assertEqual(
            plan["runtime_profile"]["publisher"]["role"],
            "operator-toolchain-owner",
        )

    def test_unknown_profile_holds_instead_of_using_default(self) -> None:
        order = self.load_example()
        order["runtime_profile"] = "profile.unknown"
        result, report = self.run_order(order)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["disposition"], "HOLD")
        self.assertEqual(report["reason"], "runtime_profile_unknown:profile.unknown")

    def test_registry_and_catalog_pin_mismatches_hold(self) -> None:
        for pin in ("registry_digest", "catalog_digest"):
            with self.subTest(pin=pin):
                order = self.load_example()
                order["pins"][pin] = f"sha256:{'0' * 64}"
                result, report = self.run_order(order)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(report["reason"], f"{pin}_mismatch")

    def test_registry_holds_when_its_catalog_pointer_digest_is_stale(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry["translators"][0]["catalog_digest"] = f"sha256:{'0' * 64}"
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            result = self.run_cli("--registry", str(registry_path), "registry")
            report = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            report["reason"],
            "registry_translator_catalog_digest_mismatch:cuda-to-hip",
        )

    def test_described_future_axis_is_not_an_implemented_adapter(self) -> None:
        order = self.load_example()
        order["target"]["processor_plane"] = "npu"
        result, report = self.run_order(order)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(report["reason"], "target_tuple_not_implemented")

    def test_leap_target_falls_back_until_correspondence_gate(self) -> None:
        order = self.load_example()
        order["trajectory"] = "leap_target"
        order["optimization_scopes"] = ["throughput"]
        result, plan = self.run_order(order)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(plan["disposition"], "SAFE_FALLBACK")
        self.assertEqual(plan["trajectory"]["selected"], "safe_reachable")
        self.assertTrue(plan["trajectory"]["fallback"])
        self.assertNotIn(
            "optimize-scoped",
            [stage["stage"] for stage in plan["stages"]],
        )
        self.assertEqual(
            plan["obligations"],
            [
                "HOLD(leap_target_missing:compiled,linked,executed-on-amd,"
                "reference-matched)"
            ],
        )

    def test_leap_target_is_admitted_only_after_required_evidence(self) -> None:
        order = self.load_example()
        order["trajectory"] = "leap_target"
        order["optimization_scopes"] = ["throughput", "wave-semantics"]
        order["evidence_available"] = [
            "compiled",
            "linked",
            "executed-on-amd",
            "reference-matched",
        ]
        result, plan = self.run_order(order)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(plan["disposition"], "READY")
        self.assertEqual(plan["trajectory"]["selected"], "leap_target")
        self.assertIn(
            "optimize-scoped",
            [stage["stage"] for stage in plan["stages"]],
        )

    def test_unknown_fields_and_absolute_source_paths_hold(self) -> None:
        cases = []
        unknown = self.load_example()
        unknown["mechanism"] = "private"
        cases.append((unknown, "work_order_keys_invalid:unknown:mechanism"))
        absolute = self.load_example()
        absolute["source"]["root_ref"] = "/private/source"
        cases.append((absolute, "source_root_ref_must_be_logical"))

        for order, expected in cases:
            with self.subTest(expected=expected):
                result, report = self.run_order(order)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(report["reason"], expected)

    def test_public_workbench_has_no_private_maxwell_mechanism_links(self) -> None:
        forbidden = (
            "/var/home/",
            "/home/",
            "file://",
            "private-runtime://",
            "research_worker/",
        )
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(WORKBENCH_ROOT.rglob("*"))
            if path.is_file() and path.suffix in {".json", ".md", ".py"}
        )
        for term in forbidden:
            with self.subTest(term=term):
                self.assertNotIn(term, text)

    def test_input_order_is_not_mutated_by_test_variants(self) -> None:
        first = self.load_example()
        second = copy.deepcopy(first)
        second["trajectory"] = "leap_target"
        self.assertEqual(first["trajectory"], "safe_reachable")


if __name__ == "__main__":
    unittest.main()
