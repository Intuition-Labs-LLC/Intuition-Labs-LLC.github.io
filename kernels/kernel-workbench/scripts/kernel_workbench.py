#!/usr/bin/env python3
"""Deterministic, content-free Kernel Workbench work-order planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = WORKBENCH_ROOT.parent
DEFAULT_REGISTRY = WORKBENCH_ROOT / "registry" / "kernel-workbench-registry.v1.json"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")

TOP_KEYS = {
    "schema",
    "work_order_id",
    "translator_id",
    "source",
    "target",
    "runtime_profile",
    "trajectory",
    "optimization_scopes",
    "evidence_available",
    "pins",
}
SOURCE_KEYS = {"language", "dialect", "root_ref", "digest"}
TARGET_KEYS = {
    "language",
    "platform",
    "architecture",
    "processor_plane",
    "memory_topology",
    "execution_topology",
}
PIN_KEYS = {"registry_digest", "catalog_digest"}

SAFE_STAGE_REQUIREMENTS = {
    "inventory": [],
    "translate-isolated": ["scanned"],
    "manual-semantic-review": ["translated"],
    "compile-link": ["manual-reviewed"],
    "execute-on-target": ["compiled", "linked"],
    "reference-match": ["executed-on-amd"],
}
LEAP_STAGE_REQUIREMENTS = {
    "profile-baseline": ["reference-matched"],
    "optimize-scoped": ["profiled"],
    "revalidate-reference": ["compiled", "linked", "executed-on-amd"],
}


class WorkbenchError(ValueError):
    """A fail-closed validation or resolution error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label}_must_be_object")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        details = [
            *(f"missing:{item}" for item in missing),
            *(f"unknown:{item}" for item in unknown),
        ]
        raise WorkbenchError(f"{label}_keys_invalid:{','.join(details)}")
    return value


def string(value: Any, label: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise WorkbenchError(f"{label}_invalid")
    return value


def hash_or_none(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise WorkbenchError(f"{label}_invalid")
    return value


def id_index(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise WorkbenchError(f"registry_{label}_invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise WorkbenchError(f"registry_{label}_row_invalid")
        row_id = string(row.get("id"), f"registry_{label}_id")
        if row_id in indexed:
            raise WorkbenchError(f"registry_{label}_duplicate:{row_id}")
        indexed[row_id] = row
    return indexed


def resolve_plugin_ref(value: Any, label: str) -> Path:
    ref = string(value, label, maximum=256)
    if ref.startswith(("/", "~")) or ".." in ref.split("/"):
        raise WorkbenchError(f"{label}_must_be_plugin_relative")
    path = (PLUGIN_ROOT / ref).resolve()
    try:
        path.relative_to(PLUGIN_ROOT)
    except ValueError as error:
        raise WorkbenchError(f"{label}_escapes_plugin") from error
    if not path.is_file():
        raise WorkbenchError(f"{label}_missing:{ref}")
    return path


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return f"sha256:{value.hexdigest()}"


def load_registry(path: Path) -> tuple[dict[str, Any], str]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkbenchError(f"registry_unreadable:{error.__class__.__name__}") from error
    if not isinstance(registry, dict) or registry.get("schema") != "kernel-workbench-registry/v1":
        raise WorkbenchError("registry_schema_invalid")

    translators = id_index(registry.get("translators"), "translators")
    targets = id_index(registry.get("targets"), "targets")
    publishers = id_index(registry.get("publishers"), "publishers")
    toolchains = id_index(registry.get("translator_toolchains"), "toolchains")
    profiles = id_index(registry.get("runtime_profiles"), "profiles")
    defaults = registry.get("defaults_by_scope")
    if not isinstance(defaults, dict) or not defaults:
        raise WorkbenchError("registry_defaults_invalid")

    work_owner = exact_keys(
        registry.get("work_owner"),
        {"id", "name", "role", "scope", "source_url"},
        "registry_work_owner",
    )
    for field in ("id", "name", "role", "scope", "source_url"):
        string(
            work_owner[field],
            f"registry_work_owner_{field}",
            maximum=256,
        )
    if (
        work_owner["id"] != "owner.intuition-labs.kernel-workbench"
        or work_owner["role"] != "kernel-workbench-owner"
        or not work_owner["source_url"].startswith("https://")
    ):
        raise WorkbenchError("registry_work_owner_invalid")

    axes = registry.get("axes")
    if not isinstance(axes, dict):
        raise WorkbenchError("registry_axes_invalid")
    for axis in (
        "source_languages",
        "target_languages",
        "processor_planes",
        "memory_topologies",
        "execution_topologies",
    ):
        id_index(axes.get(axis), f"axis_{axis}")

    implemented_source_languages = axis_ids(registry, "source_languages", implemented_only=True)
    implemented_target_languages = axis_ids(registry, "target_languages", implemented_only=True)
    for translator_id, translator in translators.items():
        if (
            translator.get("status") != "implemented"
            or translator.get("source_language") not in implemented_source_languages
            or translator.get("target_language") not in implemented_target_languages
        ):
            raise WorkbenchError(f"registry_translator_axis_invalid:{translator_id}")
        catalog_path = resolve_plugin_ref(
            translator.get("catalog_ref"),
            f"registry_translator_catalog_ref:{translator_id}",
        )
        catalog_digest = hash_or_none(
            translator.get("catalog_digest"),
            f"registry_translator_catalog_digest:{translator_id}",
        )
        if catalog_digest != file_digest(catalog_path):
            raise WorkbenchError(f"registry_translator_catalog_digest_mismatch:{translator_id}")

    for target_id, target in targets.items():
        if target.get("status") != "implemented":
            raise WorkbenchError(f"registry_target_status_invalid:{target_id}")
        for axis, field in (
            ("processor_planes", "processor_plane"),
            ("memory_topologies", "memory_topology"),
        ):
            if target.get(field) not in axis_ids(registry, axis, implemented_only=True):
                raise WorkbenchError(f"registry_target_axis_invalid:{target_id}:{field}")
        topologies = target.get("execution_topologies")
        if (
            not isinstance(topologies, list)
            or not topologies
            or any(
                item not in axis_ids(registry, "execution_topologies", implemented_only=True)
                for item in topologies
            )
        ):
            raise WorkbenchError(f"registry_target_topology_invalid:{target_id}")

    for publisher_id, publisher in publishers.items():
        if (
            not isinstance(publisher.get("source_url"), str)
            or not publisher["source_url"].startswith("https://")
            or not publisher.get("role")
            or not publisher.get("scope")
            or not publisher.get("ownership_party")
            or not publisher.get("license_disposition")
        ):
            raise WorkbenchError(f"registry_publisher_provenance_invalid:{publisher_id}")

    for toolchain_id, toolchain in toolchains.items():
        if toolchain.get("publisher_ref") not in publishers:
            raise WorkbenchError(f"registry_toolchain_publisher_invalid:{toolchain_id}")
        topology = toolchain.get("execution_topology")
        if topology not in axis_ids(registry, "execution_topologies", implemented_only=True):
            raise WorkbenchError(f"registry_toolchain_topology_invalid:{toolchain_id}")
        definition_ref = string(
            toolchain.get("definition_ref"),
            f"registry_toolchain_definition_ref:{toolchain_id}",
            maximum=256,
        )
        if definition_ref.startswith("operator://"):
            if toolchain.get("publisher_ref") != "publisher.operator":
                raise WorkbenchError(f"registry_toolchain_operator_ref_invalid:{toolchain_id}")
        else:
            resolve_plugin_ref(
                definition_ref,
                f"registry_toolchain_definition_ref:{toolchain_id}",
            )
    for profile_id, profile in profiles.items():
        if profile.get("status") != "implemented":
            raise WorkbenchError(f"registry_profile_status_invalid:{profile_id}")
        if profile.get("publisher_ref") not in publishers:
            raise WorkbenchError(f"registry_profile_publisher_invalid:{profile_id}")
        toolchain_id = profile.get("translator_toolchain_ref")
        if toolchain_id not in toolchains:
            raise WorkbenchError(f"registry_profile_toolchain_invalid:{profile_id}")
        if toolchains[toolchain_id]["publisher_ref"] != profile["publisher_ref"]:
            raise WorkbenchError(f"registry_profile_publisher_mismatch:{profile_id}")
        for translator_id in profile.get("translator_ids", []):
            if translator_id not in translators:
                raise WorkbenchError(f"registry_profile_translator_unknown:{translator_id}")
        for target_id in profile.get("target_ids", []):
            if target_id not in targets:
                raise WorkbenchError(f"registry_profile_target_unknown:{target_id}")

    for scope, profile_id in defaults.items():
        parts = scope.split("@")
        if len(parts) != 3:
            raise WorkbenchError(f"registry_default_scope_invalid:{scope}")
        translator_id, target_id, execution_topology = parts
        if (
            translator_id not in translators
            or target_id not in targets
            or profile_id not in profiles
            or execution_topology not in targets[target_id].get("execution_topologies", [])
        ):
            raise WorkbenchError(f"registry_default_invalid:{scope}")
        profile = profiles[profile_id]
        if translator_id not in profile["translator_ids"] or target_id not in profile["target_ids"]:
            raise WorkbenchError(f"registry_default_profile_scope_invalid:{scope}")
        toolchain = toolchains[profile["translator_toolchain_ref"]]
        if execution_topology != toolchain["execution_topology"]:
            raise WorkbenchError(f"registry_default_topology_invalid:{scope}")

    ladder = registry.get("evidence_ladder")
    scopes = registry.get("optimization_scopes")
    if not isinstance(ladder, list) or len(ladder) != len(set(ladder)):
        raise WorkbenchError("registry_evidence_ladder_invalid")
    if not isinstance(scopes, list) or len(scopes) != len(set(scopes)):
        raise WorkbenchError("registry_optimization_scopes_invalid")
    trajectories = registry.get("trajectories")
    if not isinstance(trajectories, dict) or trajectories.get("default") != "safe_reachable":
        raise WorkbenchError("registry_trajectories_invalid")
    return registry, digest(registry)


def load_work_order(path: Path) -> dict[str, Any]:
    try:
        order = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkbenchError(f"work_order_unreadable:{error.__class__.__name__}") from error
    exact_keys(order, TOP_KEYS, "work_order")
    if order.get("schema") != "kernel-work-order/v1":
        raise WorkbenchError("work_order_schema_invalid")
    work_order_id = string(order.get("work_order_id"), "work_order_id", maximum=96)
    if ID_RE.fullmatch(work_order_id) is None:
        raise WorkbenchError("work_order_id_invalid")
    string(order.get("translator_id"), "translator_id", maximum=64)

    source = exact_keys(order.get("source"), SOURCE_KEYS, "source")
    for key in ("language", "dialect"):
        string(source.get(key), f"source_{key}", maximum=64)
    root_ref = string(source.get("root_ref"), "source_root_ref", maximum=256)
    if root_ref.startswith(("/", "~")) or ".." in root_ref.split("/"):
        raise WorkbenchError("source_root_ref_must_be_logical")
    if hash_or_none(source.get("digest"), "source_digest") is None:
        raise WorkbenchError("source_digest_required")

    target = exact_keys(order.get("target"), TARGET_KEYS, "target")
    for key in TARGET_KEYS:
        string(target.get(key), f"target_{key}", maximum=64)

    profile = order.get("runtime_profile")
    if profile is not None:
        string(profile, "runtime_profile", maximum=96)
    if order.get("trajectory") not in {"safe_reachable", "leap_target"}:
        raise WorkbenchError("trajectory_invalid")

    for field in ("optimization_scopes", "evidence_available"):
        values = order.get(field)
        if not isinstance(values, list) or len(values) > 16 or len(values) != len(set(values)):
            raise WorkbenchError(f"{field}_invalid")
        for value in values:
            string(value, field, maximum=64)

    pins = exact_keys(order.get("pins"), PIN_KEYS, "pins")
    hash_or_none(pins.get("registry_digest"), "registry_digest")
    hash_or_none(pins.get("catalog_digest"), "catalog_digest")
    return order


def one_by_id(rows: list[dict[str, Any]], row_id: str, label: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("id") == row_id]
    if len(matches) != 1:
        raise WorkbenchError(f"{label}_unknown:{row_id}")
    return matches[0]


def axis_ids(registry: dict[str, Any], axis: str, *, implemented_only: bool = False) -> set[str]:
    rows = registry["axes"][axis]
    return {
        row["id"]
        for row in rows
        if not implemented_only or row.get("status") == "implemented"
    }


def make_stages(names: list[str], evidence: set[str]) -> list[dict[str, Any]]:
    requirements_by_stage = SAFE_STAGE_REQUIREMENTS | LEAP_STAGE_REQUIREMENTS
    stages: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        requirements = requirements_by_stage.get(name, [])
        stages.append(
            {
                "index": index,
                "stage": name,
                "requires": requirements,
                "state": "satisfied" if set(requirements) <= evidence else "planned",
            }
        )
    return stages


def resolve(order: dict[str, Any], registry: dict[str, Any], registry_digest: str) -> dict[str, Any]:
    pins = order["pins"]
    work_owner = registry["work_owner"]
    if pins["registry_digest"] is not None and pins["registry_digest"] != registry_digest:
        raise WorkbenchError("registry_digest_mismatch")

    translator = one_by_id(registry["translators"], order["translator_id"], "translator")
    source = order["source"]
    target_request = order["target"]
    if source["language"] != translator["source_language"]:
        raise WorkbenchError("translator_source_language_mismatch")
    if target_request["language"] != translator["target_language"]:
        raise WorkbenchError("translator_target_language_mismatch")
    if source["language"] == target_request["language"]:
        raise WorkbenchError("source_target_languages_must_differ")
    if pins["catalog_digest"] is not None and pins["catalog_digest"] != translator["catalog_digest"]:
        raise WorkbenchError("catalog_digest_mismatch")

    target_matches = [
        target
        for target in registry["targets"]
        if all(
            target.get(key) == target_request[key]
            for key in ("platform", "architecture", "processor_plane", "memory_topology")
        )
        and target_request["execution_topology"] in target.get("execution_topologies", [])
    ]
    if len(target_matches) != 1:
        raise WorkbenchError("target_tuple_not_implemented")
    target = target_matches[0]

    for axis, value in (
        ("source_languages", source["language"]),
        ("target_languages", target_request["language"]),
        ("processor_planes", target_request["processor_plane"]),
        ("memory_topologies", target_request["memory_topology"]),
        ("execution_topologies", target_request["execution_topology"]),
    ):
        if value not in axis_ids(registry, axis, implemented_only=True):
            raise WorkbenchError(f"axis_value_not_implemented:{axis}:{value}")

    scope = (
        f"{translator['id']}@{target['id']}@"
        f"{target_request['execution_topology']}"
    )
    profile_id = order["runtime_profile"] or registry["defaults_by_scope"].get(scope)
    if profile_id is None:
        raise WorkbenchError(f"default_profile_missing:{scope}")
    profile = one_by_id(registry["runtime_profiles"], profile_id, "runtime_profile")
    if translator["id"] not in profile["translator_ids"] or target["id"] not in profile["target_ids"]:
        raise WorkbenchError("runtime_profile_scope_mismatch")
    toolchain = one_by_id(
        registry["translator_toolchains"],
        profile["translator_toolchain_ref"],
        "translator_toolchain",
    )
    if toolchain["execution_topology"] != target_request["execution_topology"]:
        raise WorkbenchError("runtime_profile_execution_topology_mismatch")
    publisher = one_by_id(registry["publishers"], profile["publisher_ref"], "publisher")

    evidence = set(order["evidence_available"])
    unknown_evidence = evidence - set(registry["evidence_ladder"])
    if unknown_evidence:
        raise WorkbenchError(f"evidence_unknown:{sorted(unknown_evidence)[0]}")
    unknown_scopes = set(order["optimization_scopes"]) - set(registry["optimization_scopes"])
    if unknown_scopes:
        raise WorkbenchError(f"optimization_scope_unknown:{sorted(unknown_scopes)[0]}")

    requested_trajectory = order["trajectory"]
    selected_trajectory = requested_trajectory
    disposition = "READY"
    obligations: list[str] = []
    leap = registry["trajectories"]["leap_target"]
    missing_leap = [item for item in leap["admission_evidence"] if item not in evidence]
    if requested_trajectory == "leap_target" and missing_leap:
        selected_trajectory = leap["fallback"]
        disposition = "SAFE_FALLBACK"
        obligations.append(f"HOLD(leap_target_missing:{','.join(missing_leap)})")

    stage_names = list(registry["trajectories"]["safe_reachable"]["stages"])
    if selected_trajectory == "leap_target":
        stage_names.extend(leap["stages"])

    plan: dict[str, Any] = {
        "schema": "kernel-plan/v1",
        "work_order_id": order["work_order_id"],
        "work_order_digest": digest(order),
        "registry_id": registry["registry_id"],
        "registry_digest": registry_digest,
        "work_owner": {
            "id": work_owner["id"],
            "name": work_owner["name"],
            "role": work_owner["role"],
            "scope": work_owner["scope"],
            "source_url": work_owner["source_url"],
        },
        "translator": {
            "id": translator["id"],
            "kind": translator["translation_kind"],
            "catalog_ref": translator["catalog_ref"],
            "catalog_digest": translator["catalog_digest"],
        },
        "source": {
            "language": source["language"],
            "dialect": source["dialect"],
            "root_ref": source["root_ref"],
            "digest": source["digest"],
        },
        "target": {
            "id": target["id"],
            **target_request,
        },
        "runtime_profile": {
            "id": profile["id"],
            "claim_ceiling": profile["claim_ceiling"],
            "publisher": {
                "id": publisher["id"],
                "name": publisher["name"],
                "role": publisher["role"],
                "scope": publisher["scope"],
                "ownership_party": publisher["ownership_party"],
            },
            "translator_toolchain": {
                "id": toolchain["id"],
                "definition_ref": toolchain["definition_ref"],
                "image": toolchain["image"],
            },
        },
        "trajectory": {
            "requested": requested_trajectory,
            "selected": selected_trajectory,
            "fallback": selected_trajectory != requested_trajectory,
        },
        "optimization_scopes": order["optimization_scopes"],
        "evidence_available": [
            item for item in registry["evidence_ladder"] if item in evidence
        ],
        "stages": make_stages(stage_names, evidence),
        "disposition": disposition,
        "obligations": obligations,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def hold(reason: str) -> dict[str, Any]:
    return {
        "schema": "kernel-workbench-hold/v1",
        "disposition": "HOLD",
        "reason": reason,
    }


def emit(value: Any) -> None:
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True))
    sys.stdout.write("\n")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser("registry")
    validate = subcommands.add_parser("validate")
    validate.add_argument("work_order", type=Path)
    plan = subcommands.add_parser("plan")
    plan.add_argument("work_order", type=Path)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        registry, registry_digest = load_registry(args.registry)
        if args.command == "registry":
            emit(
                {
                    "schema": "kernel-workbench-registry-view/v1",
                    "registry_digest": registry_digest,
                    "registry": registry,
                }
            )
            return 0
        order = load_work_order(args.work_order)
        if args.command == "validate":
            emit(
                {
                    "schema": "kernel-work-order-validation/v1",
                    "disposition": "VALID",
                    "work_order_id": order["work_order_id"],
                    "work_order_digest": digest(order),
                    "registry_digest": registry_digest,
                }
            )
            return 0
        emit(resolve(order, registry, registry_digest))
        return 0
    except WorkbenchError as error:
        emit(hold(str(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
