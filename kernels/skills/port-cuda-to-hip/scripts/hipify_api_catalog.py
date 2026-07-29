#!/usr/bin/env python3
"""Normalize HIPIFY's full CSV documentation into one typed mapping catalog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "hipify-api-catalog/v1"
HIP_HEADER = ("CUDA", "A", "D", "C", "R", "HIP", "A", "D", "C", "R", "U", "E")
JOINT_HEADER = HIP_HEADER + ("ROC", "A", "D", "C", "R", "E")
EMPTY_ROC_HEADER = HIP_HEADER + ("", "A", "D", "C", "R", "E")
MAX_FILES = 100
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000
SUPPORT_STATES = (
    "mapped",
    "version-restricted",
    "experimental",
    "unmapped",
    "not-emitted",
)
VERSION_KEYS = {"added", "changed", "deprecated", "removed"}
FULL_JOINT_SOURCE_FILES = {
    "CUBLAS_API_supported_by_HIP_and_ROC.csv",
    "CUB_API_supported_by_HIP.csv",
    "CUDA_Device_API_supported_by_HIP.csv",
    "CUDA_Driver_API_functions_supported_by_HIP.csv",
    "CUDA_RTC_API_supported_by_HIP.csv",
    "CUDA_Runtime_API_functions_supported_by_HIP.csv",
    "CUFFT_API_supported_by_HIP.csv",
    "CURAND_API_supported_by_HIP_and_ROC.csv",
    "CUSOLVER_API_supported_by_HIP_and_ROC.csv",
    "CUSPARSE_API_supported_by_HIP_and_ROC.csv",
    "CUTENSOR_API_supported_by_HIP.csv",
    "cuComplex_API_supported_by_HIP.csv",
    "cuFile_API_supported_by_HIP.csv",
}
OFFICIAL_SOURCES = {
    "all": "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/supported_apis.html",
    "hip": "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/hip_supported_apis.html",
    "hip_and_roc": (
        "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/"
        "hip_roc_supported_apis.html"
    ),
    "roc": "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/roc_supported_apis.html",
}


class CatalogError(RuntimeError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def support_state(name: str | None, unsupported: str | None, experimental: str | None) -> str:
    if name is None:
        return "unmapped"
    if experimental is not None:
        return "experimental"
    if unsupported is not None:
        return "version-restricted"
    return "mapped"


def target(
    namespace: str,
    row: list[str],
    start: int,
    *,
    includes_unsupported: bool,
) -> dict[str, Any]:
    name = clean(row[start])
    added = clean(row[start + 1])
    deprecated = clean(row[start + 2])
    changed = clean(row[start + 3])
    removed = clean(row[start + 4])
    if includes_unsupported:
        unsupported = clean(row[start + 5])
        experimental = clean(row[start + 6])
    else:
        unsupported = None
        experimental = clean(row[start + 5])
    return {
        "experimental": experimental,
        "name": name,
        "namespace": namespace,
        "support": support_state(name, unsupported, experimental),
        "unsupported_cuda_versions": unsupported,
        "versions": {
            "added": added,
            "changed": changed,
            "deprecated": deprecated,
            "removed": removed,
        },
    }


def heading(value: str) -> str | None:
    stripped = value.strip().strip("#").strip()
    if not stripped:
        return None
    if stripped.startswith(("<", "**Note", "---", "```")):
        return None
    return stripped


def parse_csv_document(path: Path, raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError("catalog_input_is_not_utf8", path=path.name) from exc

    rows: list[dict[str, Any]] = []
    active_header: tuple[str, ...] | None = None
    current_section = "unclassified"
    pending_heading = current_section
    headers_seen = 0

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        for parsed in reader:
            line_number = reader.line_num
            parsed_header = tuple(cell.strip() for cell in parsed)
            if parsed_header in {HIP_HEADER, JOINT_HEADER, EMPTY_ROC_HEADER}:
                active_header = parsed_header
                current_section = pending_heading
                headers_seen += 1
                continue

            if active_header is not None:
                if not parsed or all(clean(cell) is None for cell in parsed):
                    active_header = None
                    continue
                if len(parsed) != len(active_header) or clean(parsed[0]) is None:
                    raise CatalogError(
                        "catalog_csv_malformed_mapping_row",
                        actual_columns=len(parsed),
                        expected_columns=len(active_header),
                        line=line_number,
                        path=path.name,
                    )
                cuda_name = clean(parsed[0])
                assert cuda_name is not None
                rows.append(
                    {
                        "cuda": {
                            "name": cuda_name,
                            "namespace": "cuda",
                            "versions": {
                                "added": clean(parsed[1]),
                                "changed": clean(parsed[3]),
                                "deprecated": clean(parsed[2]),
                                "removed": clean(parsed[4]),
                            },
                        },
                        "domain": path.stem,
                        "line": line_number,
                        "section": current_section,
                        "targets": {
                            "hip": target("hip", parsed, 5, includes_unsupported=True),
                            "roc": (
                                target("roc", parsed, 12, includes_unsupported=False)
                                if active_header in {JOINT_HEADER, EMPTY_ROC_HEADER}
                                else {
                                    "experimental": None,
                                    "name": None,
                                    "namespace": "roc",
                                    "support": "not-emitted",
                                    "unsupported_cuda_versions": None,
                                    "versions": {
                                        "added": None,
                                        "changed": None,
                                        "deprecated": None,
                                        "removed": None,
                                    },
                                }
                            ),
                        },
                    }
                )
                if len(rows) > MAX_ROWS:
                    raise CatalogError("catalog_row_limit_exceeded", limit=MAX_ROWS)
                continue

            candidate_heading = heading(" ".join(parsed))
            if candidate_heading is not None:
                pending_heading = candidate_heading
    except csv.Error as exc:
        raise CatalogError(
            "catalog_csv_parse_failed",
            line=reader.line_num,
            path=path.name,
        ) from exc

    if headers_seen == 0:
        raise CatalogError("catalog_csv_header_not_found", path=path.name)
    if not rows:
        raise CatalogError("catalog_csv_has_no_rows", path=path.name)
    return rows


def normalize(
    input_root: Path,
    tool_version: str,
    generation_command: str,
    completeness: str,
) -> dict[str, Any]:
    requested = input_root.expanduser()
    if requested.is_symlink():
        raise CatalogError("catalog_input_root_must_not_be_symlink", path=str(requested))
    root = requested.resolve()
    if not root.is_dir():
        raise CatalogError("catalog_input_root_is_not_a_directory", path=str(root))

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise CatalogError("catalog_csv_files_not_found", path=str(root))
    if len(csv_files) > MAX_FILES:
        raise CatalogError("catalog_file_limit_exceeded", limit=MAX_FILES)
    observed_names = {path.name for path in csv_files}
    if completeness == "full" and observed_names != FULL_JOINT_SOURCE_FILES:
        raise CatalogError(
            "catalog_full_joint_source_set_mismatch",
            missing=sorted(FULL_JOINT_SOURCE_FILES - observed_names),
            unexpected=sorted(observed_names - FULL_JOINT_SOURCE_FILES),
        )

    source_receipts: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in csv_files:
        if path.is_symlink():
            raise CatalogError("catalog_input_symlink_forbidden", path=path.name)
        raw = path.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise CatalogError(
                "catalog_file_size_limit_exceeded",
                path=path.name,
                limit=MAX_FILE_BYTES,
            )
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            raise CatalogError("catalog_total_byte_limit_exceeded", limit=MAX_TOTAL_BYTES)
        rows = parse_csv_document(path, raw)
        parsed_rows.extend(rows)
        source_receipts.append(
            {
                "bytes": len(raw),
                "mapping_rows": len(rows),
                "path": path.name,
                "sha256": sha256_bytes(raw),
            }
        )

    apis: dict[str, dict[str, Any]] = {}
    for row in parsed_rows:
        symbol = row["cuda"]["name"]
        api = apis.setdefault(
            symbol,
            {
                "cuda": row["cuda"],
                "variants": [],
            },
        )
        variant = {
            "domain": row["domain"],
            "section": row["section"],
            "targets": row["targets"],
        }
        if variant not in api["variants"]:
            api["variants"].append(variant)

    for api in apis.values():
        api["variants"].sort(key=lambda value: (value["domain"], value["section"]))

    state_counts = {
        namespace: {
            state: sum(
                1
                for row in parsed_rows
                if row["targets"][namespace]["support"] == state
            )
            for state in ("mapped", "version-restricted", "experimental", "unmapped", "not-emitted")
        }
        for namespace in ("hip", "roc")
    }
    evidence = (
        "tool-generated-static-mapping"
        if completeness == "full"
        else "partial-static-mapping"
    )
    catalog = {
        "apis": {symbol: apis[symbol] for symbol in sorted(apis)},
        "generation": {
            "command": generation_command,
            "completeness": completeness,
            "format": "csv",
            "mode": "full" if completeness == "full" else "partial",
            "roc_mode": "joint" if completeness == "full" else "partial",
            "tool": "hipify-clang",
            "tool_version": tool_version,
        },
        "ontology": {
            "evidence": evidence,
            "source_namespace": "cuda",
            "support_states": list(SUPPORT_STATES),
            "target_namespaces": ["hip", "roc"],
            "warning": (
                "A mapping is a transpiler capability claim for this HIPIFY build, "
                "not compile, semantic-equivalence, device-parity, or performance evidence."
            ),
        },
        "official_sources": OFFICIAL_SOURCES,
        "schema": SCHEMA,
        "sources": source_receipts,
        "summary": {
            "cuda_symbols": len(apis),
            "input_bytes": total_bytes,
            "input_files": len(source_receipts),
            "mapping_rows": len(parsed_rows),
            "target_state_rows": state_counts,
        },
    }
    validate_catalog(catalog)
    return catalog


def require_nullable_text(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise CatalogError("catalog_nested_value_invalid", field=label)


def validate_versions(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != VERSION_KEYS:
        raise CatalogError("catalog_versions_invalid", field=label)
    for key in VERSION_KEYS:
        require_nullable_text(value[key], f"{label}.{key}")


def validate_target(value: Any, namespace: str, label: str) -> str:
    expected_keys = {
        "experimental",
        "name",
        "namespace",
        "support",
        "unsupported_cuda_versions",
        "versions",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise CatalogError("catalog_target_invalid", field=label)
    if value["namespace"] != namespace:
        raise CatalogError("catalog_target_namespace_invalid", field=label)
    for key in ("experimental", "name", "unsupported_cuda_versions"):
        require_nullable_text(value[key], f"{label}.{key}")
    validate_versions(value["versions"], f"{label}.versions")
    state = value["support"]
    if state not in SUPPORT_STATES:
        raise CatalogError("catalog_support_state_invalid", field=label, state=state)
    if state == "not-emitted":
        if namespace != "roc" or any(
            value[key] is not None
            for key in ("experimental", "name", "unsupported_cuda_versions")
        ) or any(item is not None for item in value["versions"].values()):
            raise CatalogError("catalog_not_emitted_target_invalid", field=label)
    else:
        expected_state = support_state(
            value["name"],
            value["unsupported_cuda_versions"],
            value["experimental"],
        )
        if state != expected_state:
            raise CatalogError(
                "catalog_support_state_inconsistent",
                actual=state,
                expected=expected_state,
                field=label,
            )
    return state


def validate_catalog(catalog: dict[str, Any]) -> None:
    required_top = {
        "apis",
        "generation",
        "official_sources",
        "ontology",
        "schema",
        "sources",
        "summary",
    }
    if set(catalog) != required_top or catalog.get("schema") != SCHEMA:
        raise CatalogError("catalog_schema_invalid", expected=SCHEMA)

    generation = catalog["generation"]
    required_generation = {
        "command",
        "completeness",
        "format",
        "mode",
        "roc_mode",
        "tool",
        "tool_version",
    }
    if not isinstance(generation, dict) or set(generation) != required_generation:
        raise CatalogError("catalog_generation_invalid")
    completeness = generation["completeness"]
    if completeness not in {"full", "partial"}:
        raise CatalogError("catalog_completeness_invalid")
    expected_mode = "full" if completeness == "full" else "partial"
    expected_roc_mode = "joint" if completeness == "full" else "partial"
    if (
        generation["format"] != "csv"
        or generation["mode"] != expected_mode
        or generation["roc_mode"] != expected_roc_mode
        or generation["tool"] != "hipify-clang"
    ):
        raise CatalogError("catalog_generation_mode_inconsistent")
    for field in ("command", "tool_version"):
        if not isinstance(generation[field], str) or not generation[field].strip():
            raise CatalogError("catalog_generation_value_invalid", field=field)

    ontology = catalog["ontology"]
    required_ontology = {
        "evidence",
        "source_namespace",
        "support_states",
        "target_namespaces",
        "warning",
    }
    expected_evidence = (
        "tool-generated-static-mapping"
        if completeness == "full"
        else "partial-static-mapping"
    )
    if (
        not isinstance(ontology, dict)
        or set(ontology) != required_ontology
        or ontology["evidence"] != expected_evidence
        or ontology["source_namespace"] != "cuda"
        or ontology["support_states"] != list(SUPPORT_STATES)
        or ontology["target_namespaces"] != ["hip", "roc"]
        or not isinstance(ontology["warning"], str)
        or not ontology["warning"].strip()
    ):
        raise CatalogError("catalog_ontology_invalid")
    if catalog["official_sources"] != OFFICIAL_SOURCES:
        raise CatalogError("catalog_official_sources_invalid")

    sources = catalog["sources"]
    if not isinstance(sources, list) or not sources:
        raise CatalogError("catalog_sources_invalid")
    source_names: set[str] = set()
    source_rows: dict[str, int] = {}
    input_bytes = 0
    mapping_rows = 0
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {
            "bytes",
            "mapping_rows",
            "path",
            "sha256",
        }:
            raise CatalogError("catalog_source_receipt_invalid", index=index)
        path = source["path"]
        if (
            not isinstance(path, str)
            or Path(path).name != path
            or not path.endswith(".csv")
            or path in source_names
        ):
            raise CatalogError("catalog_source_path_invalid", index=index)
        if (
            not isinstance(source["bytes"], int)
            or isinstance(source["bytes"], bool)
            or source["bytes"] <= 0
            or not isinstance(source["mapping_rows"], int)
            or isinstance(source["mapping_rows"], bool)
            or source["mapping_rows"] <= 0
            or not isinstance(source["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        ):
            raise CatalogError("catalog_source_receipt_invalid", index=index)
        source_names.add(path)
        source_rows[Path(path).stem] = source["mapping_rows"]
        input_bytes += source["bytes"]
        mapping_rows += source["mapping_rows"]
    if completeness == "full" and source_names != FULL_JOINT_SOURCE_FILES:
        raise CatalogError(
            "catalog_full_joint_source_set_mismatch",
            missing=sorted(FULL_JOINT_SOURCE_FILES - source_names),
            unexpected=sorted(source_names - FULL_JOINT_SOURCE_FILES),
        )

    apis = catalog["apis"]
    if not isinstance(apis, dict) or not apis:
        raise CatalogError("catalog_apis_invalid")
    observed_state_counts = {
        namespace: {state: 0 for state in SUPPORT_STATES}
        for namespace in ("hip", "roc")
    }
    observed_domain_rows = {domain: 0 for domain in source_rows}
    observed_variants = 0
    for symbol, api in apis.items():
        if not isinstance(symbol, str) or not symbol.strip() or not isinstance(api, dict):
            raise CatalogError("catalog_api_invalid", symbol=symbol)
        if set(api) != {"cuda", "variants"}:
            raise CatalogError("catalog_api_invalid", symbol=symbol)
        cuda = api["cuda"]
        if (
            not isinstance(cuda, dict)
            or set(cuda) != {"name", "namespace", "versions"}
            or cuda["name"] != symbol
            or cuda["namespace"] != "cuda"
        ):
            raise CatalogError("catalog_cuda_entry_invalid", symbol=symbol)
        validate_versions(cuda["versions"], f"apis.{symbol}.cuda.versions")
        variants = api["variants"]
        if not isinstance(variants, list) or not variants:
            raise CatalogError("catalog_variants_invalid", symbol=symbol)
        seen_variants: set[str] = set()
        for index, variant in enumerate(variants):
            label = f"apis.{symbol}.variants.{index}"
            if not isinstance(variant, dict) or set(variant) != {
                "domain",
                "section",
                "targets",
            }:
                raise CatalogError("catalog_variant_invalid", field=label)
            domain = variant["domain"]
            if domain not in source_rows:
                raise CatalogError("catalog_variant_domain_invalid", field=label)
            if not isinstance(variant["section"], str) or not variant["section"].strip():
                raise CatalogError("catalog_variant_section_invalid", field=label)
            targets = variant["targets"]
            if not isinstance(targets, dict) or set(targets) != {"hip", "roc"}:
                raise CatalogError("catalog_targets_invalid", field=label)
            for namespace in ("hip", "roc"):
                state = validate_target(
                    targets[namespace],
                    namespace,
                    f"{label}.targets.{namespace}",
                )
                observed_state_counts[namespace][state] += 1
            fingerprint = json.dumps(variant, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen_variants:
                raise CatalogError("catalog_duplicate_variant", field=label)
            seen_variants.add(fingerprint)
            observed_domain_rows[domain] += 1
            observed_variants += 1

    summary = catalog["summary"]
    if not isinstance(summary, dict) or set(summary) != {
        "cuda_symbols",
        "input_bytes",
        "input_files",
        "mapping_rows",
        "target_state_rows",
    }:
        raise CatalogError("catalog_summary_invalid")
    expected_summary = {
        "cuda_symbols": len(apis),
        "input_bytes": input_bytes,
        "input_files": len(sources),
        "mapping_rows": observed_variants,
        "target_state_rows": observed_state_counts,
    }
    if summary != expected_summary or mapping_rows != observed_variants:
        raise CatalogError(
            "catalog_summary_inconsistent",
            expected=expected_summary,
        )
    if observed_domain_rows != source_rows:
        raise CatalogError(
            "catalog_source_row_counts_inconsistent",
            expected=source_rows,
            observed=observed_domain_rows,
        )


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    requested = path.expanduser()
    if requested.exists() or requested.is_symlink():
        raise CatalogError("catalog_output_already_exists", path=str(requested))
    if not requested.parent.exists():
        raise CatalogError("catalog_output_parent_does_not_exist", path=str(requested.parent))
    output = requested.parent.resolve() / requested.name
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise CatalogError("catalog_output_already_exists", path=str(output)) from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_catalog(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise CatalogError("catalog_symlink_forbidden", path=str(path))
    if not path.is_file():
        raise CatalogError("catalog_not_found", path=str(path))
    if path.stat().st_size > MAX_TOTAL_BYTES:
        raise CatalogError("catalog_total_byte_limit_exceeded", limit=MAX_TOTAL_BYTES)
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("catalog_is_not_valid_json", path=str(path)) from exc
    if not isinstance(catalog, dict):
        raise CatalogError("catalog_schema_invalid", expected=SCHEMA)
    validate_catalog(catalog)
    return catalog


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("input_root", type=Path)
    normalize_parser.add_argument("output", type=Path)
    normalize_parser.add_argument("--tool-version", required=True)
    normalize_parser.add_argument(
        "--generation-command",
        default="hipify-clang --csv --doc-format=full --doc-roc=joint",
    )
    normalize_parser.add_argument(
        "--completeness",
        choices=("full", "partial"),
        default="full",
        help="Require the complete known full/joint artifact set, or label a bounded fixture partial.",
    )

    lookup_parser = subparsers.add_parser("lookup")
    lookup_parser.add_argument("catalog", type=Path)
    lookup_parser.add_argument("cuda_symbol")
    lookup_parser.add_argument("--backend", choices=("hip", "roc"))

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("catalog", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "normalize":
            catalog = normalize(
                args.input_root,
                args.tool_version,
                args.generation_command,
                args.completeness,
            )
            write_new_json(args.output, catalog)
            result = {
                "disposition": "CATALOG_GENERATED",
                "evidence": catalog["ontology"]["evidence"],
                "output": str(args.output.resolve()),
                "schema": SCHEMA,
                "summary": catalog["summary"],
            }
        else:
            catalog = load_catalog(args.catalog)
            if args.command == "validate":
                result = {
                    "disposition": "CATALOG_VALID",
                    "evidence": "structurally-validated",
                    "schema": SCHEMA,
                    "summary": catalog["summary"],
                }
            else:
                api = catalog["apis"].get(args.cuda_symbol)
                if api is None:
                    raise CatalogError("cuda_symbol_not_found", symbol=args.cuda_symbol)
                if args.backend:
                    api = {
                        **api,
                        "variants": [
                            {
                                "domain": variant["domain"],
                                "section": variant["section"],
                                "target": variant["targets"][args.backend],
                            }
                            for variant in api["variants"]
                        ],
                    }
                result = {
                    "api": api,
                    "evidence": catalog["ontology"]["evidence"],
                    "schema": SCHEMA,
                }
    except CatalogError as exc:
        result = {
            "details": exc.details,
            "disposition": "HOLD",
            "reason": exc.reason,
            "schema": "hipify-api-catalog-hold/v1",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
