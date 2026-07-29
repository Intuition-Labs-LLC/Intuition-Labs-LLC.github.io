#!/usr/bin/env python3
"""Bounded CUDA inventory and isolated HIPIFY candidate generation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fnmatch
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_INVENTORY = "cuda-hip-inventory/v1"
SCHEMA_CONVERSION = "cuda-hip-conversion/v1"
SCHEMA_CATALOG = "hipify-api-catalog/v1"
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "assets" / "hipify-api-catalog.json"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hip", ".hpp", ".hxx"}
ALWAYS_GPU_SUFFIXES = {".cu", ".cuh", ".hip"}
BUILD_NAMES = {"CMakeLists.txt", "Makefile", "BUILD", "BUILD.bazel", "meson.build"}
BUILD_SUFFIXES = {".bazel", ".cmake", ".mk"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "out",
    "target",
    "vendor",
}
CUDA_MARKER = re.compile(
    r"(?:#\s*include\s*[<\"](?:cuda|cuBLAS|cublas|cudnn|cufft|cusolver|cusparse|curand|cub/)"
    r"|\b(?:cuda[A-Z]\w*|cu[A-Z]\w*|__CUDA_ARCH__|nvcc)\b"
    r"|<<<)"
)
HIP_MARKER = re.compile(
    r"(?:#\s*include\s*[<\"]hip/|\bhip[A-Z]\w*|\b__HIP(?:CC|_DEVICE_COMPILE|_PLATFORM_\w+)?__\b)"
)
GPU_SHARED_MARKER = re.compile(r"\b(?:__global__|__device__|__host__|__shared__)\b")
CUDA_LIKE_TOKEN = re.compile(
    r"\b(?:cuda[A-Z][A-Za-z0-9_]*|cu(?:blas|BLAS|dnn|DNN|fft|FFT|rand|RAND|"
    r"solver|SOLVER|sparse|SPARSE|tensor|TENSOR|File)[A-Za-z0-9_]*|"
    r"CU(?:BLAS|DNN|FFT|RAND|SOLVER|SPARSE|TENSOR|FILE)_[A-Z0-9_]+|"
    r"cu(?:Ctx|Module|Device|Mem|Launch|Stream|Event)[A-Za-z0-9_]+)\b"
)
BUILD_CUDA_MARKER = re.compile(
    r"(?:\bCUDA\b|CMAKE_CUDA|CUDAToolkit|enable_language\s*\(\s*CUDA|\bnvcc\b|\.cu\b|-lcudart?\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiskRule:
    risk_id: str
    severity: str
    description: str
    pattern: re.Pattern[str]


RISK_RULES = (
    RiskRule(
        "inline-ptx-or-assembly",
        "high",
        "Inline assembly requires architecture-specific manual review; PTX is not HIP assembly.",
        re.compile(r"\b(?:asm|__asm__)\s*(?:volatile\s*)?\(|\b(?:mov|ld|st|bar|shfl)\.(?:u|s|b|f)\d+"),
    ),
    RiskRule(
        "warp-width-or-lane-mask",
        "high",
        "Hard-coded warp width or lane-mask assumptions can differ on AMD targets.",
        re.compile(r"(?i)(?:warp|lane|shfl|ballot|activemask|match).*(?:\b31\b|\b32\b|0xffffffff|1u?\s*<<)"),
    ),
    RiskRule(
        "cuda-architecture-test",
        "high",
        "Numeric __CUDA_ARCH__ tests require feature-oriented HIP review.",
        re.compile(r"\b__CUDA_ARCH__\b"),
    ),
    RiskRule(
        "warp-intrinsic",
        "high",
        "Warp intrinsics require wave-size, mask-width, and divergence review.",
        re.compile(r"\b__(?:shfl|ballot|activemask|syncwarp|match)"),
    ),
    RiskRule(
        "two-argument-launch-bounds",
        "high",
        "HIP assigns a different meaning to the second __launch_bounds__ parameter.",
        re.compile(r"__launch_bounds__\s*\([^,\n]+,[^)]+\)"),
    ),
    RiskRule(
        "cooperative-groups",
        "high",
        "Cooperative groups require target support and divergent-synchronization review.",
        re.compile(r"\bcooperative_groups\b|\bcg::"),
    ),
    RiskRule(
        "texture-or-surface",
        "high",
        "Texture and surface APIs require explicit mapping and behavior tests.",
        re.compile(r"\b(?:cudaTexture|cudaSurface|texture\s*<|surface\s*<|tex[123]D|surf[123]D)"),
    ),
    RiskRule(
        "async-copy-or-pipeline",
        "high",
        "CUDA async-copy and pipeline features require a supported HIP design.",
        re.compile(r"\b(?:__pipeline|cuda::memcpy_async|cuda::pipeline|cuda::barrier)\b|cp\.async"),
    ),
    RiskRule(
        "cuda-library",
        "medium",
        "CUDA library usage must be checked API-by-API against HIP/ROCm mappings.",
        re.compile(
            r"(?i)(?:\b(?:cublas|cudnn|cufft|cusolver|cusparse|curand|cutensor|npp|nccl)\w*"
            r"|(?:\bcub::|[<\"]cub/))"
        ),
    ),
    RiskRule(
        "cuda-driver-or-context-api",
        "medium",
        "CUDA driver/context/module APIs differ from HIP runtime behavior.",
        re.compile(r"\bcu(?:Ctx|Module|Device|Mem|Launch|Stream|Event)[A-Z]\w*"),
    ),
    RiskRule(
        "cuda-graph-or-stream-capture",
        "medium",
        "Graph and stream-capture support must be checked on the target ROCm stack.",
        re.compile(r"\bcuda(?:Graph|StreamBeginCapture|StreamEndCapture)\w*"),
    ),
    RiskRule(
        "atomic-or-memory-order",
        "medium",
        "Atomics and fences require operation, scope, ordering, and type review.",
        re.compile(r"\b(?:atomic(?:Add|Sub|Exch|Min|Max|Inc|Dec|CAS|And|Or|Xor)|__threadfence\w*)\b"),
    ),
    RiskRule(
        "barrier-or-divergent-sync",
        "medium",
        "Barrier placement must remain convergent and match HIP synchronization semantics.",
        re.compile(r"\b__(?:syncthreads|syncwarp)\b"),
    ),
    RiskRule(
        "nvidia-specific-device-type",
        "medium",
        "NVIDIA-specific scalar/vector types require mapping and numeric tests.",
        re.compile(r"\b(?:__half|__nv_bfloat16|nv_bfloat16|nvcuda::)\b"),
    ),
    RiskRule(
        "cuda-memory-topology",
        "review",
        "Memory allocation and copies need target-specific topology and ordering review.",
        re.compile(r"\bcuda(?:MallocManaged|Malloc|HostAlloc|Memcpy|MemPrefetchAsync|MemAdvise)\w*"),
    ),
    RiskRule(
        "nvidia-build-flag",
        "high",
        "NVIDIA architecture/register flags require AMD build-system equivalents.",
        re.compile(r"(?:-gencode|--generate-code|-arch[= ]sm_|--maxrregcount|CMAKE_CUDA_ARCHITECTURES)"),
    ),
)


class HoldError(RuntimeError):
    """Expected fail-closed disposition."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_json(
    value: dict[str, Any],
    output: Path | None = None,
    forbidden_root: Path | None = None,
) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    requested = output.expanduser()
    if requested.exists() or requested.is_symlink():
        raise HoldError("inventory_output_already_exists", path=str(requested))
    if not requested.parent.exists():
        raise HoldError("inventory_output_parent_does_not_exist", path=str(requested.parent))
    output = requested.parent.resolve() / requested.name
    if forbidden_root is not None and is_within(output, forbidden_root):
        raise HoldError("inventory_output_must_be_outside_source", path=str(output))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise HoldError("inventory_output_already_exists", path=str(output)) from exc
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    print(f"Wrote {value['schema']}: {output}")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory without replacing a concurrent destination."""

    if os.name != "posix":
        raise HoldError("atomic_no_replace_unavailable", platform=os.name)
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise HoldError("atomic_no_replace_unavailable", platform=sys.platform)
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HoldError("output_root_already_exists", path=str(output))
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise HoldError("atomic_no_replace_unavailable", platform=sys.platform)
    raise OSError(error_number, os.strerror(error_number), str(output))


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def run_bounded(
    command: list[str],
    *,
    timeout: float,
    stdout_limit: int,
    stderr_limit: int,
    timeout_reason: str,
    limit_reason: str,
    watched_file: Path | None = None,
    file_limit: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess while bounding elapsed time and captured output."""

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise HoldError("translator_execution_failed") from exc

    assert process.stdout is not None
    assert process.stderr is not None
    streams = {
        process.stdout: ("stdout", stdout_limit, bytearray()),
        process.stderr: ("stderr", stderr_limit, bytearray()),
    }
    selector = selectors.DefaultSelector()
    for stream in streams:
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process(process)
                raise HoldError(timeout_reason, timeout=timeout)
            if (
                watched_file is not None
                and file_limit is not None
                and watched_file.exists()
                and watched_file.stat().st_size > file_limit
            ):
                terminate_process(process)
                raise HoldError(limit_reason, stream="output_file", limit=file_limit)
            events = selector.select(min(remaining, 0.1 if watched_file is not None else remaining))
            if not events:
                if process.poll() is None:
                    continue
                continue
            for key, _ in events:
                stream = key.fileobj
                stream_name, limit, buffer = streams[stream]
                chunk = os.read(stream.fileno(), min(65536, limit - len(buffer) + 1))
                if not chunk:
                    selector.unregister(stream)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limit:
                    terminate_process(process)
                    raise HoldError(limit_reason, stream=stream_name, limit=limit)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminate_process(process)
            raise HoldError(timeout_reason, timeout=timeout)
        return_code = process.wait(timeout=remaining)
        if watched_file is not None and file_limit is not None:
            if watched_file.is_symlink():
                raise HoldError(limit_reason, stream="output_file_symlink", limit=file_limit)
            if watched_file.exists() and watched_file.stat().st_size > file_limit:
                raise HoldError(limit_reason, stream="output_file", limit=file_limit)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process)
        raise HoldError(timeout_reason, timeout=timeout) from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()

    return subprocess.CompletedProcess(
        command,
        return_code,
        bytes(streams[process.stdout][2]),
        bytes(streams[process.stderr][2]),
    )


def matches_exclusion(relative: Path, patterns: Iterable[str]) -> bool:
    value = relative.as_posix()
    return any(fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(relative.name, pattern) for pattern in patterns)


def load_api_catalog(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = path.expanduser()
    if requested.is_symlink() or not requested.is_file():
        raise HoldError("api_catalog_not_found_or_symlinked", path=str(requested))
    raw = requested.read_bytes()
    if len(raw) > 25 * 1024 * 1024:
        raise HoldError("api_catalog_size_limit_exceeded", limit=25 * 1024 * 1024)
    try:
        catalog = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldError("api_catalog_is_not_valid_json") from exc
    if not isinstance(catalog, dict) or catalog.get("schema") != SCHEMA_CATALOG:
        raise HoldError("api_catalog_schema_invalid", expected=SCHEMA_CATALOG)
    apis = catalog.get("apis")
    summary = catalog.get("summary")
    generation = catalog.get("generation")
    if not isinstance(apis, dict) or not isinstance(summary, dict) or not isinstance(generation, dict):
        raise HoldError("api_catalog_structure_invalid")
    return apis, {
        "cuda_symbols": summary.get("cuda_symbols"),
        "evidence": catalog.get("ontology", {}).get("evidence"),
        "schema": SCHEMA_CATALOG,
        "sha256": sha256_bytes(raw),
        "tool_version": generation.get("tool_version"),
    }


def catalog_projection(api: dict[str, Any], backend: str) -> dict[str, list[str]]:
    names: set[str] = set()
    support: set[str] = set()
    for variant in api.get("variants", []):
        target_value = variant.get("targets", {}).get(backend, {})
        name = target_value.get("name")
        state = target_value.get("support")
        if isinstance(name, str):
            names.add(name)
        if isinstance(state, str):
            support.add(state)
    return {
        "names": sorted(names),
        "support": sorted(support),
    }


def candidate_kind(path: Path) -> str | None:
    if path.name in BUILD_NAMES or path.suffix.lower() in BUILD_SUFFIXES:
        return "build"
    if path.suffix.lower() in SOURCE_SUFFIXES:
        return "source"
    return None


def source_dialect(path: Path, text: str) -> str | None:
    suffix = path.suffix.lower()
    cuda_marked = suffix in {".cu", ".cuh"} or bool(CUDA_MARKER.search(text))
    hip_marked = suffix == ".hip" or ".hip." in path.name.lower() or bool(HIP_MARKER.search(text))
    if cuda_marked and hip_marked:
        return "mixed"
    if cuda_marked:
        return "cuda"
    if hip_marked:
        return "hip"
    if GPU_SHARED_MARKER.search(text):
        return "cuda"
    return None


def output_relative_path(relative: Path) -> Path:
    suffix = relative.suffix.lower()
    if suffix == ".cu":
        return relative.with_suffix(".hip")
    return relative


def scan_source(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.source_root).expanduser().resolve()
    if not root.is_dir():
        raise HoldError("source_root_is_not_a_directory", path=str(root))
    catalog_apis, catalog_receipt = load_api_catalog(args.catalog)

    exclusions = tuple(args.exclude or ())
    files: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    api_references: list[dict[str, Any]] = []
    seen_api_references: set[tuple[str, str]] = set()
    inspected_files = 0
    inspected_bytes = 0

    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories = []
        for directory in sorted(directories):
            relative = (current_path / directory).relative_to(root)
            if directory in DEFAULT_EXCLUDED_DIRS or matches_exclusion(relative, exclusions):
                continue
            candidate_directory = current_path / directory
            if candidate_directory.is_symlink():
                resolved = candidate_directory.resolve()
                if not is_within(resolved, root):
                    raise HoldError("source_symlink_escapes_root", path=relative.as_posix())
                raise HoldError("source_symlink_directory_not_followed", path=relative.as_posix())
            retained_directories.append(directory)
        directories[:] = retained_directories

        for filename in sorted(filenames):
            relative = (current_path / filename).relative_to(root)
            if matches_exclusion(relative, exclusions):
                continue
            inspected_files += 1
            if inspected_files > args.max_files:
                raise HoldError("scan_file_limit_exceeded", limit=args.max_files)

            candidate = current_path / filename
            if candidate.is_symlink():
                resolved = candidate.resolve()
                if not is_within(resolved, root):
                    raise HoldError("source_symlink_escapes_root", path=relative.as_posix())
                if candidate_kind(candidate) is not None:
                    raise HoldError("source_symlink_file_not_followed", path=relative.as_posix())
                continue

            kind = candidate_kind(candidate)
            if kind is None or not candidate.is_file():
                continue
            size = candidate.stat().st_size
            if size > args.max_file_bytes:
                raise HoldError(
                    "source_file_size_limit_exceeded",
                    path=relative.as_posix(),
                    bytes=size,
                    limit=args.max_file_bytes,
                )
            inspected_bytes += size
            if inspected_bytes > args.max_total_bytes:
                raise HoldError("scan_total_byte_limit_exceeded", limit=args.max_total_bytes)

            raw = candidate.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise HoldError("source_is_not_utf8", path=relative.as_posix())

            dialect = "build"
            if kind == "source":
                dialect = source_dialect(candidate, text)
                if dialect is None:
                    continue

            file_risks: set[str] = set()
            for line_number, line in enumerate(text.splitlines(), start=1):
                for symbol in sorted(set(CUDA_LIKE_TOKEN.findall(line))):
                    reference_key = (relative.as_posix(), symbol)
                    if reference_key in seen_api_references:
                        continue
                    seen_api_references.add(reference_key)
                    api = catalog_apis.get(symbol)
                    if api is None:
                        file_risks.add("cuda-api-not-in-catalog")
                        findings.append(
                            {
                                "description": (
                                    "CUDA-like identifier is absent from the bundled "
                                    "HIPIFY mapping catalog and requires manual review."
                                ),
                                "line": line_number,
                                "path": relative.as_posix(),
                                "risk_id": "cuda-api-not-in-catalog",
                                "severity": "high",
                            }
                        )
                        if len(findings) > args.max_findings:
                            raise HoldError(
                                "scan_finding_limit_exceeded",
                                limit=args.max_findings,
                            )
                    else:
                        api_references.append(
                            {
                                "cuda": symbol,
                                "hip": catalog_projection(api, "hip"),
                                "line": line_number,
                                "path": relative.as_posix(),
                                "roc": catalog_projection(api, "roc"),
                            }
                        )
                    if len(api_references) > args.max_api_references:
                        raise HoldError(
                            "scan_api_reference_limit_exceeded",
                            limit=args.max_api_references,
                        )
                for rule in RISK_RULES:
                    if not rule.pattern.search(line):
                        continue
                    file_risks.add(rule.risk_id)
                    findings.append(
                        {
                            "description": rule.description,
                            "line": line_number,
                            "path": relative.as_posix(),
                            "risk_id": rule.risk_id,
                            "severity": rule.severity,
                        }
                    )
                    if len(findings) > args.max_findings:
                        raise HoldError("scan_finding_limit_exceeded", limit=args.max_findings)

            if kind == "build" and not file_risks and not BUILD_CUDA_MARKER.search(text):
                continue
            files.append(
                {
                    "bytes": size,
                    "dialect": dialect,
                    "kind": kind,
                    "path": relative.as_posix(),
                    "risk_ids": sorted(file_risks),
                    "sha256": sha256_bytes(raw),
                }
            )

    severity_counts = {
        severity: sum(1 for finding in findings if finding["severity"] == severity)
        for severity in ("high", "medium", "review")
    }
    source_count = sum(1 for item in files if item["kind"] == "source")
    cuda_source_count = sum(
        1 for item in files if item["kind"] == "source" and item["dialect"] in {"cuda", "mixed"}
    )
    hip_source_count = sum(
        1 for item in files if item["kind"] == "source" and item["dialect"] in {"hip", "mixed"}
    )
    if source_count == 0:
        disposition = "NO_GPU_SOURCE_CANDIDATES"
    elif findings:
        disposition = "REVIEW_REQUIRED"
    else:
        disposition = "NO_STATIC_FINDINGS"
    return {
        "api_catalog": catalog_receipt,
        "api_references": api_references,
        "disposition": disposition,
        "evidence": "scanned",
        "files": files,
        "findings": findings,
        "limits": {
            "max_file_bytes": args.max_file_bytes,
            "max_files": args.max_files,
            "max_findings": args.max_findings,
            "max_total_bytes": args.max_total_bytes,
        },
        "obligations": [
            "Reconcile every finding and HIPIFY diagnostic.",
            "Compile and link for the exact AMD gfx target.",
            "Run the frozen reference corpus on observed AMD hardware.",
            "Profile only after reference parity passes.",
        ],
        "schema": SCHEMA_INVENTORY,
        "source_root": str(root),
        "summary": {
            "candidate_files": len(files),
            "catalog_api_references": len(api_references),
            "cuda_source_files": cuda_source_count,
            "findings": len(findings),
            "inspected_bytes": inspected_bytes,
            "inspected_files": inspected_files,
            "hip_source_files": hip_source_count,
            "severity_counts": severity_counts,
            "source_files": source_count,
            "unique_catalog_api_symbols": len({item["cuda"] for item in api_references}),
        },
    }


def select_translator(requested: str) -> tuple[str, str]:
    if requested == "auto":
        clang = shutil.which("hipify-clang")
        if clang:
            return "hipify-clang", clang
        raise HoldError(
            "semantic_translator_unavailable",
            hipify_perl_present=bool(shutil.which("hipify-perl")),
            next="Install/locate hipify-clang or explicitly choose --tool hipify-perl.",
        )
    resolved = shutil.which(requested)
    if not resolved:
        raise HoldError("translator_unavailable", tool=requested)
    return requested, resolved


def translator_version(tool_name: str, tool_path: str, timeout: float) -> dict[str, Any]:
    result = run_bounded(
        [tool_path, "--version"],
        timeout=min(timeout, 10),
        stdout_limit=64 * 1024,
        stderr_limit=64 * 1024,
        timeout_reason="translator_version_timeout",
        limit_reason="translator_version_output_limit_exceeded",
    )
    if result.returncode != 0:
        raise HoldError("translator_version_probe_failed", exit_code=result.returncode)
    output = result.stdout + result.stderr
    decoded = output.decode("utf-8", errors="replace")
    first_line = next((line.strip() for line in decoded.splitlines() if line.strip()), "")
    if not first_line:
        raise HoldError("translator_version_missing")
    sanitized_version = "".join(character for character in first_line if character.isprintable())[:200]
    return {
        "exit_code": result.returncode,
        "output_sha256": sha256_bytes(output),
        "output_lines": len(output.splitlines()),
        "tool": tool_name,
        "version": sanitized_version or None,
    }


def build_translator_argv(
    tool_name: str,
    tool_path: str,
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [tool_path]
    if args.roc:
        command.append("--roc")
    if tool_name == "hipify-clang":
        command.extend(["-o", str(output)])
        if args.cuda_path:
            cuda_path = Path(args.cuda_path).expanduser().resolve()
            if not cuda_path.is_dir():
                raise HoldError("cuda_path_is_not_a_directory", path=str(cuda_path))
            command.append(f"--cuda-path={cuda_path}")
        if args.compile_db:
            compile_db = Path(args.compile_db).expanduser().resolve()
            if compile_db.is_file() and compile_db.name == "compile_commands.json":
                compile_db = compile_db.parent
            if not (compile_db / "compile_commands.json").is_file():
                raise HoldError("compile_database_not_found", path=str(compile_db))
            command.extend(["-p", str(compile_db)])
    elif args.cuda_path or args.compile_db:
        raise HoldError("hipify_perl_does_not_accept_semantic_inputs")
    command.append(str(source))
    return command


def convert_source(args: argparse.Namespace) -> dict[str, Any]:
    inventory = scan_source(args)
    source_root = Path(inventory["source_root"])
    mixed_files = [
        item["path"]
        for item in inventory["files"]
        if item["kind"] == "source" and item["dialect"] == "mixed"
    ]
    if mixed_files:
        raise HoldError(
            "mixed_cuda_hip_sources_require_manual_split",
            count=len(mixed_files),
            paths=mixed_files,
        )
    source_files = [
        item
        for item in inventory["files"]
        if item["kind"] == "source" and item["dialect"] == "cuda"
    ]
    if not source_files:
        raise HoldError("no_cuda_source_files_found")

    output_requested = Path(args.output_root).expanduser()
    if output_requested.exists() or output_requested.is_symlink():
        raise HoldError("output_root_already_exists", path=str(output_requested))
    if not output_requested.parent.exists():
        raise HoldError("output_parent_does_not_exist", path=str(output_requested.parent))
    output_root = output_requested.parent.resolve() / output_requested.name
    if is_within(output_root, source_root):
        raise HoldError("output_root_must_be_outside_source", path=str(output_root))

    tool_name, tool_path = select_translator(args.tool)
    deadline = time.monotonic() + args.total_timeout
    tool_receipt = translator_version(
        tool_name,
        tool_path,
        max(0.001, deadline - time.monotonic()),
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    converted: list[dict[str, Any]] = []
    total_output_bytes = 0
    try:
        for item in source_files:
            relative = Path(item["path"])
            source = source_root / relative
            candidate_relative = output_relative_path(relative)
            candidate = staging / candidate_relative
            candidate.parent.mkdir(parents=True, exist_ok=True)

            command = build_translator_argv(tool_name, tool_path, source, candidate, args)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HoldError("translation_total_timeout", timeout=args.total_timeout)
            result = run_bounded(
                command,
                timeout=min(args.timeout, remaining),
                stdout_limit=args.max_output_bytes,
                stderr_limit=args.max_diagnostic_bytes,
                timeout_reason="translation_timeout",
                limit_reason="translation_output_limit_exceeded",
                watched_file=candidate if tool_name == "hipify-clang" else None,
                file_limit=args.max_output_bytes if tool_name == "hipify-clang" else None,
            )

            if result.returncode != 0:
                raise HoldError(
                    "translation_failed",
                    path=relative.as_posix(),
                    exit_code=result.returncode,
                    diagnostic_lines=len(result.stderr.splitlines()),
                    diagnostic_sha256=sha256_bytes(result.stderr),
                )
            if tool_name == "hipify-clang":
                if candidate.is_symlink() or not candidate.is_file():
                    raise HoldError("translator_output_file_missing", path=relative.as_posix())
                candidate_size = candidate.stat().st_size
                if candidate_size > args.max_output_bytes:
                    raise HoldError(
                        "translation_output_limit_exceeded",
                        stream="output_file",
                        limit=args.max_output_bytes,
                    )
                diagnostics = result.stdout + result.stderr
            else:
                candidate_bytes = result.stdout
                diagnostics = result.stderr
                candidate.write_bytes(candidate_bytes)
                candidate_size = len(candidate_bytes)
            if candidate_size == 0:
                raise HoldError("translator_returned_empty_output", path=relative.as_posix())
            total_output_bytes += candidate_size + len(diagnostics)
            if total_output_bytes > args.max_total_output_bytes:
                raise HoldError(
                    "translation_total_output_limit_exceeded",
                    limit=args.max_total_output_bytes,
                )

            diagnostic_relative = None
            if diagnostics:
                diagnostic_relative = Path("diagnostics") / candidate_relative.with_suffix(
                    candidate_relative.suffix + ".stderr.txt"
                )
                diagnostic = staging / diagnostic_relative
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                diagnostic.write_bytes(diagnostics)

            converted.append(
                {
                    "diagnostic_lines": len(diagnostics.splitlines()),
                    "diagnostic_path": diagnostic_relative.as_posix() if diagnostic_relative else None,
                    "diagnostic_sha256": sha256_bytes(diagnostics),
                    "input_path": relative.as_posix(),
                    "input_sha256": item["sha256"],
                    "output_path": candidate_relative.as_posix(),
                    "output_sha256": sha256_file(candidate),
                }
            )

        manifest = {
            "disposition": "MANUAL_REVIEW_REQUIRED",
            "evidence": "translated",
            "files": converted,
            "inventory_summary": inventory["summary"],
            "api_catalog": inventory["api_catalog"],
            "obligations": inventory["obligations"],
            "options": {
                "compile_database_supplied": bool(args.compile_db),
                "cuda_path_supplied": bool(args.cuda_path),
                "roc_mappings": bool(args.roc),
            },
            "limits": {
                "max_diagnostic_bytes": args.max_diagnostic_bytes,
                "max_output_bytes": args.max_output_bytes,
                "max_total_output_bytes": args.max_total_output_bytes,
                "per_file_timeout_seconds": args.timeout,
                "total_timeout_seconds": args.total_timeout,
            },
            "schema": SCHEMA_CONVERSION,
            "source_root": str(source_root),
            "tool": tool_receipt,
        }
        (staging / "cuda-hip-conversion-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory_no_replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "disposition": "MANUAL_REVIEW_REQUIRED",
        "evidence": "translated",
        "files": len(converted),
        "output_root": str(output_root),
        "schema": SCHEMA_CONVERSION,
        "tool": tool_name,
    }


def add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_root")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--exclude", action="append", default=[], help="Relative glob to exclude.")
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-file-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--max-findings", type=int, default=5000)
    parser.add_argument("--max-api-references", type=int, default=5000)


def positive_ints(args: argparse.Namespace) -> None:
    for name in (
        "max_files",
        "max_file_bytes",
        "max_total_bytes",
        "max_findings",
        "max_api_references",
        "max_output_bytes",
        "max_diagnostic_bytes",
        "max_total_output_bytes",
        "total_timeout",
    ):
        if not hasattr(args, name):
            continue
        if getattr(args, name) <= 0:
            raise HoldError("limit_must_be_positive", option=name)
    if getattr(args, "timeout", 1) <= 0:
        raise HoldError("limit_must_be_positive", option="timeout")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory CUDA risks or generate an isolated HIPIFY candidate tree."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Read-only CUDA source and build inventory.")
    add_scan_arguments(inventory)
    inventory.add_argument("--output", type=Path, help="Optional JSON report path.")

    convert = subparsers.add_parser("convert", help="Translate into a new tree outside the source root.")
    add_scan_arguments(convert)
    convert.add_argument("output_root")
    convert.add_argument("--tool", choices=("auto", "hipify-clang", "hipify-perl"), default="auto")
    convert.add_argument("--cuda-path")
    convert.add_argument("--compile-db")
    convert.add_argument("--roc", action="store_true")
    convert.add_argument("--timeout", type=int, default=120)
    convert.add_argument("--total-timeout", type=int, default=1800)
    convert.add_argument("--max-output-bytes", type=int, default=20 * 1024 * 1024)
    convert.add_argument("--max-diagnostic-bytes", type=int, default=5 * 1024 * 1024)
    convert.add_argument("--max-total-output-bytes", type=int, default=200 * 1024 * 1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        positive_ints(args)
        if args.command == "inventory":
            report = scan_source(args)
            emit_json(report, args.output, Path(report["source_root"]))
        else:
            emit_json(convert_source(args))
    except HoldError as exc:
        emit_json(
            {
                "details": exc.details,
                "disposition": "HOLD",
                "reason": exc.reason,
                "schema": "cuda-hip-hold/v1",
            }
        )
        return 2
    except Exception as exc:  # pragma: no cover - last-resort fail-closed boundary
        emit_json(
            {
                "details": {"exception_type": type(exc).__name__},
                "disposition": "HOLD",
                "reason": "unexpected_error",
                "schema": "cuda-hip-hold/v1",
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
