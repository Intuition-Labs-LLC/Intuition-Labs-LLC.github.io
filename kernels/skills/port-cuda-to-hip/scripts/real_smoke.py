#!/usr/bin/env python3
"""Run the bundled CUDA-to-HIP translation, gfx compile, and device parity smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rocm_port import HoldError, run_bounded


SKILL_ROOT = Path(__file__).resolve().parents[1]
PORT_HELPER = SKILL_ROOT / "scripts" / "rocm_port.py"
SMOKE_SOURCE_ROOT = SKILL_ROOT / "assets" / "smoke"
PASS_PATTERN = re.compile(
    r"^CUDA_HIP_SMOKE PASS count=(?P<count>\d+) "
    r"expected_sum=(?P<expected>-?\d+) observed_sum=(?P<observed>-?\d+)$"
)


class SmokeHold(RuntimeError):
    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    timeout: int,
    output_limit: int = 1024 * 1024,
    watched_file: Path | None = None,
    file_limit: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return run_bounded(
            command,
            timeout=timeout,
            stdout_limit=output_limit,
            stderr_limit=output_limit,
            timeout_reason="smoke_command_timeout",
            limit_reason="smoke_command_output_limit_exceeded",
            watched_file=watched_file,
            file_limit=file_limit,
        )
    except HoldError as exc:
        raise SmokeHold(exc.reason, command=command[0], **exc.details) from exc


def first_version(binary: str) -> str:
    result = run([binary, "--version"], timeout=10, output_limit=128 * 1024)
    if result.returncode != 0:
        raise SmokeHold("smoke_tool_version_failed", tool=binary, exit_code=result.returncode)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    line = next((value.strip() for value in output.splitlines() if value.strip()), "")
    if not line:
        raise SmokeHold("smoke_tool_version_missing", tool=binary)
    return line[:200]


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SmokeHold("smoke_tool_missing", tool=name)
    return path


def cuda_version(cuda_path: Path) -> int:
    header = cuda_path / "include" / "cuda.h"
    if not header.is_file():
        raise SmokeHold("smoke_cuda_header_missing", path=str(header))
    match = re.search(
        r"^\s*#\s*define\s+CUDA_VERSION\s+(\d+)\s*$",
        header.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if match is None:
        raise SmokeHold("smoke_cuda_version_missing", path=str(header))
    version = int(match.group(1))
    if version < 7000 or version >= 13000:
        raise SmokeHold("smoke_cuda_version_outside_hipify_range", cuda_version=version)
    return version


def parse_json_stdout(result: subprocess.CompletedProcess[bytes], stage: str) -> dict[str, Any]:
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeHold("smoke_stage_receipt_invalid", stage=stage) from exc
    if result.returncode != 0:
        raise SmokeHold(
            "smoke_stage_held",
            stage=stage,
            exit_code=result.returncode,
            receipt=value,
        )
    if not isinstance(value, dict):
        raise SmokeHold("smoke_stage_receipt_invalid", stage=stage)
    return value


def hipify_statistics(
    hipify_clang: str,
    source: Path,
    cuda_path: Path,
    timeout: int,
) -> dict[str, Any]:
    result = run(
        [
            hipify_clang,
            "--no-output",
            "--print-stats",
            str(source),
            f"--cuda-path={cuda_path}",
        ],
        timeout=timeout,
        output_limit=5 * 1024 * 1024,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise SmokeHold(
            "smoke_hipify_statistics_failed",
            exit_code=result.returncode,
            output_sha256=hashlib.sha256(output).hexdigest(),
        )
    text = output.decode("utf-8", errors="replace")

    def integer(label: str) -> int:
        match = re.search(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", text, re.MULTILINE)
        if match is None:
            raise SmokeHold("smoke_hipify_statistics_missing", label=label)
        return int(match.group(1))

    return {
        "changed_lines": integer("CHANGED lines of code"),
        "converted_references": integer("CONVERTED refs count"),
        "output_lines": len(output.splitlines()),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "total_bytes": integer("TOTAL bytes"),
        "total_lines": integer("TOTAL lines of code"),
        "unconverted_references": integer("UNCONVERTED refs count"),
        "warning_lines": sum(
            1 for line in text.splitlines() if line.lstrip().startswith("warning:")
        ),
    }


def validate_profile_trace(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SmokeHold("smoke_profile_json_missing", path=path.name)
    size = path.stat().st_size
    if size <= 0 or size > 50 * 1024 * 1024:
        raise SmokeHold("smoke_profile_size_invalid", bytes=size)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeHold("smoke_profile_json_invalid", path=path.name) from exc
    sessions = document.get("rocprofiler-sdk-tool") if isinstance(document, dict) else None
    if not isinstance(sessions, list) or not sessions:
        raise SmokeHold("smoke_profile_schema_invalid")

    for session in sessions:
        if not isinstance(session, dict):
            continue
        agents = session.get("agents")
        symbols = session.get("kernel_symbols")
        records = session.get("buffer_records")
        if not isinstance(agents, list) or not isinstance(symbols, list) or not isinstance(records, dict):
            continue
        gpu_agents = {
            agent.get("id", {}).get("handle"): agent
            for agent in agents
            if isinstance(agent, dict)
            and agent.get("type") == 2
            and isinstance(agent.get("id"), dict)
            and isinstance(agent["id"].get("handle"), int)
        }
        smoke_symbols = {
            symbol.get("kernel_id"): symbol
            for symbol in symbols
            if isinstance(symbol, dict)
            and isinstance(symbol.get("kernel_id"), int)
            and (
                symbol.get("truncated_kernel_name") == "add_bias"
                or (
                    isinstance(symbol.get("formatted_kernel_name"), str)
                    and symbol["formatted_kernel_name"].startswith("add_bias(")
                )
            )
        }
        dispatches = records.get("kernel_dispatch")
        if not isinstance(dispatches, list):
            continue
        for dispatch in dispatches:
            if not isinstance(dispatch, dict):
                continue
            info = dispatch.get("dispatch_info")
            if not isinstance(info, dict):
                continue
            agent_reference = info.get("agent_id")
            if not isinstance(agent_reference, dict):
                continue
            agent_id = agent_reference.get("handle")
            kernel_id = info.get("kernel_id")
            if agent_id not in gpu_agents or kernel_id not in smoke_symbols:
                continue
            if not isinstance(dispatch.get("start_timestamp"), int) or not isinstance(
                dispatch.get("end_timestamp"), int
            ) or dispatch["end_timestamp"] < dispatch["start_timestamp"]:
                continue
            agent = gpu_agents[agent_id]
            symbol = smoke_symbols[kernel_id]
            return {
                "agent": {
                    "gfx_target_version": agent.get("gfx_target_version"),
                    "handle": agent_id,
                    "name": agent.get("name"),
                    "vendor_id": agent.get("vendor_id"),
                },
                "dispatch_id": info.get("dispatch_id"),
                "kernel": {
                    "id": kernel_id,
                    "name": symbol.get("formatted_kernel_name")
                    or symbol.get("truncated_kernel_name"),
                },
                "record_counts": {
                    name: len(records.get(name, []))
                    if isinstance(records.get(name), list)
                    else 0
                    for name in ("hip_api", "kernel_dispatch", "memory_copy")
                },
            }
    raise SmokeHold("smoke_profile_kernel_dispatch_missing", kernel="add_bias")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-path", type=Path, default=os.environ.get("CUDA_PATH"))
    parser.add_argument("--gfx", default="gfx1151")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if not re.fullmatch(r"gfx[0-9a-f]+", args.gfx):
            raise SmokeHold("smoke_gfx_invalid", gfx=args.gfx)
        if args.timeout <= 0:
            raise SmokeHold("smoke_timeout_invalid")
        if args.cuda_path is None:
            raise SmokeHold("smoke_cuda_path_missing")
        resolved_cuda_path = args.cuda_path.expanduser().resolve()
        observed_cuda_version = cuda_version(resolved_cuda_path)
        hipify_clang = require_tool("hipify-clang")
        hipcc = require_tool("hipcc")
        rocminfo = require_tool("rocminfo")
        rocprofv3 = require_tool("rocprofv3") if args.profile else None

        rocminfo_result = run([rocminfo], timeout=30, output_limit=5 * 1024 * 1024)
        rocminfo_output = (rocminfo_result.stdout + rocminfo_result.stderr).decode(
            "utf-8",
            errors="replace",
        )
        if rocminfo_result.returncode != 0 or args.gfx not in rocminfo_output:
            raise SmokeHold(
                "smoke_expected_gfx_not_observed",
                gfx=args.gfx,
                rocminfo_exit_code=rocminfo_result.returncode,
            )

        with tempfile.TemporaryDirectory(prefix="cuda-hip-real-smoke.") as directory:
            work_root = Path(directory)
            candidate_root = work_root / "hip-candidate"
            inventory_result = run(
                [
                    sys.executable,
                    str(PORT_HELPER),
                    "inventory",
                    str(SMOKE_SOURCE_ROOT),
                ],
                timeout=args.timeout,
            )
            inventory = parse_json_stdout(inventory_result, "inventory")
            statistics = hipify_statistics(
                hipify_clang,
                SMOKE_SOURCE_ROOT / "cuda_hip_smoke.cu",
                resolved_cuda_path,
                args.timeout,
            )
            conversion_result = run(
                [
                    sys.executable,
                    str(PORT_HELPER),
                    "convert",
                    str(SMOKE_SOURCE_ROOT),
                    str(candidate_root),
                    "--tool",
                    "hipify-clang",
                    "--cuda-path",
                    str(resolved_cuda_path),
                    "--timeout",
                    str(args.timeout),
                    "--total-timeout",
                    str(args.timeout),
                ],
                timeout=args.timeout + 15,
            )
            conversion = parse_json_stdout(conversion_result, "translation")
            hip_source = candidate_root / "cuda_hip_smoke.hip"
            if not hip_source.is_file():
                raise SmokeHold("smoke_translated_source_missing")
            executable = work_root / "cuda-hip-smoke"
            compile_result = run(
                [
                    hipcc,
                    "-O2",
                    f"--offload-arch={args.gfx}",
                    str(hip_source),
                    "-o",
                    str(executable),
                ],
                timeout=args.timeout,
                output_limit=5 * 1024 * 1024,
            )
            if compile_result.returncode != 0:
                raise SmokeHold(
                    "smoke_compile_failed",
                    diagnostic_sha256=hashlib.sha256(compile_result.stderr).hexdigest(),
                    diagnostic_lines=len(compile_result.stderr.splitlines()),
                    exit_code=compile_result.returncode,
                )
            execution = run([str(executable)], timeout=args.timeout)
            output = execution.stdout.decode("utf-8", errors="replace").strip()
            match = PASS_PATTERN.fullmatch(output)
            if execution.returncode != 0 or match is None:
                raise SmokeHold(
                    "smoke_device_execution_or_parity_failed",
                    exit_code=execution.returncode,
                    output_sha256=hashlib.sha256(execution.stdout + execution.stderr).hexdigest(),
                )
            if match.group("expected") != match.group("observed"):
                raise SmokeHold("smoke_reference_mismatch")

            profile_receipt = None
            if args.profile:
                assert rocprofv3 is not None
                profile_root = work_root / "profile"
                profile_root.mkdir()
                profile_json = profile_root / "cuda-hip-smoke_results.json"
                profile_result = run(
                    [
                        rocprofv3,
                        "--output-directory",
                        str(profile_root),
                        "--output-file",
                        "cuda-hip-smoke",
                        "--output-format",
                        "json",
                        "--hip-runtime-trace",
                        "--kernel-trace",
                        "--memory-copy-trace",
                        "--stats",
                        "--summary",
                        "--",
                        str(executable),
                    ],
                    timeout=args.timeout,
                    output_limit=5 * 1024 * 1024,
                    watched_file=profile_json,
                    file_limit=50 * 1024 * 1024,
                )
                if profile_result.returncode != 0:
                    raise SmokeHold(
                        "smoke_profile_failed",
                        exit_code=profile_result.returncode,
                        output_sha256=hashlib.sha256(
                            profile_result.stdout + profile_result.stderr
                        ).hexdigest(),
                    )
                profile_files = sorted(path for path in profile_root.rglob("*") if path.is_file())
                if not profile_files or len(profile_files) > 100:
                    raise SmokeHold(
                        "smoke_profile_file_count_invalid",
                        count=len(profile_files),
                    )
                total_profile_bytes = sum(path.stat().st_size for path in profile_files)
                if total_profile_bytes <= 0 or total_profile_bytes > 50 * 1024 * 1024:
                    raise SmokeHold(
                        "smoke_profile_size_invalid",
                        bytes=total_profile_bytes,
                    )
                trace_receipt = validate_profile_trace(profile_json)
                profile_receipt = {
                    "files": [
                        {
                            "bytes": path.stat().st_size,
                            "path": path.relative_to(profile_root).as_posix(),
                            "sha256": sha256_file(path),
                        }
                        for path in profile_files
                    ],
                    "output_sha256": hashlib.sha256(
                        profile_result.stdout + profile_result.stderr
                    ).hexdigest(),
                    "tool": first_version(rocprofv3),
                    "total_bytes": total_profile_bytes,
                    "trace": trace_receipt,
                }

            evidence = [
                "scanned",
                "translated",
                "compiled",
                "linked",
                "executed-on-amd",
                "reference-matched",
            ]
            if profile_receipt is not None:
                evidence.append("profiled")
            receipt = {
                "artifacts": {
                    "candidate_sha256": sha256_file(hip_source),
                    "executable_sha256": sha256_file(executable),
                    "source_sha256": sha256_file(SMOKE_SOURCE_ROOT / "cuda_hip_smoke.cu"),
                },
                "cuda_version": observed_cuda_version,
                "disposition": "REFERENCE_MATCHED",
                "evidence": evidence,
                "gfx": args.gfx,
                "inventory_summary": inventory["summary"],
                "hipify_statistics": statistics,
                "output": {
                    "count": int(match.group("count")),
                    "expected_sum": int(match.group("expected")),
                    "observed_sum": int(match.group("observed")),
                },
                "profile": profile_receipt,
                "schema": "cuda-hip-real-smoke/v1",
                "tools": {
                    "hipcc": first_version(hipcc),
                    "hipify_clang": first_version(hipify_clang),
                },
                "translation": conversion,
            }
    except SmokeHold as exc:
        print(
            json.dumps(
                {
                    "details": exc.details,
                    "disposition": "HOLD",
                    "reason": exc.reason,
                    "schema": "cuda-hip-real-smoke-hold/v1",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
