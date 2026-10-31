#!/usr/bin/env python3
"""Command-line entry point for the MRI preprocessing pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "preprocess"

STAGES = {
    "convert": "convert.py",
    "register": "register.py",
    "normalize256": "normalize_256.py",
    "normalize512": "normalize_512.py",
    "generate256": "generate_lr_hr_256.py",
    "generate512": "generate_lr_hr_512.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MRI preprocessing pipeline with configurable paths."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", *STAGES],
        default=["all"],
        help="Stages to run. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def load_local_env(env: dict[str, str]) -> dict[str, str]:
    """Load config.local.env without requiring an extra dependency."""
    config_path = PROJECT_ROOT / "config.local.env"
    if not config_path.exists():
        return env

    loaded = dict(env)
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return loaded


def resolve_stages(selected: list[str]) -> list[str]:
    if "all" in selected:
        return list(STAGES)
    return selected


def run_stage(stage: str, env: dict[str, str], dry_run: bool = False) -> None:
    script = PREPROCESS_DIR / STAGES[stage]
    command = [sys.executable, str(script)]
    print(f"\n[RUNNING] {stage}: {' '.join(command)}")
    if dry_run:
        return

    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
    print(f"[SUCCESS] {stage}")


def main() -> int:
    args = build_parser().parse_args()
    env = load_local_env(dict(os.environ))

    for stage in resolve_stages(args.stages):
        try:
            run_stage(stage, env, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(
                f"[FAILED] {stage} exited with code {exc.returncode}",
                file=sys.stderr,
            )
            return exc.returncode or 1

    print("\n[PIPELINE FINISHED]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
