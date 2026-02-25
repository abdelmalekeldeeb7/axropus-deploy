#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="platform/engine/cpp")
    p.add_argument("--build-dir", default="build/engine-cuda")
    p.add_argument("--cuda", action="store_true", default=True)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    src = (root / args.source).resolve()
    bld = (root / args.build_dir).resolve()
    bld.mkdir(parents=True, exist_ok=True)

    cmake_args = ["cmake", "-S", str(src), "-B", str(bld)]
    if args.cuda:
        cmake_args.append("-DKORITH_ENGINE_USE_CUDA=ON")
    run(cmake_args, cwd=root)
    run(["cmake", "--build", str(bld), "-j"], cwd=root)


if __name__ == "__main__":
    main()

