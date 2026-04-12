"""axropus.cli — Command line interface for the AMF server and benchmarks.

The CLI is intentionally minimal (§7 of the design doc):

    axropus serve [--model MODEL] [--port N] [--config FILE]
    axropus bench [--config FILE]
    axropus --version

No chat UI, no model downloader wizard, no consumer flow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .config import AxropusConfig
from .version import __version__


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="axropus",
        description="Axropus AMF — compressed multi-prefix KV cache for vLLM",
    )
    p.add_argument("--version", action="version", version=f"axropus {__version__}")
    subs = p.add_subparsers(dest="command", required=True)

    # serve
    serve = subs.add_parser("serve", help="Run the AMF HTTP server")
    serve.add_argument("--model", default=None, help="Model identifier to serve")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--config", default=None, help="YAML/JSON config file")
    serve.add_argument("--log-level", default=None)
    serve.add_argument("--api-key", default=None)

    # bench
    bench = subs.add_parser("bench", help="Run benchmark harness")
    bench.add_argument("--config", default="benchmarks/standard.yaml")
    bench.add_argument("--output", default=None)
    bench.add_argument("--quick", action="store_true", help="Run a short smoke benchmark")

    # info
    info = subs.add_parser("info", help="Print system info and exit")

    return p


# ── Commands ────────────────────────────────────────────────────────────────


def _cmd_serve(args: argparse.Namespace) -> int:
    overrides = {}
    if args.model:     overrides["model"]     = args.model
    if args.host:      overrides["host"]      = args.host
    if args.port:      overrides["port"]      = args.port
    if args.log_level: overrides["log_level"] = args.log_level
    if args.api_key:   overrides["api_key"]   = args.api_key

    cfg = AxropusConfig.load(config_file=args.config, overrides=overrides)
    _setup_logging(cfg.log_level)

    logging.getLogger("axropus").info("Starting axropus %s", __version__)
    logging.getLogger("axropus").info("Config:\n%s", cfg.summary())

    try:
        from .server import create_app
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("error: uvicorn is required for `axropus serve`", file=sys.stderr)
        return 2

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from benchmarks.codec_sweep import run_codec_sweep
    from benchmarks.kernel_microbench import run_kernel_microbench

    _setup_logging("INFO")
    logging.getLogger("axropus.bench").info("Starting benchmarks")

    results = {
        "version": __version__,
        "codec_sweep":      run_codec_sweep(quick=args.quick),
        "kernel_microbench": run_kernel_microbench(quick=args.quick),
    }
    text = json.dumps(results, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    _setup_logging("INFO")
    try:
        import torch

        sm = 0
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm = major * 10 + minor
        info = {
            "axropus":         __version__,
            "torch":           torch.__version__,
            "cuda_available":  torch.cuda.is_available(),
            "cuda_device":     torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "sm_version":      sm,
        }
    except ImportError:
        info = {"axropus": __version__, "torch": "not installed"}

    print(json.dumps(info, indent=2))
    return 0


# ── Main entrypoint ─────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "info":
        return _cmd_info(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
