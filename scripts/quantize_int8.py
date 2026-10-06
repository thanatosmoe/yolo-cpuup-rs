#!/usr/bin/env python3
"""Dynamically quantize a YOLO ONNX model to INT8 for faster CPU inference.

This is the single biggest model-level CPU win: dynamic INT8 quantization
typically makes the ONNX graph 2-4x smaller and, on CPU, meaningfully faster,
at the cost of a small accuracy drop. The Rust side needs no changes -- it loads
the quantized `*.int8.onnx` like any other model.

Usage:
    pip install onnxruntime
    python scripts/quantize_int8.py yolo11n.onnx
    # -> yolo11n.int8.onnx

    cargo run --release --example detect -- yolo11n.int8.onnx bus.jpg --profile 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="input FP32 ONNX model")
    parser.add_argument("-o", "--output", type=Path, help="output path (default: <stem>.int8.onnx)")
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="per-channel weight quantization (better accuracy, may be slower)",
    )
    args = parser.parse_args()

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        print(
            f"error: {exc}. Install the dependencies with:\n"
            "  pip install onnxruntime onnx",
            file=sys.stderr,
        )
        return 1

    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1

    output = args.output or args.model.with_suffix(".int8.onnx")

    quantize_dynamic(
        model_input=str(args.model),
        model_output=str(output),
        weight_type=QuantType.QUInt8,
        per_channel=args.per_channel,
    )

    before = args.model.stat().st_size / 1e6
    after = output.stat().st_size / 1e6
    print(f"wrote {output} ({before:.1f} MB -> {after:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
