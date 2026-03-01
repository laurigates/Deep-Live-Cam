#!/usr/bin/env python3
"""Convert GFPGAN ONNX model from FP32 to FP16.

Usage:
    uv run scripts/convert_gfpgan_fp16.py [--input models/gfpgan-1024.onnx] [--output models/gfpgan-1024-fp16.onnx]
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert GFPGAN ONNX model to FP16")
    parser.add_argument(
        "--input",
        default=os.path.join("models", "gfpgan-1024.onnx"),
        help="Path to FP32 ONNX model (default: models/gfpgan-1024.onnx)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("models", "gfpgan-1024-fp16.onnx"),
        help="Path for FP16 output (default: models/gfpgan-1024-fp16.onnx)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input model not found: {args.input}")
        sys.exit(1)

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("Error: Required packages not installed.")
        print("Install with: uv pip install onnxconverter-common")
        sys.exit(1)

    print(f"Loading FP32 model: {args.input}")
    model = onnx.load(args.input)

    print("Converting to FP16 (keeping IO types as FP32)...")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    print(f"Saving FP16 model: {args.output}")
    onnx.save(model_fp16, args.output)

    input_size = os.path.getsize(args.input) / (1024 * 1024)
    output_size = os.path.getsize(args.output) / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100
    print(f"Size: {input_size:.1f} MB -> {output_size:.1f} MB ({reduction:.0f}% reduction)")
    print("Done! Test with: uv run run.py --execution-provider coreml")


if __name__ == "__main__":
    main()
