# yolo-cpuup-rs

[![CI](https://github.com/thanatosmoe/yolo-cpuup-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/thanatosmoe/yolo-cpuup-rs/actions/workflows/ci.yml)

CPU-optimized [YOLO](https://docs.ultralytics.com) (v8 / v11) object detection
inference in Rust, built on [ONNX Runtime](https://onnxruntime.ai).

This is a from-scratch rework of the inference pipeline in
[`pan93412/yolo-rs`](https://github.com/pan93412/yolo-rs), tuned for low-latency
**CPU** inference. No Python, no PyTorch, no GPU required.

## Why it's faster

The upstream pipeline is clean and correct but leaves a lot of CPU performance on
the table. The main changes:

| # | Change | Where |
|---|--------|-------|
| 1 | **SIMD + multi-threaded resize** (`fast_image_resize`, Catmull-Rom) instead of the `image` crate's scalar resize | [`src/preprocess.rs`](src/preprocess.rs) |
| 2 | **Lookup-table normalization** — no per-pixel `f32` division | [`src/preprocess.rs`](src/preprocess.rs) |
| 3 | **Single contiguous planar NCHW write** instead of strided `ndarray` indexing | [`src/preprocess.rs`](src/preprocess.rs) |
| 4 | **Cache-friendly decode** straight from ONNX Runtime's contiguous output slice (no `reversed_axes`-style strided iteration) | [`src/postprocess.rs`](src/postprocess.rs) |
| 5 | **Explicit session tuning**: graph optimizations, intra/inter-op threads, memory-pattern reuse, explicit CPU EP | [`src/model.rs`](src/model.rs) |
| 6 | **Optional INT8 quantization** of the ONNX graph (biggest model-level CPU win) | [`scripts/quantize_int8.py`](scripts/quantize_int8.py) |

All of this is transparent: the preprocessing change is verified against the
original pipeline in [`examples/parity.rs`](examples/parity.rs).

## Benchmark

Measured on the bundled YOLO11n model (`yolo11n.onnx`) and `bus.jpg` (810x1080),
50 timed iterations after 5 warmups, `cargo build --release`.

<!-- BENCHMARK:BEGIN -->
Environment: WSL2, Intel Core Ultra 9 275HX (**2 vCPUs available**), model
YOLO11n (`yolo11n.onnx`), `bus.jpg` (810x1080), 100 timed iterations after 5
warmups, `cargo build --release`. Only 2 vCPUs are exposed to the VM, so the
absolute inference numbers are high; the speedups are the point.

| Stage | Upstream FP32 | Optimized FP32 | Optimized INT8 |
|-------|--------------:|---------------:|---------------:|
| Preprocess | 62.78 ms | **6.24 ms** | 6.46 ms |
| Inference + postprocess | 274.95 ms | 187.41 ms | **153.39 ms** |
| **Total** | **337.73 ms** | **193.65 ms** | **159.84 ms** |
| vs. upstream | 1.00x | **1.74x** | **2.11x** |
| Model size | 10.7 MB | 10.7 MB | **3.0 MB** |

* Preprocessing speedup comes from the SIMD/multi-threaded resize and the LUT +
  contiguous NCHW write.
* Inference speedup comes from session tuning (notably intra/inter-op thread
  settings — the default inter-op pool oversubscribes a small VM), plus optional
  INT8 quantization for a further ~20%.
* Detections are stable: `examples/parity` reports the same 5 boxes (same class,
  within 2px) for both preprocessing paths, with a max tensor difference of
  `0.0235` and a mean difference of `0.0007`.
<!-- BENCHMARK:END -->

Reproduce with this crate's own benchmark example:

```bash
cargo run --release --example bench -- yolo11n.onnx bus.jpg 100
```

It prints average preprocess / inference / postprocess / total times over the
timed iterations.

## Requirements

* Rust 1.87+
* A C compiler
* On Linux, `pkg-config` and OpenSSL development headers:

  ```bash
  sudo apt install build-essential pkg-config libssl-dev
  ```

  If you cannot install system packages, build with the `vendored-tls` feature
  instead and OpenSSL is compiled from source:

  ```bash
  cargo build --release --features vendored-tls
  ```

## Quick start (library)

```rust
use yolo_cpuup::YoloModel;

fn main() -> Result<(), yolo_cpuup::YoloError> {
    let mut model = YoloModel::from_file("yolo11n.onnx")?;
    let image = image::open("bus.jpg").unwrap();

    for detection in model.predict(&image)? {
        println!(
            "{:>12}  {:.2}  [{:.1}, {:.1}, {:.1}, {:.1}]",
            detection.label,
            detection.confidence,
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        );
    }
    Ok(())
}
```

Tune the session for your CPU:

```rust
use yolo_cpuup::{ModelOptions, YoloModel};

let mut model = YoloModel::from_file_with(
    "yolo11n.onnx",
    &ModelOptions { intra_threads: 4, ..ModelOptions::default() },
)?;
```

## CLI

```bash
# Detect on one image
cargo run --release --example detect -- yolo11n.onnx bus.jpg

# Per-stage timing breakdown
cargo run --release --example detect -- yolo11n.onnx bus.jpg --profile 100

# Correctness check against the reference preprocessing
cargo run --release --example parity -- yolo11n.onnx bus.jpg
```

## Exporting a model

Export any Ultralytics YOLOv8 / YOLO11 detection model to ONNX (Python side,
only needed once):

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=onnx
```

## Faster CPU inference with INT8

Dynamic quantization shrinks the graph and speeds up CPU inference further:

```bash
pip install onnxruntime onnx
python scripts/quantize_int8.py yolo11n.onnx          # -> yolo11n.int8.onnx
cargo run --release --example detect -- yolo11n.int8.onnx bus.jpg --profile 100
```

## Project layout

```
src/
  lib.rs          crate docs + re-exports
  model.rs        session configuration + high-level inference API
  preprocess.rs   fast resize + LUT normalization (NCHW)
  postprocess.rs  output decoding + NMS
  labels.rs       COCO label table
  error.rs        error type
examples/
  detect.rs       CLI
  bench.rs        single-line benchmark
  parity.rs       fast-vs-reference correctness check
scripts/
  quantize_int8.py
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option — the same terms as the upstream project.

This project is a derivative work of
[`pan93412/yolo-rs`](https://github.com/pan93412/yolo-rs) by Yi-Jyun Pan
(MIT / Apache-2.0). See [`NOTICE`](NOTICE). The YOLO model weights / ONNX graphs
are licensed separately by [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0).
