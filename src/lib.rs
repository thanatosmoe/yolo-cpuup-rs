//! # yolo-cpuup-rs
//!
//! CPU-optimized [YOLO](https://docs.ultralytics.com) (v8 / v11) object detection
//! inference in Rust, built on [ONNX Runtime](https://onnxruntime.ai).
//!
//! This crate is a from-scratch rework of the inference path in
//! [`pan93412/yolo-rs`](https://github.com/pan93412/yolo-rs) (MIT / Apache-2.0),
//! focused on squeezing latency out of the CPU path:
//!
//! * SIMD + multi-threaded resizing ([`fast_image_resize`]) instead of the
//!   `image` crate's scalar Catmull-Rom resize.
//! * Lookup-table normalization and a single contiguous planar `NCHW` write
//!   instead of per-pixel `f32` division into a strided `ndarray`.
//! * Cache-friendly, strided-free output decoding straight from ONNX Runtime's
//!   contiguous output slice.
//! * Explicit session graph optimizations, thread tuning and memory-pattern
//!   reuse.
//!
//! ## Example
//!
//! ```no_run
//! use yolo_cpuup::YoloModel;
//!
//! # fn main() -> Result<(), yolo_cpuup::YoloError> {
//! let mut model = YoloModel::from_file("yolo11n.onnx")?;
//! let image = image::open("bus.jpg").unwrap();
//!
//! for detection in model.predict(&image)? {
//!     println!(
//!         "{:>12}  {:.2}  [{:.1}, {:.1}, {:.1}, {:.1}]",
//!         detection.label,
//!         detection.confidence,
//!         detection.bbox.x1,
//!         detection.bbox.y1,
//!         detection.bbox.x2,
//!         detection.bbox.y2,
//!     );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`fast_image_resize`]: https://docs.rs/fast_image_resize

pub mod error;
pub mod labels;
pub mod model;
pub mod postprocess;
pub mod preprocess;
mod types;

pub use error::YoloError;
pub use labels::COCO_LABELS;
pub use model::{ModelOptions, Profile, Timings, YoloModel};
pub use preprocess::{
    preprocess, preprocess_reference_into, PreprocessedInput, Preprocessor, INPUT_SIZE, PLANE,
};
pub use types::{BoundingBox, Detection};
