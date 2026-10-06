//! Correctness check: fast preprocessing vs. the upstream-compatible reference.
//!
//! ```text
//! cargo run --release --example parity -- yolo11n.onnx bus.jpg
//! ```
//!
//! It compares the two input tensors numerically and then checks that the
//! detections produced from them agree (same class, matching box within 2 px).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use yolo_cpuup::preprocess::{preprocess_reference_into, PreprocessedInput, Preprocessor};
use yolo_cpuup::{Detection, YoloModel};

#[derive(Parser, Debug)]
#[command(about = "Compare optimized preprocessing against the reference pipeline")]
struct Args {
    model: PathBuf,
    image: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let image = image::open(&args.image)
        .with_context(|| format!("failed to open image {}", args.image.display()))?;

    let mut fast = PreprocessedInput::new();
    Preprocessor::new().run_into(&image, &mut fast)?;

    let mut reference = PreprocessedInput::new();
    preprocess_reference_into(&image, &mut reference)?;

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (a, b) in fast.data.iter().zip(reference.data.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / fast.data.len() as f64;

    println!("input tensor (NCHW f32, {} elements)", fast.data.len());
    println!("  max |fast - reference| : {max_abs:.6}");
    println!("  mean|fast - reference| : {mean_abs:.8}");

    let mut model = YoloModel::from_file(&args.model)
        .with_context(|| format!("failed to load model {}", args.model.display()))?;

    let fast_detections = model.detect(&fast)?;
    let reference_detections = model.detect(&reference)?;

    println!();
    println!("detections: fast={} reference={}", fast_detections.len(), reference_detections.len());

    let mut matched = 0usize;
    for reference_detection in &reference_detections {
        if fast_detections
            .iter()
            .any(|d| close_enough(d, reference_detection))
        {
            matched += 1;
        }
    }
    println!(
        "matched (same class, box within 2px): {matched}/{}",
        reference_detections.len()
    );

    Ok(())
}

fn close_enough(a: &Detection, b: &Detection) -> bool {
    a.class_id == b.class_id
        && (a.confidence - b.confidence).abs() < 0.05
        && (a.bbox.x1 - b.bbox.x1).abs() < 2.0
        && (a.bbox.y1 - b.bbox.y1).abs() < 2.0
        && (a.bbox.x2 - b.bbox.x2).abs() < 2.0
        && (a.bbox.y2 - b.bbox.y2).abs() < 2.0
}
