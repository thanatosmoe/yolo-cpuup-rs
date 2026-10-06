//! Minimal CPU inference CLI.
//!
//! ```text
//! cargo run --release --example detect -- yolo11n.onnx bus.jpg
//! cargo run --release --example detect -- yolo11n.onnx bus.jpg --profile 50
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use yolo_cpuup::{Detection, ModelOptions, YoloModel};

#[derive(Parser, Debug)]
#[command(version, about = "CPU-optimized YOLO (v8/v11) inference")]
struct Args {
    /// Path to a YOLO v8/v11 ONNX model.
    model: PathBuf,
    /// Path to the input image.
    image: PathBuf,
    /// Minimum class confidence to keep a detection.
    #[arg(long, default_value_t = 0.5)]
    conf: f32,
    /// IoU threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.7)]
    iou: f32,
    /// ONNX Runtime intra-op threads (0 = runtime default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Timed iterations after 3 warmups; 0 disables profiling.
    #[arg(long, default_value_t = 0)]
    profile: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let image = image::open(&args.image)
        .with_context(|| format!("failed to open image {}", args.image.display()))?;

    let options = ModelOptions {
        intra_threads: args.threads,
        ..ModelOptions::default()
    };
    let mut model = YoloModel::from_file_with(&args.model, &options)
        .with_context(|| format!("failed to load model {}", args.model.display()))?;
    model.probability_threshold = args.conf;
    model.iou_threshold = args.iou;

    println!(
        "image {}x{}  model={}",
        image.width(),
        image.height(),
        args.model.display()
    );

    if args.profile > 0 {
        let profile = model.predict_profiled(&image, 3, args.profile)?;
        let t = profile.timings;
        let total = t.total().as_secs_f64() * 1e3;
        println!("runs            : {}", args.profile);
        println!(
            "preprocess      : {:>8.3} ms",
            t.preprocess.as_secs_f64() * 1e3
        );
        println!(
            "inference       : {:>8.3} ms",
            t.inference.as_secs_f64() * 1e3
        );
        println!(
            "postprocess     : {:>8.3} ms",
            t.postprocess.as_secs_f64() * 1e3
        );
        println!("total           : {:>8.3} ms", total);
        println!("throughput      : {:>8.1} FPS", 1e3 / total);
        println!();
        for detection in &profile.detections {
            println!("{}", format_detection(detection));
        }
    } else {
        for detection in model.predict(&image)? {
            println!("{}", format_detection(&detection));
        }
    }

    Ok(())
}

fn format_detection(detection: &Detection) -> String {
    format!(
        "{:>14}  {:.3}  [{:>7.1}, {:>7.1}, {:>7.1}, {:>7.1}]",
        detection.label,
        detection.confidence,
        detection.bbox.x1,
        detection.bbox.y1,
        detection.bbox.x2,
        detection.bbox.y2,
    )
}
