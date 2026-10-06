// Benchmark for the optimized pipeline.
//
// cargo run --release --features vendored-tls --example bench -- <model.onnx> <image> [runs]

use std::path::PathBuf;

use anyhow::Result;
use yolo_cpuup::YoloModel;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().expect("usage: bench <model.onnx> <image> [runs]"));
    let image_path = PathBuf::from(args.next().expect("usage: bench <model.onnx> <image> [runs]"));
    let runs: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(50);

    let image = image::open(&image_path)?;
    let mut model = YoloModel::from_file(&model_path)?;

    let profile = model.predict_profiled(&image, 5, runs)?;
    let t = profile.timings;
    println!(
        "optimized  preprocess {:8.3} ms  inference {:8.3} ms  postprocess {:8.3} ms  total {:8.3} ms  detections {}",
        t.preprocess.as_secs_f64() * 1e3,
        t.inference.as_secs_f64() * 1e3,
        t.postprocess.as_secs_f64() * 1e3,
        t.total().as_secs_f64() * 1e3,
        profile.detections.len()
    );

    Ok(())
}
