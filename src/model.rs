//! Model loading, session configuration and the high level inference API.

use std::path::Path;
use std::time::{Duration, Instant};

use arcstr::ArcStr;
use image::DynamicImage;
use ort::ep::CPU;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::TensorRef;
use ort::inputs;

use crate::error::YoloError;
use crate::labels::COCO_LABELS;
use crate::postprocess;
use crate::preprocess::{PreprocessedInput, Preprocessor};

/// Tunables for the ONNX Runtime session.
///
/// All fields have sensible CPU-oriented defaults; [`ModelOptions::default`] is
/// usually all you need.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Number of intra-op threads. `0` lets ONNX Runtime pick (all cores).
    ///
    /// For small models this is worth tuning: on many machines one thread per
    /// physical performance core beats "use everything".
    pub intra_threads: usize,
    /// Number of inter-op threads. Defaults to `1`, which is right for a single
    /// sequential inference stream.
    pub inter_threads: usize,
    /// Graph optimization level. Defaults to [`GraphOptimizationLevel::Level3`]
    /// (enable all).
    pub optimization_level: GraphOptimizationLevel,
    /// Register the CPU execution provider explicitly so a GPU provider enabled
    /// elsewhere in the process does not silently take over. Defaults to `true`.
    pub cpu_only: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            intra_threads: 0,
            inter_threads: 1,
            optimization_level: GraphOptimizationLevel::Level3,
            cpu_only: true,
        }
    }
}

/// Per-stage timings produced by [`YoloModel::predict_profiled`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub preprocess: Duration,
    pub inference: Duration,
    pub postprocess: Duration,
}

impl Timings {
    /// Total time of all stages.
    pub fn total(&self) -> Duration {
        self.preprocess + self.inference + self.postprocess
    }
}

/// Result of [`YoloModel::predict_profiled`].
#[derive(Debug, Clone)]
pub struct Profile {
    pub timings: Timings,
    pub detections: Vec<crate::Detection>,
}

/// A loaded YOLO model ready to run inference.
pub struct YoloModel {
    session: Session,
    labels: Vec<ArcStr>,
    preprocessor: Preprocessor,
    /// Minimum class confidence to keep a detection. Default `0.5`.
    pub probability_threshold: f32,
    /// IoU threshold for non-maximum suppression. Default `0.7`.
    pub iou_threshold: f32,
}

impl std::fmt::Debug for YoloModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YoloModel")
            .field("num_labels", &self.labels.len())
            .field("probability_threshold", &self.probability_threshold)
            .field("iou_threshold", &self.iou_threshold)
            .finish_non_exhaustive()
    }
}

impl YoloModel {
    /// Load a YOLOv8 / YOLO11 ONNX model with default [`ModelOptions`] and the
    /// standard COCO label table.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, YoloError> {
        Self::from_file_with(path, &ModelOptions::default())
    }

    /// Load a model with explicit [`ModelOptions`].
    pub fn from_file_with(
        path: impl AsRef<Path>,
        options: &ModelOptions,
    ) -> Result<Self, YoloError> {
        let session = build_session(path.as_ref(), options)?;
        Ok(Self::with_session(session))
    }

    /// Wrap an already-built ONNX Runtime session.
    pub fn with_session(session: Session) -> Self {
        Self {
            session,
            labels: COCO_LABELS.iter().map(|s| ArcStr::from(*s)).collect(),
            preprocessor: Preprocessor::new(),
            probability_threshold: 0.5,
            iou_threshold: 0.7,
        }
    }

    /// Replace the label table (for custom-trained models).
    pub fn with_labels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<ArcStr>,
    {
        self.labels = labels.into_iter().map(Into::into).collect();
        self
    }

    /// The current label table.
    pub fn labels(&self) -> &[ArcStr] {
        &self.labels
    }

    /// Access the underlying ONNX Runtime session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Mutably access the underlying ONNX Runtime session.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Run the full pipeline (preprocess + inference + postprocess) on an image.
    pub fn predict(&mut self, image: &DynamicImage) -> Result<Vec<crate::Detection>, YoloError> {
        let mut input = PreprocessedInput::new();
        self.predict_into(image, &mut input)
    }

    /// Like [`YoloModel::predict`], but reuses a caller-owned input buffer so no
    /// per-frame allocation happens.
    pub fn predict_into(
        &mut self,
        image: &DynamicImage,
        input: &mut PreprocessedInput,
    ) -> Result<Vec<crate::Detection>, YoloError> {
        self.preprocessor.run_into(image, input)?;
        self.detect(input)
    }

    /// Run inference on an already preprocessed input.
    pub fn detect(
        &mut self,
        input: &PreprocessedInput,
    ) -> Result<Vec<crate::Detection>, YoloError> {
        Ok(self.detect_timed(input)?.0)
    }

    fn detect_timed(
        &mut self,
        input: &PreprocessedInput,
    ) -> Result<(Vec<crate::Detection>, Duration, Duration), YoloError> {
        let tensor = TensorRef::from_array_view((input.shape(), input.data.as_slice()))
            .map_err(|e| YoloError::InputTensor(e.to_string()))?;

        let started = Instant::now();
        let outputs = self
            .session
            .run(inputs!["images" => tensor])
            .map_err(|e| YoloError::Inference(e.to_string()))?;
        let inference = started.elapsed();

        let started = Instant::now();
        let (shape, data) = outputs["output0"]
            .try_extract_tensor::<f32>()
            .map_err(|e| YoloError::OutputExtract(e.to_string()))?;
        let shape = shape.to_vec();
        let detections = postprocess::decode(
            data,
            &shape,
            &self.labels,
            self.probability_threshold,
            self.iou_threshold,
            input.scale(),
        )?;
        let postprocess = started.elapsed();

        Ok((detections, inference, postprocess))
    }

    /// Run the pipeline `warmup + runs` times and average the per-stage timings.
    ///
    /// Useful for benchmarking; the returned detections are from the last run.
    pub fn predict_profiled(
        &mut self,
        image: &DynamicImage,
        warmup: usize,
        runs: usize,
    ) -> Result<Profile, YoloError> {
        let runs = runs.max(1);
        let mut input = PreprocessedInput::new();

        for _ in 0..warmup {
            self.predict_into(image, &mut input)?;
        }

        let mut preprocess = Duration::ZERO;
        let mut inference = Duration::ZERO;
        let mut postprocess = Duration::ZERO;
        let mut detections = Vec::new();

        for _ in 0..runs {
            let started = Instant::now();
            self.preprocessor.run_into(image, &mut input)?;
            preprocess += started.elapsed();

            let (current, infer, post) = self.detect_timed(&input)?;
            inference += infer;
            postprocess += post;
            detections = current;
        }

        let n = runs as u32;
        Ok(Profile {
            timings: Timings {
                preprocess: preprocess / n,
                inference: inference / n,
                postprocess: postprocess / n,
            },
            detections,
        })
    }
}

fn build_session(path: &Path, options: &ModelOptions) -> Result<Session, YoloError> {
    let mut builder: SessionBuilder = Session::builder().map_err(config_err)?;

    builder = builder
        .with_optimization_level(options.optimization_level)
        .map_err(config_err)?;

    if options.intra_threads > 0 {
        builder = builder
            .with_intra_threads(options.intra_threads)
            .map_err(config_err)?;
    }

    builder = builder
        .with_inter_threads(options.inter_threads.max(1))
        .map_err(config_err)?;

    // Reusing the input memory between runs is a small but free CPU win.
    builder = builder.with_memory_pattern(true).map_err(config_err)?;

    if options.cpu_only {
        // NOTE: `ep::CPU::default()` has `use_arena = false`, which *disables*
        // the ONNX Runtime memory arena and can make inference several times
        // slower. Explicitly keep it enabled to match ORT's default behaviour.
        builder = builder
            .with_execution_providers([CPU::default().with_arena_allocator(true).build()])
            .map_err(config_err)?;
    }

    builder
        .commit_from_file(path)
        .map_err(|e| YoloError::SessionLoad(e.to_string()))
}

#[inline]
fn config_err(e: impl std::fmt::Display) -> YoloError {
    YoloError::SessionConfig(e.to_string())
}
