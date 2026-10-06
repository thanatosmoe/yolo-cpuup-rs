//! Error types for `yolo-cpuup-rs`.

use thiserror::Error;

/// Errors that can occur while loading or running a YOLO model.
#[derive(Debug, Error)]
pub enum YoloError {
    /// Failed to configure the ONNX Runtime session builder.
    #[error("failed to configure ONNX Runtime session: {0}")]
    SessionConfig(String),

    /// Failed to load the ONNX model into a session.
    #[error("failed to load ONNX model: {0}")]
    SessionLoad(String),

    /// Failed to build the input tensor.
    #[error("failed to build input tensor: {0}")]
    InputTensor(String),

    /// The inference call itself failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// The output tensor could not be extracted.
    #[error("failed to extract output tensor: {0}")]
    OutputExtract(String),

    /// The output tensor did not have the expected shape.
    #[error(
        "unexpected model output shape {0:?} \
         (expected [1, 4+classes, anchors] or [1, anchors, 4+classes])"
    )]
    UnexpectedOutputShape(Vec<i64>),

    /// A class index was out of range for the label table.
    #[error("class index {index} is out of range for {len} labels")]
    ClassIndexOutOfRange { index: usize, len: usize },

    /// Image resizing failed.
    #[error("image resize failed: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),

    /// Building the source image buffer failed.
    #[error("image buffer error: {0}")]
    ImageBuffer(#[from] fast_image_resize::ImageBufferError),
}
