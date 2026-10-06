//! Fast CPU preprocessing: resize + normalize an image into a YOLO NCHW tensor.
//!
//! The upstream implementation resizes with the `image` crate and then walks the
//! resized image pixel-by-pixel, converting every pixel to `f32` with a floating
//! point division and writing into a strided `ndarray`. That is the single
//! largest CPU cost outside of the model itself.
//!
//! This module:
//! 1. resizes with [`fast_image_resize`] (SIMD, optionally multi-threaded),
//! 2. normalizes through a 256-entry lookup table (identical maths, no divides),
//! 3. writes a flat, contiguous planar `NCHW` buffer that is handed to ONNX
//!    Runtime without an intermediate `ndarray`.

use fast_image_resize::{images::Image, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::DynamicImage;

use crate::error::YoloError;

/// The spatial input size expected by YOLOv8 / YOLO11 detection models.
pub const INPUT_SIZE: u32 = 640;

/// Number of elements in one channel plane.
pub const PLANE: usize = (INPUT_SIZE * INPUT_SIZE) as usize;

/// `u8 -> f32` normalization table, identical to `v as f32 / 255.0`.
pub static LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
};

/// A preprocessed image ready to be fed to the model.
///
/// The `data` buffer is a flat, contiguous `NCHW` tensor of shape
/// `[1, 3, 640, 640]`, so it can be borrowed directly by ONNX Runtime.
#[derive(Debug, Clone)]
pub struct PreprocessedInput {
    /// Flat `[1, 3, 640, 640]` tensor data (R plane, then G, then B).
    pub data: Vec<f32>,
    /// Width of the original (un-resized) image.
    pub raw_width: u32,
    /// Height of the original (un-resized) image.
    pub raw_height: u32,
}

impl PreprocessedInput {
    /// Allocate a zeroed input buffer.
    pub fn new() -> Self {
        Self {
            data: vec![0.0; 3 * PLANE],
            raw_width: 0,
            raw_height: 0,
        }
    }

    /// The tensor shape as ONNX Runtime expects it.
    #[inline]
    pub fn shape(&self) -> [usize; 4] {
        [1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize]
    }

    /// Scale factors used to map box coordinates from the 640x640 tensor space
    /// back to the original image space: `(raw_width / 640, raw_height / 640)`.
    #[inline]
    pub fn scale(&self) -> (f32, f32) {
        (
            self.raw_width as f32 / INPUT_SIZE as f32,
            self.raw_height as f32 / INPUT_SIZE as f32,
        )
    }
}

impl Default for PreprocessedInput {
    fn default() -> Self {
        Self::new()
    }
}

/// Reusable preprocessing state so buffers are not reallocated on every frame.
///
/// Holds the [`Resizer`] (which caches its internal convolution buffers) and the
/// destination image. Create one per worker / per model and reuse it.
#[derive(Debug)]
pub struct Preprocessor {
    resizer: Resizer,
    dst: Image<'static>,
    options: ResizeOptions,
}

impl Preprocessor {
    /// Create a new preprocessor.
    pub fn new() -> Self {
        Self {
            resizer: Resizer::new(),
            dst: Image::new(INPUT_SIZE, INPUT_SIZE, PixelType::U8x3),
            options: ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
        }
    }

    /// Resize `image` to 640x640 (Catmull-Rom, matching the reference YOLO
    /// pipeline) and normalize it into `out` as a contiguous NCHW `f32` tensor.
    pub fn run_into(
        &mut self,
        image: &DynamicImage,
        out: &mut PreprocessedInput,
    ) -> Result<(), YoloError> {
        let (width, height) = (image.width(), image.height());

        // `to_rgb8()` drops alpha / expands grayscale exactly like the reference
        // pipeline, but works on a contiguous RGB byte buffer.
        let rgb = image.to_rgb8();
        let src = Image::from_vec_u8(width, height, rgb.into_raw(), PixelType::U8x3)?;

        self.resizer.resize(&src, &mut self.dst, &self.options)?;
        normalize_into(self.dst.buffer(), &mut out.data);

        out.raw_width = width;
        out.raw_height = height;
        Ok(())
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot convenience wrapper around [`Preprocessor`].
pub fn preprocess(image: &DynamicImage) -> Result<PreprocessedInput, YoloError> {
    let mut out = PreprocessedInput::new();
    Preprocessor::new().run_into(image, &mut out)?;
    Ok(out)
}

/// Reference preprocessing, byte-for-byte compatible with the upstream project:
/// `DynamicImage::resize_exact(640, 640, CatmullRom)` followed by per-pixel
/// normalization. Kept for correctness checks and benchmarking.
pub fn preprocess_reference_into(
    image: &DynamicImage,
    out: &mut PreprocessedInput,
) -> Result<(), YoloError> {
    let resized =
        image.resize_exact(INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    normalize_into(rgb.as_raw(), &mut out.data);
    out.raw_width = image.width();
    out.raw_height = image.height();
    Ok(())
}

/// Interleaved RGB8 -> planar NCHW `f32` through [`LUT`].
#[inline]
fn normalize_into(rgb: &[u8], planar: &mut [f32]) {
    debug_assert_eq!(rgb.len(), 3 * PLANE);
    debug_assert_eq!(planar.len(), 3 * PLANE);

    let (r_plane, rest) = planar.split_at_mut(PLANE);
    let (g_plane, b_plane) = rest.split_at_mut(PLANE);

    for (i, px) in rgb.chunks_exact(3).enumerate() {
        r_plane[i] = LUT[px[0] as usize];
        g_plane[i] = LUT[px[1] as usize];
        b_plane[i] = LUT[px[2] as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_division() {
        for v in 0u16..=255 {
            assert_eq!(LUT[v as usize], v as f32 / 255.0);
        }
    }

    #[test]
    fn normalize_is_planar_nchw() {
        let mut rgb = vec![0u8; 3 * PLANE];
        // First pixel: R=255, G=128, B=0.
        rgb[0] = 255;
        rgb[1] = 128;
        rgb[2] = 0;
        let mut planar = vec![0.0f32; 3 * PLANE];
        normalize_into(&rgb, &mut planar);
        assert_eq!(planar[0], 1.0);
        assert_eq!(planar[PLANE], 128.0 / 255.0);
        assert_eq!(planar[2 * PLANE], 0.0);
    }
}
