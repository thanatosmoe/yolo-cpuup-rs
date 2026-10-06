//! Detection result types shared across the crate.

use arcstr::ArcStr;

/// An axis-aligned bounding box in the coordinate space of the original image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    /// Width of the box.
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Height of the box.
    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Area of the box.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// A single detected object.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    /// Numeric class id (index into the model's label table).
    pub class_id: usize,
    /// Human readable class label.
    pub label: ArcStr,
    /// Class confidence in `[0, 1]`.
    pub confidence: f32,
}
