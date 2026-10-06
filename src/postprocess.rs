//! Output decoding and non-maximum suppression.
//!
//! The upstream project transposes the raw output with `reversed_axes()` and
//! then iterates it with an outer stride, which is cache-hostile. Here we read
//! the raw output as the contiguous slice ONNX Runtime gives us and scan it
//! attribute-major, so every read is sequential.

use arcstr::ArcStr;

use crate::{error::YoloError, BoundingBox, Detection};

/// Decode a raw YOLOv8 / YOLO11 detection output into final detections.
///
/// Supports both common output layouts:
/// * `[1, 4 + classes, anchors]` (Ultralytics default)
/// * `[1, anchors, 4 + classes]`
///
/// `scale` is `(raw_width / 640, raw_height / 640)`.
pub fn decode(
    data: &[f32],
    shape: &[i64],
    labels: &[ArcStr],
    probability_threshold: f32,
    iou_threshold: f32,
    scale: (f32, f32),
) -> Result<Vec<Detection>, YoloError> {
    if shape.len() != 3 || shape[0] != 1 {
        return Err(YoloError::UnexpectedOutputShape(shape.to_vec()));
    }

    let (dim1, dim2) = (shape[1] as usize, shape[2] as usize);
    let (attributes, anchors, attribute_major) = if dim1 < dim2 {
        // [1, 4+classes, anchors]
        (dim1, dim2, true)
    } else {
        // [1, anchors, 4+classes]
        (dim2, dim1, false)
    };

    if attributes < 5 {
        return Err(YoloError::UnexpectedOutputShape(shape.to_vec()));
    }
    let num_classes = attributes - 4;
    if data.len() < anchors * attributes {
        return Err(YoloError::UnexpectedOutputShape(shape.to_vec()));
    }

    let candidates = if attribute_major {
        decode_attribute_major(
            data,
            anchors,
            num_classes,
            labels,
            probability_threshold,
            scale,
        )?
    } else {
        decode_anchor_major(
            data,
            anchors,
            num_classes,
            labels,
            probability_threshold,
            scale,
        )?
    };

    Ok(non_maximum_suppression(candidates, iou_threshold))
}

/// `[1, 4 + classes, anchors]`: each attribute is a contiguous run of `anchors`.
fn decode_attribute_major(
    data: &[f32],
    anchors: usize,
    num_classes: usize,
    labels: &[ArcStr],
    probability_threshold: f32,
    scale: (f32, f32),
) -> Result<Vec<Detection>, YoloError> {
    // Class-major scan: sequential reads, one running best per anchor.
    let mut best = vec![f32::NEG_INFINITY; anchors];
    let mut best_class = vec![0u16; anchors];
    for class in 0..num_classes {
        let base = (4 + class) * anchors;
        let row = &data[base..base + anchors];
        for (anchor, &score) in row.iter().enumerate() {
            if score > best[anchor] {
                best[anchor] = score;
                best_class[anchor] = class as u16;
            }
        }
    }

    let cx = &data[0..anchors];
    let cy = &data[anchors..2 * anchors];
    let w = &data[2 * anchors..3 * anchors];
    let h = &data[3 * anchors..4 * anchors];
    let (scale_x, scale_y) = scale;

    let mut candidates = Vec::new();
    for anchor in 0..anchors {
        let confidence = best[anchor];
        if confidence < probability_threshold {
            continue;
        }
        let class_id = best_class[anchor] as usize;
        let label = labels
            .get(class_id)
            .ok_or(YoloError::ClassIndexOutOfRange {
                index: class_id,
                len: labels.len(),
            })?
            .clone();
        candidates.push(make_detection(
            cx[anchor] * scale_x,
            cy[anchor] * scale_y,
            w[anchor] * scale_x,
            h[anchor] * scale_y,
            class_id,
            label,
            confidence,
        ));
    }

    Ok(candidates)
}

/// `[1, anchors, 4 + classes]`: each anchor is a contiguous run of `attributes`.
fn decode_anchor_major(
    data: &[f32],
    anchors: usize,
    num_classes: usize,
    labels: &[ArcStr],
    probability_threshold: f32,
    scale: (f32, f32),
) -> Result<Vec<Detection>, YoloError> {
    let attributes = num_classes + 4;
    let (scale_x, scale_y) = scale;
    let mut candidates = Vec::new();

    for anchor in 0..anchors {
        let row = &data[anchor * attributes..anchor * attributes + attributes];

        let mut class_id = 0usize;
        let mut confidence = f32::NEG_INFINITY;
        for (class, &score) in row[4..].iter().enumerate() {
            if score > confidence {
                confidence = score;
                class_id = class;
            }
        }
        if confidence < probability_threshold {
            continue;
        }

        let label = labels
            .get(class_id)
            .ok_or(YoloError::ClassIndexOutOfRange {
                index: class_id,
                len: labels.len(),
            })?
            .clone();
        candidates.push(make_detection(
            row[0] * scale_x,
            row[1] * scale_y,
            row[2] * scale_x,
            row[3] * scale_y,
            class_id,
            label,
            confidence,
        ));
    }

    Ok(candidates)
}

#[inline]
fn make_detection(
    xc: f32,
    yc: f32,
    w: f32,
    h: f32,
    class_id: usize,
    label: ArcStr,
    confidence: f32,
) -> Detection {
    Detection {
        bbox: BoundingBox {
            x1: xc - w / 2.0,
            y1: yc - h / 2.0,
            x2: xc + w / 2.0,
            y2: yc + h / 2.0,
        },
        class_id,
        label,
        confidence,
    }
}

/// Greedy, class-agnostic non-maximum suppression (same behaviour as upstream).
fn non_maximum_suppression(mut candidates: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    if candidates.len() <= 1 {
        return candidates;
    }

    candidates.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut keep: Vec<Detection> = Vec::with_capacity(candidates.len());
    'next: for candidate in candidates {
        for selected in &keep {
            if intersection_over_union(&selected.bbox, &candidate.bbox) >= iou_threshold {
                continue 'next;
            }
        }
        keep.push(candidate);
    }
    keep
}

#[inline]
fn intersection_over_union(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_w = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let inter_h = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let intersection = inter_w * inter_h;
    let union = a.area() + b.area() - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcstr::ArcStr;

    fn labels() -> Vec<ArcStr> {
        vec![arcstr::literal!("a"), arcstr::literal!("b")]
    }

    #[test]
    fn decode_attribute_major_layout() {
        // shape [1, attributes=6, anchors=10] -> attribute-major
        // attributes = (cx, cy, w, h, cls0, cls1), 10 anchors
        let mut data = vec![0.0f32; 6 * 10];
        data[0] = 320.0; // anchor 0 cx
        data[10] = 320.0; // anchor 0 cy
        data[2 * 10] = 40.0; // anchor 0 w
        data[3 * 10] = 20.0; // anchor 0 h
        data[4 * 10] = 0.95; // anchor 0 cls 0
        data[5 * 10] = 0.10; // anchor 0 cls 1

        let detections = decode(&data, &[1, 6, 10], &labels(), 0.5, 0.7, (2.0, 2.0)).unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 0);
        assert!((detections[0].confidence - 0.95).abs() < 1e-6);
        // scale of 2.0: centre (640, 640), box 80x40
        assert!((detections[0].bbox.x1 - 600.0).abs() < 1e-3);
        assert!((detections[0].bbox.y1 - 620.0).abs() < 1e-3);
        assert!((detections[0].bbox.x2 - 680.0).abs() < 1e-3);
        assert!((detections[0].bbox.y2 - 660.0).abs() < 1e-3);
    }

    #[test]
    fn nms_suppresses_overlapping_boxes() {
        let mk = |x: f32, conf: f32| Detection {
            bbox: BoundingBox {
                x1: x,
                y1: 0.0,
                x2: x + 10.0,
                y2: 10.0,
            },
            class_id: 0,
            label: arcstr::literal!("a"),
            confidence: conf,
        };
        let out = non_maximum_suppression(vec![mk(0.0, 0.9), mk(1.0, 0.8), mk(100.0, 0.7)], 0.5);
        assert_eq!(out.len(), 2);
    }
}
