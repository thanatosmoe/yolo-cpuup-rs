//! Public-API tests for output decoding (no ONNX model required).

use arcstr::ArcStr;
use yolo_cpuup::postprocess::decode;

fn labels() -> Vec<ArcStr> {
    vec![arcstr::literal!("cat"), arcstr::literal!("dog")]
}

#[test]
fn decodes_attribute_major_layout() {
    // shape [1, 6, 10]: attributes = (cx, cy, w, h, c0, c1), 10 anchors
    let mut data = vec![0.0f32; 6 * 10];
    data[0] = 320.0; // anchor 0 cx
    data[10] = 320.0; // anchor 0 cy
    data[2 * 10] = 40.0; // anchor 0 w
    data[3 * 10] = 20.0; // anchor 0 h
    data[4 * 10] = 0.95; // anchor 0 class 0
    data[5 * 10] = 0.10; // anchor 0 class 1

    let detections = decode(&data, &[1, 6, 10], &labels(), 0.5, 0.7, (2.0, 2.0)).unwrap();

    assert_eq!(detections.len(), 1);
    let d = &detections[0];
    assert_eq!(d.class_id, 0);
    assert_eq!(d.label.as_str(), "cat");
    assert!((d.confidence - 0.95).abs() < 1e-6);
    // scale of 2.0: centre (640, 640), box 80x40
    assert!((d.bbox.x1 - 600.0).abs() < 1e-3);
    assert!((d.bbox.y1 - 620.0).abs() < 1e-3);
    assert!((d.bbox.x2 - 680.0).abs() < 1e-3);
    assert!((d.bbox.y2 - 660.0).abs() < 1e-3);
}

#[test]
fn decodes_anchor_major_layout() {
    // shape [1, 10, 6]: 10 anchors, each a contiguous run of (cx, cy, w, h, c0, c1)
    let mut data = vec![0.0f32; 10 * 6];
    data[0] = 100.0;
    data[1] = 100.0;
    data[2] = 10.0;
    data[3] = 10.0;
    data[4] = 0.1;
    data[5] = 0.9; // class 1 wins

    let detections = decode(&data, &[1, 10, 6], &labels(), 0.5, 0.7, (1.0, 1.0)).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].class_id, 1);
    assert_eq!(detections[0].label.as_str(), "dog");
}

#[test]
fn rejects_unexpected_shape() {
    let data = vec![0.0f32; 10];
    assert!(decode(&data, &[1, 5], &labels(), 0.5, 0.7, (1.0, 1.0)).is_err());
}
