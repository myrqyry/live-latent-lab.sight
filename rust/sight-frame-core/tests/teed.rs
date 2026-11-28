use sight_frame_core::{rgb8_to_chw_scaled, teed_edges_to_gray};

#[test]
fn teed_input_is_unscaled_rgb_to_chw() {
    let rgb = [10, 20, 30, 40, 50, 60];
    let mut dst = [0f32; 6];
    assert!(rgb8_to_chw_scaled(
        &rgb, 2, 1, 1.0, [0.0; 3], [1.0; 3], &mut dst
    ));
    assert_eq!(dst, [10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);
    assert!(!rgb8_to_chw_scaled(
        &rgb,
        2,
        1,
        f32::NAN,
        [0.0; 3],
        [1.0; 3],
        &mut dst
    ));
    assert_eq!(dst, [10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);
}

#[test]
fn four_logit_maps_match_reference_sigmoid_and_safe_step() {
    let maps = [
        [0.0, -20.0, 8.0, 0.0],
        [0.0, -20.0, 8.0, 0.0],
        [0.0, -20.0, 8.0, 0.0],
        [0.0, -20.0, 8.0, 0.0],
    ];
    let mut out = [99u8; 4];
    assert!(teed_edges_to_gray(
        [&maps[0], &maps[1], &maps[2], &maps[3]],
        2,
        &mut out
    ));
    assert_eq!(out, [127, 0, 255, 127]);
    assert!(teed_edges_to_gray(
        [&maps[0], &maps[1], &maps[2], &maps[3]],
        0,
        &mut out
    ));
    assert_eq!(out, [127, 0, 254, 127]); // float -> uint8 truncation
    let before = out;
    assert!(!teed_edges_to_gray(
        [&maps[0], &maps[1], &maps[2], &maps[3]],
        11,
        &mut out
    ));
    assert_eq!(out, before);
    let bad = [f32::NAN, 0.0, 0.0, 0.0];
    assert!(!teed_edges_to_gray(
        [&bad, &maps[1], &maps[2], &maps[3]],
        2,
        &mut out
    ));
    assert_eq!(out, before);
}
