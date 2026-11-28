use sight_frame_core::normalize_rgb_to_chw;

#[test]
fn rgb_to_chw_respects_channels_and_normalization() {
    let rgb = [0u8, 127, 255, 255, 0, 64];
    let mut dst = [0.0f32; 6];
    assert!(normalize_rgb_to_chw(
        &rgb,
        2,
        1,
        [0.1, 0.2, 0.3],
        [1.0, 0.5, 0.25],
        &mut dst
    ));
    let expected = [
        0f32 / 255.0 - 0.1,
        255f32 / 255.0 - 0.1,
        (127f32 / 255.0 - 0.2) / 0.5,
        (0f32 / 255.0 - 0.2) / 0.5,
        (255f32 / 255.0 - 0.3) / 0.25,
        (64f32 / 255.0 - 0.3) / 0.25,
    ];
    for (actual, expected) in dst.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn invalid_layout_or_scaling_never_mutates_output() {
    let rgb = [1u8; 6];
    let mut dst = [88.0f32; 6];
    let good_mean = [0.0; 3];
    let good_std = [1.0; 3];
    assert!(!normalize_rgb_to_chw(
        &rgb, 3, 1, good_mean, good_std, &mut dst
    ));
    assert!(!normalize_rgb_to_chw(
        &rgb,
        2,
        1,
        good_mean,
        [1.0, 0.0, 1.0],
        &mut dst
    ));
    assert!(!normalize_rgb_to_chw(
        &rgb,
        2,
        1,
        [f32::NAN, 0.0, 0.0],
        good_std,
        &mut dst
    ));
    assert!(!normalize_rgb_to_chw(
        &rgb,
        u32::MAX,
        u32::MAX,
        good_mean,
        good_std,
        &mut dst
    ));
    assert_eq!(dst, [88.0; 6]);
}
