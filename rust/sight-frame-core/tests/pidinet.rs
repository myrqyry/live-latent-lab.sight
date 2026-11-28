use sight_frame_core::pidinet_probabilities_to_gray;

#[test]
fn prob_map_matches_safe_step_and_raw_reference() {
    let probabilities = [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0];
    let mut out = [88u8; 7];
    assert!(pidinet_probabilities_to_gray(
        &probabilities,
        false,
        &mut out
    ));
    assert_eq!(out, [0, 25, 63, 127, 191, 252, 255]);
    assert!(pidinet_probabilities_to_gray(
        &probabilities,
        true,
        &mut out
    ));
    assert_eq!(out, [0, 0, 0, 127, 255, 255, 255]);
}

#[test]
fn invalid_probabilities_never_modify_output() {
    let mut out = [88u8; 4];
    assert!(!pidinet_probabilities_to_gray(
        &[0.0, f32::NAN, 0.25, 1.0],
        true,
        &mut out
    ));
    assert!(!pidinet_probabilities_to_gray(
        &[0.0, 0.25, 1.0],
        true,
        &mut out
    ));
    assert_eq!(out, [88u8; 4]);
}
