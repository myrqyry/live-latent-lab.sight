//! Dependency-free, allocation-free CPU frame and model tensor preparation.
//! External buffer ownership stays with Sight; the C ABI never retains an OBS pointer.

/// Normalize packed RGB8 pixels into planar NCHW-compatible float32 channels.
/// All dimensions refer to a single frame with no row padding.
/// On failure the destination remains unchanged.
pub fn normalize_rgb_to_chw(
    rgb: &[u8],
    width: u32,
    height: u32,
    mean: [f32; 3],
    std: [f32; 3],
    dst: &mut [f32],
) -> bool {
    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return false;
    };
    let Some(expected) = pixels.checked_mul(3) else {
        return false;
    };
    if pixels == 0
        || rgb.len() != expected
        || dst.len() != expected
        || mean.iter().any(|v| !v.is_finite())
        || std.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return false;
    }
    for pixel in 0..pixels {
        for channel in 0..3 {
            dst[channel * pixels + pixel] =
                (f32::from(rgb[pixel * 3 + channel]) / 255.0 - mean[channel]) / std[channel];
        }
    }
    true
}

/// C ABI: 1 on success; 0 without output mutation for invalid inputs.
///
/// # Safety
///
/// RGB and output must be valid disjoint allocations for the stated sizes.
/// Mean and std must each point to three readable floats. No pointers retained.
#[no_mangle]
pub unsafe extern "C" fn sight_normalize_rgb_to_chw_f32_v1(
    rgb: *const u8,
    rgb_len: usize,
    width: u32,
    height: u32,
    mean: *const f32,
    std: *const f32,
    output: *mut f32,
    output_len: usize,
) -> i32 {
    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return 0;
    };
    let Some(required) = pixels.checked_mul(3) else {
        return 0;
    };
    if pixels == 0
        || rgb.is_null()
        || output.is_null()
        || mean.is_null()
        || std.is_null()
        || rgb_len != required
        || output_len != required
        || (output as usize) % std::mem::align_of::<f32>() != 0
        || (mean as usize) % std::mem::align_of::<f32>() != 0
        || (std as usize) % std::mem::align_of::<f32>() != 0
    {
        return 0;
    }
    let Some(rgb_end) = (rgb as usize).checked_add(rgb_len) else {
        return 0;
    };
    let Some(output_bytes) = output_len.checked_mul(std::mem::size_of::<f32>()) else {
        return 0;
    };
    let Some(output_end) = (output as usize).checked_add(output_bytes) else {
        return 0;
    };
    if (rgb as usize) < output_end && (output as usize) < rgb_end {
        return 0;
    }
    let means = unsafe { [*mean, *mean.add(1), *mean.add(2)] };
    let stds = unsafe { [*std, *std.add(1), *std.add(2)] };
    if means.iter().any(|v| !v.is_finite()) || stds.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return 0;
    }
    let input = unsafe { std::slice::from_raw_parts(rgb, rgb_len) };
    let destination = unsafe { std::slice::from_raw_parts_mut(output, output_len) };
    i32::from(normalize_rgb_to_chw(
        input,
        width,
        height,
        means,
        stds,
        destination,
    ))
}

/// RGB8 into planar float32 CHW using an explicit model input scale.
/// TEED requires raw 0..255 channels (scale=1), unlike Depth Anything.
/// No output mutation on an invalid contract.
pub fn rgb8_to_chw_scaled(
    rgb: &[u8],
    width: u32,
    height: u32,
    scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
    dst: &mut [f32],
) -> bool {
    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return false;
    };
    let Some(required) = pixels.checked_mul(3) else {
        return false;
    };
    if pixels == 0
        || required != rgb.len()
        || required != dst.len()
        || !scale.is_finite()
        || scale <= 0.0
        || mean.iter().any(|v| !v.is_finite())
        || std.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return false;
    }
    for p in 0..pixels {
        for c in 0..3 {
            dst[c * pixels + p] = (f32::from(rgb[p * 3 + c]) * scale - mean[c]) / std[c];
        }
    }
    true
}

/// Version 2 model-input contract adds an explicit scale.
///
/// # Safety
///
/// Pointer arguments must denote valid accessible memory for all stated
/// elements. The output buffer must not overlap the RGB buffer. No pointers
/// are retained, and the destination is not modified on rejected arguments.
#[no_mangle]
pub unsafe extern "C" fn sight_rgb8_to_chw_f32_v2(
    rgb: *const u8,
    rgb_len: usize,
    width: u32,
    height: u32,
    scale: f32,
    mean: *const f32,
    std: *const f32,
    output: *mut f32,
    output_len: usize,
) -> i32 {
    let Some(pixels) = (width as usize).checked_mul(height as usize) else {
        return 0;
    };
    let Some(required) = pixels.checked_mul(3) else {
        return 0;
    };
    if pixels == 0
        || rgb.is_null()
        || output.is_null()
        || mean.is_null()
        || std.is_null()
        || rgb_len != required
        || output_len != required
        || !scale.is_finite()
        || scale <= 0.0
        || (mean as usize) % std::mem::align_of::<f32>() != 0
        || (std as usize) % std::mem::align_of::<f32>() != 0
        || (output as usize) % std::mem::align_of::<f32>() != 0
    {
        return 0;
    }
    let Some(rgb_end) = (rgb as usize).checked_add(rgb_len) else {
        return 0;
    };
    let Some(output_bytes) = output_len.checked_mul(std::mem::size_of::<f32>()) else {
        return 0;
    };
    let Some(output_end) = (output as usize).checked_add(output_bytes) else {
        return 0;
    };
    if (rgb as usize) < output_end && (output as usize) < rgb_end {
        return 0;
    }
    let means = unsafe { [*mean, *mean.add(1), *mean.add(2)] };
    let stds = unsafe { [*std, *std.add(1), *std.add(2)] };
    if !scale.is_finite()
        || means.iter().any(|v| !v.is_finite())
        || stds.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return 0;
    }
    let rgb = unsafe { std::slice::from_raw_parts(rgb, rgb_len) };
    let out = unsafe { std::slice::from_raw_parts_mut(output, output_len) };
    i32::from(rgb8_to_chw_scaled(
        rgb, width, height, scale, means, stds, out,
    ))
}

/// TEED's exact four-logit-map reduction:
/// mean logits (float32) -> sigmoid (float64) -> optional safe_step ->
/// saturating float-to-uint8 conversion. No output change on bad data.
pub fn teed_edges_to_gray(maps: [&[f32]; 4], safe_steps: u32, output: &mut [u8]) -> bool {
    if output.is_empty()
        || safe_steps > 10
        || maps.iter().any(|m| m.len() != output.len())
        || maps.iter().any(|m| m.iter().any(|v| !v.is_finite()))
    {
        return false;
    }
    for (p, dst) in output.iter_mut().enumerate() {
        let logits = ((maps[0][p] + maps[1][p]) + maps[2][p] + maps[3][p]) * 0.25;
        let sigmoid = 1.0_f64 / (1.0_f64 + (-(logits as f64)).exp());
        let quantized = if safe_steps == 0 {
            sigmoid
        } else {
            let level = ((sigmoid as f32) * (safe_steps + 1) as f32).trunc() as i32;
            (level as f32 / safe_steps as f32) as f64
        };
        *dst = (quantized * 255.0).clamp(0.0, 255.0) as u8;
    }
    true
}

/// Four TEED map pointers with identical lengths and a caller-owned u8 result.
///
/// # Safety
///
/// Every pointer must be valid for its declared byte span. The output must
/// not overlap any input. No allocation, pointer retention, or OBS access.
#[no_mangle]
pub unsafe extern "C" fn sight_teed_edges_to_gray_v1(
    edge0: *const f32,
    edge1: *const f32,
    edge2: *const f32,
    edge3: *const f32,
    pixels: usize,
    safe_steps: u32,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    if pixels == 0 || pixels != out_len || out.is_null() || safe_steps > 10 {
        return 0;
    }
    let pointers = [edge0, edge1, edge2, edge3];
    let Some(bytes) = pixels.checked_mul(std::mem::size_of::<f32>()) else {
        return 0;
    };
    let Some(out_end) = (out as usize).checked_add(out_len) else {
        return 0;
    };
    for ptr in pointers {
        if ptr.is_null() || (ptr as usize) % std::mem::align_of::<f32>() != 0 {
            return 0;
        }
        let Some(end) = (ptr as usize).checked_add(bytes) else {
            return 0;
        };
        if (ptr as usize) < out_end && (out as usize) < end {
            return 0;
        }
    }
    let arrays = [
        unsafe { std::slice::from_raw_parts(edge0, pixels) },
        unsafe { std::slice::from_raw_parts(edge1, pixels) },
        unsafe { std::slice::from_raw_parts(edge2, pixels) },
        unsafe { std::slice::from_raw_parts(edge3, pixels) },
    ];
    let output = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
    i32::from(teed_edges_to_gray(arrays, safe_steps, output))
}

// PiDiNet float32 probability map -> 8-bit edge image.
// Reference: optional safe_step(x,2), clipping then truncation to uint8.
// Validate all values before changing the output.
pub fn pidinet_probabilities_to_gray(probabilities: &[f32], safe: bool, output: &mut [u8]) -> bool {
    if probabilities.is_empty()
        || probabilities.len() != output.len()
        || probabilities.iter().any(|x| !x.is_finite())
    {
        return false;
    }
    for (src, dst) in probabilities.iter().zip(output.iter_mut()) {
        let quantized = if safe {
            ((*src * 3.0).trunc() as i32) as f32 / 2.0
        } else {
            *src
        };
        *dst = (quantized * 255.0).clamp(0.0, 255.0) as u8;
    }
    true
}

/// # Safety
///
/// Inputs must be valid for declared lengths and non-overlapping. Alignment
/// is checked but cannot establish allocation validity. No pointers retained.
#[no_mangle]
pub unsafe extern "C" fn sight_pidinet_probs_to_gray_v1(
    probabilities: *const f32,
    pixels: usize,
    safe: i32,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    if pixels == 0
        || pixels != output_len
        || probabilities.is_null()
        || output.is_null()
        || (probabilities as usize) % std::mem::align_of::<f32>() != 0
        || (safe != 0 && safe != 1)
    {
        return 0;
    }
    let Some(byte_len) = pixels.checked_mul(std::mem::size_of::<f32>()) else {
        return 0;
    };
    let Some(input_end) = (probabilities as usize).checked_add(byte_len) else {
        return 0;
    };
    let Some(output_end) = (output as usize).checked_add(output_len) else {
        return 0;
    };
    if (probabilities as usize) < output_end && (output as usize) < input_end {
        return 0;
    }
    let input = unsafe { std::slice::from_raw_parts(probabilities, pixels) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, output_len) };
    i32::from(pidinet_probabilities_to_gray(input, safe != 0, dst))
}
