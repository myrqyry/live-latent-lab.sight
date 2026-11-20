#include "litert_model_catalog.hpp"

namespace lll {

const std::vector<LiteRtModelSpec> &litert_model_catalog()
{
    // Digests are the SHA-256 Xet etags reported by Hugging Face for each
    // pinned artifact. Custom models remain available through the manager.
    static const std::vector<LiteRtModelSpec> catalog{
        {"depth-anything-3-small", "Depth Anything 3 Small",
         "https://huggingface.co/litert-community/Depth-Anything-3-Small/resolve/main/da3_small_gpu_fp16.tflite",
         "e170369a72ba1bba7486a4d2de555639fccd0595a9bb5b5349f7733ed4aebd1f",
         "depth_anything_3_small.tflite"},
        {"6drepnet-head-pose", "6DRepNet Head Pose",
         "https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT/resolve/main/6drepnet.tflite",
         "44cc9679a18242fac31c9ad369465caad93a34c721eed3fe0ff003f712769dee",
         "6drepnet.tflite"},
        {"blaze-face-full-range", "BlazeFace Full Range",
         "https://huggingface.co/langminer/blaze-face-full-range/resolve/main/blaze_face_full_range.tflite",
         "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
         "blaze_face_full_range.tflite"},
        {"sam2-tiny-image-encoder", "SAM2.1 Hiera Tiny Image Encoder",
         "https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Image-Encoder/resolve/main/sam2_tiny_image_encoder_v2_fp16.tflite",
         "9f87693052c7c9bee2271f86d947420d7409ef0c2bec36c1885335afab90f6be",
         "sam2_tiny_image_encoder.tflite"},
        {"sam2-tiny-mask-decoder", "SAM2.1 Hiera Tiny Mask Decoder",
         "https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder/resolve/main/sam2_tiny_mask_decoder_v2_fp16.tflite",
         "80d668b19156c548f31a8c8eb3cc1da2127de36e24e96ef923467a1e54c99e76",
         "sam2_tiny_mask_decoder.tflite"},
        {"ormbg", "ORMBG Portrait Matting",
         "https://huggingface.co/litert-community/ormbg-LiteRT/resolve/main/ormbg.tflite",
         "f6e359cf35c55e7bf216eb3141f6d76828667a5e8e67a929f53e9f7f38fae030",
         "ormbg.tflite"},
        {"realesrgan-x4v3", "Real-ESRGAN x4v3",
         "https://huggingface.co/litert-community/real-esrgan-x4v3-litert/resolve/main/realesr_general_x4v3.tflite",
         "6d597b46d812ada42bdde641ec6e78c0ef0b78ff5a825fcfd74b9a1b3bcbe9a4",
         "realesr_general_x4v3.tflite"},
        {"nafnet-sidd", "NAFNet SIDD Restoration",
         "https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT/resolve/main/nafnet_sidd_width32_fp16.tflite",
         "f8fbaa422411683c53e802cf7cc7cf9be0a0de00886ad4af057232e26b172a0c",
         "nafnet_sidd.tflite"},
        {"fast-style-candy", "Fast Neural Style Candy",
         "https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT/resolve/main/style_candy_fp16.tflite",
         "b09d87e4f15ad6ba5f61c6599614bee8db097e09c751a4cf2f15abb5e5b543f9",
         "style_candy.tflite"},
        {"rtmw-wholebody", "RTMW Whole Body Pose",
         "https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT/resolve/main/rtmw_fp16.tflite",
         "66f79625d2054dfc5021fa22b38f0b76a1d5f7111cc790d5d9b65ee04a5985d6",
         "rtmw_wholebody.tflite"},
        {"rtmpose-pose17", "RTMPose S Pose",
         "https://huggingface.co/litert-community/RTMPose-s-LiteRT/resolve/main/rtmpose_s_fp16.tflite",
         "89f2d5c921bd824a39b37fc718550909cbb60d90e1cba48aa88b3ef71fab631d",
         "rtmpose_s.tflite"},
        {"rtmpose-face98", "RTMPose Face WFLW",
         "https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT/resolve/main/rtm_face_fp16.tflite",
         "652788790a2dac0ff117e5bc33c17eeb7048fd7ac0e6642e98b9fb60e65e24c7",
         "rtmpose_face.tflite"},
        {"rtmpose-hand21", "RTMPose Hand",
         "https://huggingface.co/litert-community/RTMPose-Hand-LiteRT/resolve/main/rtmhand_fp16.tflite",
         "8b48a941de7b6f5ac61b43bb68b18f253bc651e0bcbc89a6d11bb5a5f9288160",
         "rtmpose_hand.tflite"},
    };
    return catalog;
}

const LiteRtModelSpec *find_litert_model(const std::string &model_id)
{
    for (const auto &spec : litert_model_catalog()) {
        if (spec.id == model_id) {
            return &spec;
        }
    }
    return nullptr;
}

}  // namespace lll
