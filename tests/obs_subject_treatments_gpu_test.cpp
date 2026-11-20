#include <obs.h>
#include <graphics/graphics.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "../src/core/subject_matte_history.hpp"

// Tests the REAL OBS shader against a deterministic foreground matte.
// The purpose is compositing correctness, not a claim that an ONNX model ran.
namespace {
constexpr unsigned W = 128, H = 96;
using Frame = std::vector<uint8_t>;
Frame render(gs_effect_t *fx, gs_texture_t *source, gs_texture_t *mask,
             int mode, float strength, bool refine = false,
             unsigned mask_w = W, unsigned mask_h = H,
             float gamma = 1.f, float low = 0.f, float high = 1.f, int view = 0)
{
    auto *target = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    auto *stage = gs_stagesurface_create(W, H, GS_RGBA);
    Frame image;
    gs_set_cull_mode(GS_NEITHER);
    if (target && stage && gs_texrender_begin(target, W, H)) {
        gs_projection_push();
        gs_matrix_push();
        gs_matrix_identity();
        gs_ortho(0, float(W), 0, float(H), -100, 100);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx, "original_frame"), source);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx, "segmentation_mask"), mask);
        gs_effect_set_int(gs_effect_get_param_by_name(fx, "treatment_mode"), mode);
        gs_effect_set_float(gs_effect_get_param_by_name(fx, "treatment_strength"), strength);
        struct vec2 texel {};
        vec2_set(&texel, 1.f / mask_w, 1.f / mask_h);
        if (auto *param = gs_effect_get_param_by_name(fx, "refine_edges"))
            gs_effect_set_bool(param, refine);
        gs_effect_set_vec2(gs_effect_get_param_by_name(fx, "mask_texel"), &texel);
        for (auto [name, value] : {std::pair{"matte_gamma",gamma}, {"matte_low",low}, {"matte_high",high}})
            if (auto *param=gs_effect_get_param_by_name(fx,name)) gs_effect_set_float(param,value);
        if (auto *param=gs_effect_get_param_by_name(fx,"matte_view")) gs_effect_set_int(param,view);
        while (gs_effect_loop(fx, "Draw"))
            gs_draw_sprite(source, 0, W, H);
        gs_matrix_pop();
        gs_projection_pop();
        gs_texrender_end(target);
        gs_stage_texture(stage, gs_texrender_get_texture(target));
        uint8_t *data = nullptr;
        uint32_t pitch = 0;
        if (gs_stagesurface_map(stage, &data, &pitch)) {
            image.resize(size_t(W) * H * 4);
            for (unsigned y = 0; y < H; ++y)
                std::memcpy(image.data() + size_t(y) * W * 4,
                            data + size_t(y) * pitch, W * 4);
            gs_stagesurface_unmap(stage);
        }
    }
    if (stage) gs_stagesurface_destroy(stage);
    if (target) gs_texrender_destroy(target);
    return image;
}
int channel(const Frame &frame, unsigned x, unsigned y, unsigned c)
{
    return frame[(size_t(y) * W + x) * 4 + c];
}
bool close(int a, int b, int tolerance=3)
{
    return std::abs(a-b) <= tolerance;
}
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    if (!obs_startup("en-US", nullptr, nullptr)) return 1;
    obs_video_info settings {};
    settings.graphics_module = "libobs-opengl";
    settings.fps_num = 30; settings.fps_den = 1;
    settings.base_width = W; settings.base_height = H;
    settings.output_width = W; settings.output_height = H;
    settings.output_format = VIDEO_FORMAT_RGBA;
    settings.scale_type = OBS_SCALE_DISABLE;
    if (obs_reset_video(&settings) != OBS_VIDEO_SUCCESS) {
        obs_shutdown(); return 1;
    }
    obs_enter_graphics();
    char *errors = nullptr;
    auto *effect = gs_effect_create_from_file(argv[1], &errors);
    if (errors) {
        std::cerr << "Shader compiler: " << errors << "\n";
        bfree(errors);
    }
    bool okay = false;
    if (effect) {
        Frame source(size_t(W) * H * 4);
        std::vector<uint8_t> mask(size_t(W) * H, 0);
        for (unsigned y = 0; y < H; ++y)
            for (unsigned x = 0; x < W; ++x) {
                auto i = (size_t(y) * W + x) * 4;
                source[i] = 90; source[i+1] = 110;
                source[i+2] = 140; source[i+3] = 255;
                if (x >= 42 && x < 86 && y >= 20 && y < 76)
                    mask[size_t(y) * W + x] = 255;
            }
        const uint8_t *p = source.data();
        const uint8_t *m = mask.data();
        auto *input = gs_texture_create(W, H, GS_RGBA, 1, &p, 0);
        auto *matte = gs_texture_create(W, H, GS_R8, 1, &m, 0);
        if (input && matte) {
            const auto cutout = render(effect, input, matte, 0, .7f);
            const auto spotlight = render(effect, input, matte, 1, 1.f);
            const auto warm = render(effect, input, matte, 2, 1.f);
            const auto glow = render(effect, input, matte, 3, 1.f);
            const auto spotlight_zero = render(effect, input, matte, 1, 0.f);
            const auto warm_zero = render(effect, input, matte, 2, 0.f);
            const auto glow_zero = render(effect, input, matte, 3, 0.f);
            const bool sizes = cutout.size()==source.size()
                            && spotlight.size()==source.size()
                            && warm.size()==source.size()
                            && glow.size()==source.size();
            bool cut = false, spot = false, tinted = false, halo = false;
            bool zero_strength = false;
            if (sizes) {
                cut = channel(cutout, 8, 8, 3)==0
                      && channel(cutout, 64, 45, 3)>=248
                      && close(channel(cutout,64,45,0),90);
                spot = channel(spotlight, 8, 8, 0)<75
                       && close(channel(spotlight,64,45,0),90)
                       && channel(spotlight,8,8,3)==255;
                tinted = channel(warm,64,45,0)>95
                         && close(channel(warm,8,8,0),90)
                         && channel(warm,8,8,3)==255;
                halo = channel(glow,41,44,0)>90
                       && close(channel(glow,8,8,0),90)
                       && channel(glow,8,8,3)==255;
                // Each non-cutout treatment must return the original RGB and
                // alpha when its user-facing strength slider is zero.
                zero_strength = spotlight_zero.size()==source.size()
                    && warm_zero.size()==source.size()
                    && glow_zero.size()==source.size();
                if (zero_strength) {
                    for (const auto *result : {&spotlight_zero, &warm_zero, &glow_zero}) {
                        for (size_t i=0; i<source.size(); ++i) {
                            if (!close((*result)[i], source[i], 3)) {
                                zero_strength = false;
                                break;
                            }
                        }
                    }
                }
            }
            // Coarse mask aligns with a sharp image edge at x=64. Ordinary
            // bilinear sampling bleeds across that boundary.
            Frame edge_source = source;
            for (unsigned y = 0; y < H; ++y)
                for (unsigned x = 0; x < W; ++x)
                    for (unsigned c = 0; c < 3; ++c)
                        edge_source[(size_t(y)*W+x)*4+c] = x < W/2 ? 20 : 230;
            constexpr unsigned MW = 8, MH = 6;
            std::vector<uint8_t> coarse(MW * MH);
            for (unsigned y = 0; y < MH; ++y)
                for (unsigned x = 0; x < MW; ++x)
                    coarse[y*MW+x] = x < MW/2 ? 0 : 255;
            const uint8_t *ep = edge_source.data(), *cp = coarse.data();
            auto *edge_input = gs_texture_create(W,H,GS_RGBA,1,&ep,0);
            auto *coarse_mask = gs_texture_create(MW,MH,GS_R8,1,&cp,0);
            bool refined = false, flat_guide = false;
            if (edge_input && coarse_mask) {
                const auto linear = render(effect,edge_input,coarse_mask,0,1.f,false,MW,MH);
                const auto guided = render(effect,edge_input,coarse_mask,0,1.f,true,MW,MH);
                const auto flat_linear = render(effect,input,coarse_mask,0,1.f,false,MW,MH);
                const auto flat_refined = render(effect,input,coarse_mask,0,1.f,true,MW,MH);
                if (linear.size()==source.size() && guided.size()==source.size()) {
                    const int l = channel(linear,63,45,3), r = channel(linear,64,45,3);
                    const int gl = channel(guided,63,45,3), gr = channel(guided,64,45,3);
                    refined = l > 70 && r < 185 && gl < 15 && gr > 240
                        && channel(guided,8,45,3)==0 && channel(guided,120,45,3)==255;
                    std::cout << "EDGE_ALPHA_LINEAR=" << l << "," << r
                              << " REFINED=" << gl << "," << gr << "\n";
                }
                flat_guide = flat_linear.size()==source.size() && flat_refined.size()==source.size();
                if (flat_guide)
                    for (size_t i=0; i<source.size(); ++i)
                        if (!close(flat_linear[i],flat_refined[i],3)) {flat_guide=false;break;}
            }
            if (coarse_mask) gs_texture_destroy(coarse_mask);
            if (edge_input) gs_texture_destroy(edge_input);
            std::cout << "EDGE_REFINEMENT=" << refined << " FLAT_GUIDE=" << flat_guide << "\n";
            std::vector<uint8_t> soft(W*H);
            for (unsigned y=0;y<H;++y)
                for (unsigned x=0;x<W;++x) soft[y*W+x]=x<32?26:x<64?128:x<96?230:255;
            const uint8_t *sp=soft.data();
            auto *soft_mask=gs_texture_create(W,H,GS_R8,1,&sp,0);
            bool calibration=false, inspector=false;
            if (soft_mask) {
                auto base=render(effect,input,soft_mask,0,1.f);
                auto cal=render(effect,input,soft_mask,0,1.f,false,W,H,1.f,.2f,.8f);
                auto gamma=render(effect,input,soft_mask,0,1.f,false,W,H,2.f);
                auto final=render(effect,input,soft_mask,0,1.f,false,W,H,1.f,.2f,.8f,1);
                auto raw=render(effect,input,soft_mask,0,1.f,false,W,H,2.f,.2f,.8f,2);
                auto stable=render(effect,input,soft_mask,0,1.f,false,W,H,2.f,.2f,.8f,3);
                auto orig=render(effect,input,soft_mask,0,1.f,false,W,H,2.f,.2f,.8f,4);
                if (base.size()==source.size() && cal.size()==source.size() && gamma.size()==source.size())
                    calibration=close(channel(base,16,40,3),26) && channel(cal,16,40,3)==0
                    && channel(cal,80,40,3)==255 && close(channel(cal,48,40,3),128)
                    && close(channel(gamma,48,40,3),64) && channel(gamma,112,40,3)==255;
                if (final.size()==source.size() && raw.size()==source.size() && stable.size()==source.size() && orig.size()==source.size())
                    inspector=channel(final,80,40,0)==255 && channel(final,16,40,0)==0
                    && channel(final,16,40,3)==255 && close(channel(raw,48,40,0),128)
                    && close(channel(stable,48,40,0),128) && channel(raw,48,40,3)==255
                    && channel(stable,48,40,3)==255 && orig==source;
                gs_texture_destroy(soft_mask);
            }
            sight::SubjectMatteHistory history;
            std::vector<uint8_t> rgb(6,80), alpha{100,100};
            history.apply(alpha,rgb,2,1,.5f);
            bool temporal=alpha==std::vector<uint8_t>{100,100};
            alpha={140,140}; rgb[3]=200;
            history.apply(alpha,rgb,2,1,.5f);
            temporal=temporal && alpha==std::vector<uint8_t>{120,140};
            alpha={255,0}; history.apply(alpha,rgb,2,1,.5f);
            temporal=temporal && alpha==std::vector<uint8_t>{255,0};
            alpha={235,20}; history.apply(alpha,rgb,2,1,0.f);
            temporal=temporal && alpha==std::vector<uint8_t>{235,20};
            alpha={215,40}; history.apply(alpha,rgb,2,1,.5f);
            temporal=temporal && alpha==std::vector<uint8_t>{215,40};
            alpha={195,60}; history.apply(alpha,rgb,1,2,.5f);
            temporal=temporal && alpha==std::vector<uint8_t>{195,60};
            history.reset(); alpha={175,80}; history.apply(alpha,rgb,1,2,.5f);
            temporal=temporal && alpha==std::vector<uint8_t>{175,80};
            std::cout<<"CALIBRATION="<<calibration<<" INSPECTOR="<<inspector<<" TEMPORAL="<<temporal<<"\n";
            okay = cut && spot && tinted && halo && zero_strength && refined && flat_guide
                && calibration && inspector && temporal;
            std::cout << "CUTOUT=" << cut << " SPOTLIGHT=" << spot
                      << " WARM_SUBJECT=" << tinted << " EDGE_GLOW=" << halo
                      << " ZERO_STRENGTH_PASSTHROUGH=" << zero_strength << "\n";
        }
        if (matte) gs_texture_destroy(matte);
        if (input) gs_texture_destroy(input);
        gs_effect_destroy(effect);
    }
    obs_leave_graphics();
    obs_shutdown();
    std::cout << (okay ? "PASS subject effects GPU compositing\n"
                       : "FAIL subject effects GPU compositing\n");
    return okay ? 0 : 1;
}
