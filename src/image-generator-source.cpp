#include "common/preferences.hpp"
#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/dstr.h>
#include <util/platform.h>
#include <util/threading.h>
#include <media-io/video-io.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "engines/stable_diffusion_backend.hpp"
#include "core/native_sd_feedback.hpp"
#include "core/native_sd_exposure.hpp"
#include "core/native_sd_schedule.hpp"
#include "core/native_sd_preview.hpp"
#include "core/native_sd_preview_flow.hpp"
#include <graphics/vec2.h>
#include <graphics/vec4.h>
#include <util/bmem.h>
#include "core/native_sd_session.hpp"
#include <filesystem>
#include <chrono>
#ifdef SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE
#include "core/onnx_depth_processor.hpp"
#include <opencv2/imgproc.hpp>
#endif
#ifdef SIGHT_SD_GENERATOR_CUDA_GUARD
#include <cuda_runtime_api.h>
#endif

// Keep the image loader private to this source: VRM already provides an
// independent stb_image implementation in the same OBS plugin module.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "common/stb_image.h"

#define S_MODEL_PATH "model_path"
#define S_DIFFUSION_PATH "diffusion_model_path"
#define S_HIGH_NOISE_PATH "high_noise_diffusion_model_path"
#define S_VAE_PATH "vae_path"
#define S_TAESD_PATH "taesd_path"
#define S_CLIP_L_PATH "clip_l_path"
#define S_CLIP_G_PATH "clip_g_path"
#define S_T5_PATH "t5xxl_path"
#define S_LLM_PATH "llm_path"
#define S_LORA_PATH "lora_path"
#define S_SAMPLER_NAME "sampler_name"
#define S_SCHEDULER_NAME "scheduler_name"
#define S_CACHE_MODE "cache_mode"
#define S_DISTILLED_GUIDANCE "distilled_guidance"
#define S_PROMPT "prompt"
#define S_NEGATIVE_PROMPT "negative_prompt"
#define S_INIT_IMAGE "init_image"
#define S_STEPS "steps"
#define S_CFG_SCALE "cfg_scale"
#define S_SEED "seed"
#define S_WIDTH "width"
#define S_HEIGHT "height"
#define S_BUTTON_GENERATE "generate_button"

#define S_ENDLESS_MODE "endless_mode"
#define S_ZOOM "zoom_factor"
#define S_PAN_X "pan_x"
#define S_PAN_Y "pan_y"
#define S_SAVE_EVERY "save_every_frame"
#define S_OUTPUT_DIR "output_dir"
#define S_STRENGTH "denoising_strength"

struct image_gen_data {
    obs_source_t *context;
    
    // Properties
    std::string model_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string lora_path;
    std::string sampler_name;
    std::string scheduler_name;
    std::string cache_mode;
    float distilled_guidance = -1.0f;
    std::string prompt;
    std::string negative_prompt;
    std::string init_image_path;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    std::atomic<int> width{512},height{512};
    
    // Endless mode properties
    bool endless_mode = false;
    float zoom_factor = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    bool save_every_frame = false;
    std::string output_dir;
    float strength = 0.7f;

    // State
    gs_texture_t *texture = nullptr;
    gs_texture_t *previous_texture=nullptr;
    gs_effect_t *preview_effect=nullptr;
    gs_texture_t *preview_flow_texture=nullptr;
    sight::sd::PreviewFlowResult pending_flow{};
    float displayed_flow_coverage=0;
    std::atomic<bool> preview_flow{false};
    std::atomic<uint64_t> flow_draws{0};
    std::atomic<float> drawn_flow_coverage{0.f};
    sight::sd::PreviewKey pending_key{},displayed_key{};
    uint64_t blend_started=0,blend_duration=0;
    std::atomic<unsigned> preview_ms{0};
    std::atomic<uint64_t> preview_draws{0};
    std::atomic<float> preview_progress{1.f};
    std::vector<uint8_t> texture_data;
    uint32_t texture_width = 0;
    uint32_t texture_height = 0;
    bool new_texture_ready = false;
    
    std::atomic<uint64_t> epoch{0},published{0},feedback_frames{0},guided_frames{0};
    std::atomic<uint64_t> exposure_frames{0};
    std::atomic<bool> context_loaded{false};

    // Worker
    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};
    std::atomic<bool> trigger_generation{false};
    std::mutex curr_mutex; // Protects data exchanged with worker (only path/prompt/etc copies)
    std::mutex param_mutex; // Protects worker_job parameters
    
    // Internal copies for the worker to avoid locking the main struct too long
    struct {
         std::string device,controlnet,guide,depth_model;
         std::string motion_schedule,prompt_schedule;
         int seed_stride=0,refresh_interval=0,frame_limit=0;
         float exposure_strength=0;
         float control_strength=.65f;
         bool continuous=false;
         std::string model_path;
         std::string diffusion_model_path;
         std::string high_noise_diffusion_model_path;
         std::string vae_path;
         std::string taesd_path;
         std::string clip_l_path;
         std::string clip_g_path;
         std::string t5xxl_path;
         std::string llm_path;
         std::string lora_path;
         std::string sampler_name;
         std::string scheduler_name;
         std::string cache_mode;
         float distilled_guidance;
         std::string prompt;
         std::string negative_prompt;
         std::string init_image_path;
         int steps;
         float cfg_scale;
         int64_t seed;
         int width;
         int height;

         bool endless_mode;
         TransformParams transform;
         bool save_every_frame;
         std::string output_dir;
         float strength;
    } worker_job{};
    
    obs_hotkey_id hotkey_gen = OBS_INVALID_HOTKEY_ID;
    
    mutable std::mutex status_mutex;
    std::string status_buffer;
};

// -------------------------------------------------------------------------------------------------
// WORKER THREAD
// -------------------------------------------------------------------------------------------------

static void set_status(image_gen_data *data, const char *msg) {
    std::lock_guard<std::mutex> lock(data->status_mutex);
    data->status_buffer = msg ? msg : "";
}

// Simple PPM Writer (Uncompressed RGB)
static bool save_ppm(const std::string& path, int width, int height, const uint8_t* data) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    // P6 header
    ofs << "P6\n" << width << " " << height << "\n255\n";

    // Write pixel data (convert RGBA to RGB)
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        rgb[i*3+0] = data[i*4+0];
        rgb[i*3+1] = data[i*4+1];
        rgb[i*3+2] = data[i*4+2];
    }

    ofs.write((const char*)rgb.data(), rgb.size());
    ofs.flush();
    return bool(ofs);
}

using Backend=lll::engines::StableDiffusionBackend;
static void update(void *data, obs_data_t *settings);
static void command(image_gen_data*d,bool start){
    // OBS defers video-source update callbacks. Start must snapshot the
    // source's current settings even before the next video tick.
    if(start){
        if(auto*settings=obs_source_get_settings(d->context)){
            update(d,settings);
            obs_data_release(settings);
        }
    }
    std::lock_guard lock(d->param_mutex);
    ++d->epoch;
    if(start){d->published=0;d->feedback_frames=0;d->guided_frames=0;d->exposure_frames=0;}
    d->trigger_generation=start;
    set_status(d,start?"Queued":"Stopped");
}
static bool headroom(const std::string&device,bool cold,bool control){
    if(device=="cpu")return true;
#ifdef SIGHT_SD_GENERATOR_CUDA_GUARD
    size_t free=0,total=0;
    return device=="CUDA0"&&cudaSetDevice(0)==cudaSuccess&&cudaMemGetInfo(&free,&total)==cudaSuccess&&
        free>=(cold?(control?6000ull:4500ull):700ull)*1024*1024;
#else
    (void)cold;(void)control;return false;
#endif
}
static void worker_func(image_gen_data *d) {
    Backend backend;
    std::unique_lock<std::mutex> slot(sight::sd::session_slot,std::defer_lock);
    Backend::EngineConfig active{};
    sight::sd::Feedback feedback;
    sight::sd::ExposureAnchor exposure;
    std::string anchor_prompt;
    sight::sd::PreviewFlow preview_flow_estimator;
    uint64_t run=0,frame=0;
    int64_t base_seed=42;
    sight::sd::AnimationSchedule schedule;
    std::string parsed_motion,parsed_prompts;
    bool schedule_valid=false;
#ifdef SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE
    std::unique_ptr<lll::OnnxDepthProcessor> depth;
    std::string depth_path;
#endif
    auto release=[&]{
        backend.shutdown();d->context_loaded=false;
#ifdef SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE
        depth.reset();depth_path.clear();
#endif
        if(slot.owns_lock())slot.unlock();
    };
    auto stale=[&](uint64_t id){return d->stop_thread||!d->trigger_generation||d->epoch!=id;};
    while(!d->stop_thread){
        decltype(d->worker_job) job;
        uint64_t id;
        {
            std::lock_guard lock(d->param_mutex);
            id=d->epoch;job=d->worker_job;
        }
        if(!d->trigger_generation){
            if(slot.owns_lock())release();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));continue;
        }
        try{
            if(stale(id))continue;
            if(id!=run){
                feedback.reset();exposure.reset();anchor_prompt.clear();frame=0;run=id;
                base_seed=job.seed>=0?job.seed:int64_t(uint64_t(std::chrono::system_clock::now().time_since_epoch().count())&uint64_t(INT64_MAX));
            }
            if(job.endless_mode&&job.frame_limit>0&&frame>=uint64_t(job.frame_limit)){
                std::lock_guard lock(d->param_mutex);
                if(!stale(id)){d->trigger_generation=false;set_status(d,"Sequence complete");}
                continue;
            }
            if(!sight::sd::valid_feedback_size(job.width,job.height)||job.steps<1||job.steps>50||
                !std::isfinite(job.cfg_scale)||job.cfg_scale<0||job.cfg_scale>20||!std::isfinite(job.strength)||job.strength<0||job.strength>1)
                throw std::runtime_error("Invalid generator dimensions or sampling settings");
            sight::sd::AnimationValues animation{job.transform,job.strength,job.control_strength,job.prompt};
            if(job.endless_mode){
                if(!schedule_valid||job.motion_schedule!=parsed_motion||job.prompt_schedule!=parsed_prompts){
                    std::string error;
                    if(!schedule.parse(job.motion_schedule,job.prompt_schedule,error))
                        throw std::runtime_error(error);
                    parsed_motion=job.motion_schedule;parsed_prompts=job.prompt_schedule;schedule_valid=true;
                }
            }
            const bool guided=job.guide!="off";
            if(!std::isfinite(job.control_strength)||job.control_strength<0||job.control_strength>1)
                throw std::runtime_error("Invalid ControlNet strength");
            if(guided&&(!job.endless_mode||job.controlnet.empty()||
                !std::filesystem::is_regular_file(job.controlnet)||
                (job.guide!="canny"&&job.guide!="tile"&&job.guide!="depth")))
                throw std::runtime_error("Feedback guidance needs Endless Mode and a matching local ControlNet");
            if(job.guide=="depth"&&!std::filesystem::is_regular_file(job.depth_model))
                throw std::runtime_error("Depth guidance requires a local Depth Anything V2 ONNX model");
            Backend::EngineConfig config{};
            config.model_path=job.model_path;config.diffusion_model_path=job.diffusion_model_path;
            config.high_noise_diffusion_model_path=job.high_noise_diffusion_model_path;
            config.vae_path=job.vae_path;config.taesd_path=job.taesd_path;
            config.clip_l_path=job.clip_l_path;config.clip_g_path=job.clip_g_path;
            config.t5xxl_path=job.t5xxl_path;config.llm_path=job.llm_path;config.lora_path=job.lora_path;
            config.controlnet_path=guided?job.controlnet:"";config.backend_device=job.device;
            config.use_gpu=job.device!="cpu";config.num_threads=2;
            const bool same=backend.is_ready()&&active.model_path==config.model_path&&
                active.diffusion_model_path==config.diffusion_model_path&&
                active.high_noise_diffusion_model_path==config.high_noise_diffusion_model_path&&
                active.vae_path==config.vae_path&&active.taesd_path==config.taesd_path&&
                active.clip_l_path==config.clip_l_path&&active.clip_g_path==config.clip_g_path&&
                active.t5xxl_path==config.t5xxl_path&&active.llm_path==config.llm_path&&
                active.lora_path==config.lora_path&&active.controlnet_path==config.controlnet_path&&
                active.backend_device==config.backend_device;
            if(!same){
                release();feedback.reset();exposure.reset();anchor_prompt.clear();frame=0;
                if(!slot.try_lock())throw std::runtime_error("Another native diffusion context is active");
                if(config.model_path.empty()&&config.diffusion_model_path.empty())throw std::runtime_error("Select a local checkpoint");
                if(!headroom(job.device,true,guided))throw std::runtime_error("Insufficient CUDA headroom or CUDA backend unavailable; no CPU fallback");
                set_status(d,"Loading native diffusion model");
                if(!backend.initialize(config))throw std::runtime_error(backend.last_error());
                if((job.continuous||guided)&&backend.model_version()!="SD 1.x")
                    throw std::runtime_error("Continuous denoise and guided feedback currently require SD 1.x");
                active=config;d->context_loaded=true;
            }
            if((job.continuous||guided)&&backend.model_version()!="SD 1.x")
                throw std::runtime_error("Continuous denoise and guided feedback currently require SD 1.x");
            if(stale(id))continue;
            if(!headroom(job.device,false,guided))throw std::runtime_error("Insufficient free CUDA memory");
            if(job.endless_mode)animation=schedule.evaluate(frame,std::move(animation),job.prompt);
            Backend::GenerationParams p{};
            p.prompt=animation.prompt;p.negative_prompt=job.negative_prompt;
            p.width=job.width;p.height=job.height;p.steps=job.steps;
            p.cfg_scale=job.cfg_scale;p.strength=animation.denoise;p.continuous_denoise=job.continuous;
            p.seed=job.endless_mode?sight::sd::feedback_seed(base_seed,frame,job.seed_stride):base_seed;
            p.sample_method_name=job.sampler_name;p.scheduler_name=job.scheduler_name;
            p.cache_mode=job.cache_mode;p.distilled_guidance=job.distilled_guidance;
            const bool refresh=frame>0&&job.refresh_interval>0&&frame%uint64_t(job.refresh_interval)==0;
            std::vector<uint8_t> init,guide;
            if(job.endless_mode&&!refresh&&!feedback.rgba().empty()){
                init=feedback.init_rgb(p.width,p.height,animation.transform);
                if(init.empty())throw std::runtime_error("Invalid feedback geometry or transform");
            }else if(frame==0&&!job.init_image_path.empty()){
                int iw=0,ih=0,ic=0;
                std::unique_ptr<uint8_t,decltype(&stbi_image_free)> img(stbi_load(job.init_image_path.c_str(),&iw,&ih,&ic,4),&stbi_image_free);
                if(!img||iw<=0||ih<=0)throw std::runtime_error("Unable to read selected initial image");
                std::vector<uint8_t> resized(size_t(p.width)*p.height*4);
                if(!resize_bilinear(img.get(),size_t(iw)*ih*4,iw,ih,resized.data(),resized.size(),p.width,p.height))
                    throw std::runtime_error("Initial image resize failed");
                init.resize(size_t(p.width)*p.height*3);
                lll::convert_rgba_to_rgb(resized.data(),init.data(),size_t(p.width)*p.height);
            }
            const bool recursive=job.endless_mode&&frame>0&&!refresh&&!init.empty();
            if(!init.empty())p.init_image=init.data();
            if(guided&&!init.empty()){
#ifdef SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE
                if(job.guide=="depth"){
                    if(!depth||depth_path!=job.depth_model){
                        depth=std::make_unique<lll::OnnxDepthProcessor>();
                        if(!depth->load(job.depth_model,false))throw std::runtime_error(depth->last_error());
                        depth_path=job.depth_model;
                    }
                    const double scale=std::min(1.,280./std::max(p.width,p.height));
                    const int dw=std::max(14,int(std::round(p.width*scale/14))*14);
                    const int dh=std::max(14,int(std::round(p.height*scale/14))*14);
                    cv::Mat small,large;
                    cv::resize(cv::Mat(int(p.height),int(p.width),CV_8UC3,init.data()),small,cv::Size(dw,dh),0,0,cv::INTER_AREA);
                    std::vector<uint8_t> gray;
                    if(!depth->estimate(small.data,dw,dh,gray)||gray.size()!=size_t(dw)*dh)
                        throw std::runtime_error("Depth preprocessor failed: "+depth->last_error());
                    cv::resize(cv::Mat(dh,dw,CV_8UC1,gray.data()),large,cv::Size(p.width,p.height),0,0,cv::INTER_LINEAR);
                    guide.resize(init.size());
                    for(size_t i=0;i<large.total();++i)for(unsigned c=0;c<3;++c)guide[i*3+c]=large.data[i];
                }else if(!sight::sd::make_native_guide(init,p.width,p.height,job.guide,guide))
                    throw std::runtime_error("Native feedback preprocessor failed");
#else
                throw std::runtime_error("Feedback preprocessing unavailable in this build");
#endif
                p.control_image=guide.data();p.control_strength=animation.control_strength;
            }
            set_status(d,recursive?"Generating feedback frame":"Generating initial image");
            backend.set_progress_callback([&](int,int,float){if(stale(id))backend.cancel_current_generation();});
            auto output=backend.generate(p);backend.set_progress_callback({});
            if(stale(id))continue;
            if(output.width!=p.width||output.height!=p.height||
               (output.channels!=3&&output.channels!=4)||output.data.size()!=size_t(p.width)*p.height*output.channels)
                throw std::runtime_error("Invalid native image output: "+backend.last_error());
            sight::sd::ExposureResult exposure_result;
            if(job.endless_mode){
                const bool new_anchor=!recursive||anchor_prompt!=p.prompt;
                exposure_result=exposure.apply(output.data,output.channels,new_anchor,job.exposure_strength);
                anchor_prompt=p.prompt;
            }
            if(!feedback.accept(output.data,output.width,output.height,output.channels))
                throw std::runtime_error("Invalid native image output: "+backend.last_error());
            // The init already contains the commanded camera transform.
            // Estimate only the remaining scene motion, once per keyframe.
            sight::sd::PreviewFlowResult flow;
            if(recursive&&d->preview_ms&&d->preview_flow)
                flow=preview_flow_estimator.estimate(init,feedback.rgba(),p.width,p.height);

            {
                std::lock_guard lock(d->param_mutex);
                if(stale(id))continue;
                {
                    std::lock_guard publish(d->curr_mutex);
                    d->texture_data=feedback.rgba();d->texture_width=p.width;d->texture_height=p.height;
                    d->pending_key={id,frame,uint32_t(p.width),uint32_t(p.height),os_gettime_ns(),recursive,animation.transform};
                    d->pending_flow=std::move(flow);
                    d->new_texture_ready=true;
                }
                ++d->published;if(recursive)++d->feedback_frames;if(!guide.empty())++d->guided_frames;
                if(exposure_result.applied)++d->exposure_frames;
                // Export the last completed frame before reporting completion.
                if(!job.endless_mode)d->trigger_generation=false;
            }
            if(exposure_result.applied&&(frame<3||frame%60==0))
                blog(LOG_INFO,"[Sight] SIGHT_SD_EXPOSURE_20261124_E61 frame=%llu before=%.3f after=%.3f",
                     (unsigned long long)frame,exposure_result.before_error,exposure_result.after_error);
            if(job.endless_mode&&(!job.motion_schedule.empty()||!job.prompt_schedule.empty())&&(frame<3||frame%60==0))
                blog(LOG_INFO,"[Sight] SIGHT_SD_SCHEDULE_20261124_K82 frame=%llu zoom=%.4f pan_x=%.3f denoise=%.3f prompt_changed=%d",
                     (unsigned long long)frame,animation.transform.zoom_factor,animation.transform.pan_x,
                     animation.denoise,int(p.prompt!=job.prompt));
            if(frame<3||frame%60==0)blog(LOG_INFO,"[Sight] SIGHT_SD_FEEDBACK_20261124_F13 frame=%llu feedback=%d guided=%d seed=%lld",
                (unsigned long long)frame,int(recursive),int(!guide.empty()),(long long)p.seed);
            if(job.endless_mode&&job.save_every_frame&&!job.output_dir.empty()){
                auto file=std::filesystem::path(job.output_dir)/("frame_"+std::to_string(base_seed)+"_"+std::to_string(id)+"_"+std::to_string(frame)+".ppm");
                if(std::filesystem::exists(file)||!save_ppm(file.string(),p.width,p.height,feedback.rgba().data()))
                    throw std::runtime_error("Frame export failed or filename already exists");
            }
            ++frame;
            {
                std::lock_guard lock(d->param_mutex);
                if(d->epoch==id){
                    if(!job.endless_mode)set_status(d,"Idle");
                    else if(d->trigger_generation&&d->worker_job.frame_limit>0&&frame>=uint64_t(d->worker_job.frame_limit)){
                        d->trigger_generation=false;set_status(d,"Sequence complete");
                    }
                }
            }
        }catch(const std::exception&e){
            backend.set_progress_callback({});
            release();feedback.reset();
            std::lock_guard lock(d->param_mutex);
            if(d->epoch==id){d->trigger_generation=false;set_status(d,e.what());blog(LOG_WARNING,"[Sight/Generator] %s",e.what());}
        }
    }
    release();
}

// -------------------------------------------------------------------------------------------------
// OBS CALLBACKS
// -------------------------------------------------------------------------------------------------

static const char *get_name(void *unused) { return "lll.sight Generator (Experimental)"; }

static void *create(obs_data_t *settings, obs_source_t *context) {
    image_gen_data *data = new image_gen_data();
    data->context = context;
    
    update(data,settings);
    obs_enter_graphics();
    if(char*path=obs_module_file("sight-generator-preview.effect")){
        char*errors=nullptr;
        data->preview_effect=gs_effect_create_from_file(path,&errors);
        if(errors){blog(LOG_WARNING,"[Sight/Generator] Preview effect: %s",errors);bfree(errors);}
        bfree(path);
    }
    obs_leave_graphics();
    data->worker_thread = std::thread(worker_func, data);
    
    // Hotkey
    auto hotkey_cb = [](void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
        if (pressed) {
            auto *d = (image_gen_data*)data;
            command(d,true);
        }
    };
    data->hotkey_gen = obs_hotkey_register_source(context, "ai_image_gen.trigger", "Generate AI Image", hotkey_cb, data);
    
    auto*proc=obs_source_get_proc_handler(context);
    proc_handler_add(proc,"void sight_generator_start_v1()",[](void*ctx,calldata_t*){command(static_cast<image_gen_data*>(ctx),true);},data);
    proc_handler_add(proc,"void sight_generator_stop_v1()",[](void*ctx,calldata_t*){command(static_cast<image_gen_data*>(ctx),false);},data);
    proc_handler_add(proc,"void sight_generator_stats_v1(out int frames,out int feedback,out int guided,out bool running,out bool loaded,out string status)",[](void*ctx,calldata_t*args){
        auto*d=static_cast<image_gen_data*>(ctx);
        std::lock_guard state(d->param_mutex);
        calldata_set_int(args,"frames",d->published);calldata_set_int(args,"feedback",d->feedback_frames);
        calldata_set_int(args,"guided",d->guided_frames);calldata_set_bool(args,"running",d->trigger_generation);
        calldata_set_bool(args,"loaded",d->context_loaded);
        std::lock_guard lock(d->status_mutex);calldata_set_string(args,"status",d->status_buffer.c_str());
    },data);
    proc_handler_add(proc,"void sight_generator_exposure_stats_v1(out int frames)",[](void*ctx,calldata_t*args){
        auto*d=static_cast<image_gen_data*>(ctx);
        calldata_set_int(args,"frames",d->exposure_frames.load());
    },data);
    proc_handler_add(proc,"void sight_generator_preview_stats_v1(out int draws,out float phase,out bool available)",[](void*ctx,calldata_t*args){
        auto*d=static_cast<image_gen_data*>(ctx);
        calldata_set_int(args,"draws",d->preview_draws.load());
        calldata_set_float(args,"phase",d->preview_progress.load());
        calldata_set_bool(args,"available",d->preview_effect!=nullptr);
    },data);
    proc_handler_add(proc,"void sight_generator_preview_flow_stats_v1(out int draws,out float coverage)",[](void*ctx,calldata_t*args){
        auto*d=static_cast<image_gen_data*>(ctx);
        calldata_set_int(args,"draws",d->flow_draws.load());
        calldata_set_float(args,"coverage",d->drawn_flow_coverage.load());
    },data);
    return data;
}

static void destroy(void *data) {
    image_gen_data *d = (image_gen_data*)data;
    command(d,false);
    d->stop_thread = true;
    obs_hotkey_unregister(d->hotkey_gen);
    if (d->worker_thread.joinable()) d->worker_thread.join();
    // The native context belongs to the finished worker and has been freed.
    obs_enter_graphics();
    if (d->texture) gs_texture_destroy(d->texture);
    if(d->previous_texture)gs_texture_destroy(d->previous_texture);
    if(d->preview_effect)gs_effect_destroy(d->preview_effect);
    if(d->preview_flow_texture)gs_texture_destroy(d->preview_flow_texture);
    obs_leave_graphics();
    delete d;
}

static void update(void *data, obs_data_t *settings) {
    image_gen_data *d = (image_gen_data*)data;
    
    std::lock_guard<std::mutex> lock(d->param_mutex);
    const auto previous=d->worker_job;
    std::string model_path = obs_data_get_string(settings, S_MODEL_PATH);
    std::string init_image_path = obs_data_get_string(settings, S_INIT_IMAGE);
    std::string output_dir = obs_data_get_string(settings, S_OUTPUT_DIR);
    
    // These are explicit local file-picker paths. Absolute paths must remain
    // valid; the former blanket rejection made normal OBS file selections
    // unusable. Model loaders report missing or incompatible files.
    d->model_path = model_path;
    d->diffusion_model_path = obs_data_get_string(settings, S_DIFFUSION_PATH);
    d->high_noise_diffusion_model_path = obs_data_get_string(settings, S_HIGH_NOISE_PATH);
    d->taesd_path = obs_data_get_string(settings, S_TAESD_PATH);
    d->vae_path = obs_data_get_string(settings, S_VAE_PATH);
    d->clip_l_path = obs_data_get_string(settings, S_CLIP_L_PATH);
    d->clip_g_path = obs_data_get_string(settings, S_CLIP_G_PATH);
    d->t5xxl_path = obs_data_get_string(settings, S_T5_PATH);
    d->llm_path = obs_data_get_string(settings, S_LLM_PATH);
    d->lora_path = obs_data_get_string(settings, S_LORA_PATH);
    d->sampler_name = obs_data_get_string(settings, S_SAMPLER_NAME);
    d->scheduler_name = obs_data_get_string(settings, S_SCHEDULER_NAME);
    d->cache_mode = obs_data_get_string(settings, S_CACHE_MODE);
    d->distilled_guidance = (float)obs_data_get_double(settings, S_DISTILLED_GUIDANCE);
    d->prompt = obs_data_get_string(settings, S_PROMPT);
    d->negative_prompt = obs_data_get_string(settings, S_NEGATIVE_PROMPT);
    d->init_image_path = init_image_path;
    d->steps = obs_data_get_int(settings, S_STEPS);
    d->cfg_scale = (float)obs_data_get_double(settings, S_CFG_SCALE);
    d->seed = obs_data_get_int(settings, S_SEED);
    auto dimension=[&](const char*key){return int(std::clamp<int64_t>(obs_data_get_int(settings,key),128,1024)/64*64);};
    d->width=dimension(S_WIDTH);d->height=dimension(S_HEIGHT);
    
    // Endless properties
    d->endless_mode = obs_data_get_bool(settings, S_ENDLESS_MODE);
    d->zoom_factor = (float)obs_data_get_double(settings, S_ZOOM);
    d->pan_x = (float)obs_data_get_double(settings, S_PAN_X);
    d->pan_y = (float)obs_data_get_double(settings, S_PAN_Y);
    d->save_every_frame = obs_data_get_bool(settings, S_SAVE_EVERY);
    d->output_dir = output_dir;
    d->strength = (float)obs_data_get_double(settings, S_STRENGTH);

    // Update worker job
    d->worker_job.model_path = d->model_path;
    d->worker_job.diffusion_model_path = d->diffusion_model_path;
    d->worker_job.high_noise_diffusion_model_path = d->high_noise_diffusion_model_path;
    d->worker_job.vae_path = d->vae_path;
    d->worker_job.taesd_path = d->taesd_path;
    d->worker_job.cache_mode = d->cache_mode;
    d->worker_job.clip_l_path = d->clip_l_path;
    d->worker_job.clip_g_path = d->clip_g_path;
    d->worker_job.t5xxl_path = d->t5xxl_path;
    d->worker_job.llm_path = d->llm_path;
    d->worker_job.lora_path = d->lora_path;
    d->worker_job.sampler_name = d->sampler_name;
    d->worker_job.scheduler_name = d->scheduler_name;
    d->worker_job.distilled_guidance = d->distilled_guidance;
    d->worker_job.prompt = d->prompt;
    d->worker_job.negative_prompt = d->negative_prompt;
    d->worker_job.init_image_path = d->init_image_path;
    d->worker_job.steps = d->steps;
    d->worker_job.cfg_scale = d->cfg_scale;
    d->worker_job.seed = d->seed;
    d->worker_job.width = d->width;
    d->worker_job.height = d->height;

    d->worker_job.endless_mode = d->endless_mode;
    d->worker_job.transform.zoom_factor = d->zoom_factor;
    d->worker_job.transform.pan_x = d->pan_x;
    d->worker_job.transform.pan_y = d->pan_y;
    d->worker_job.save_every_frame = d->save_every_frame;
    d->worker_job.output_dir = d->output_dir;
    d->worker_job.strength = d->strength;
    auto&j=d->worker_job;
    j.device=obs_data_get_string(settings,"device");
    j.controlnet=obs_data_get_string(settings,"controlnet_path");
    j.guide=obs_data_get_string(settings,"feedback_guide");
    j.depth_model=obs_data_get_string(settings,"guide_depth_model");
    j.motion_schedule=obs_data_get_string(settings,"motion_schedule");
    j.prompt_schedule=obs_data_get_string(settings,"prompt_schedule");
    d->preview_ms=unsigned(std::clamp<int64_t>(obs_data_get_int(settings,"preview_ms"),0,2000));
#ifdef SIGHT_HAVE_DENSE_FLOW
    d->preview_flow=obs_data_get_bool(settings,"preview_flow");
#endif
    j.control_strength=float(obs_data_get_double(settings,"control_strength"));
    j.continuous=obs_data_get_bool(settings,"continuous_denoise");
    j.seed_stride=int(std::clamp<int64_t>(obs_data_get_int(settings,"seed_stride"),0,1000));
    j.refresh_interval=int(std::clamp<int64_t>(obs_data_get_int(settings,"refresh_interval"),0,10000));
    j.frame_limit=int(std::clamp<int64_t>(obs_data_get_int(settings,"frame_limit"),0,1000000));
    const auto exposure_value=obs_data_get_double(settings,"exposure_strength");
    j.exposure_strength=std::isfinite(exposure_value)?float(std::clamp(exposure_value,0.,1.)):0.f;
    j.transform.rotation_deg=float(obs_data_get_double(settings,"rotation_deg"));
    // Prompt/denoise/motion edits apply on the next frame. A new geometry,
    // seed, model or mode invalidates in-flight feedback instead of mixing runs.
    if(d->trigger_generation&&(previous.width!=j.width||previous.height!=j.height||
       previous.seed!=j.seed||previous.init_image_path!=j.init_image_path||
       previous.endless_mode!=j.endless_mode||previous.device!=j.device||
       previous.model_path!=j.model_path||previous.diffusion_model_path!=j.diffusion_model_path||
       previous.high_noise_diffusion_model_path!=j.high_noise_diffusion_model_path||
       previous.vae_path!=j.vae_path||previous.taesd_path!=j.taesd_path||
       previous.clip_l_path!=j.clip_l_path||previous.clip_g_path!=j.clip_g_path||
       previous.t5xxl_path!=j.t5xxl_path||previous.llm_path!=j.llm_path||previous.lora_path!=j.lora_path||
       previous.controlnet!=j.controlnet||previous.guide!=j.guide||previous.depth_model!=j.depth_model)){
        ++d->epoch;d->published=0;d->feedback_frames=0;d->guided_frames=0;d->exposure_frames=0;
    }
}

static bool generate_clicked(obs_properties_t *props, obs_property_t *p, void *data) {
    image_gen_data *d = (image_gen_data*)data;
    command(d,true);
    return false;
}

static bool stop_clicked(obs_properties_t*,obs_property_t*,void*data){
    command(static_cast<image_gen_data*>(data),false);return false;
}
static obs_properties_t *get_properties(void *data) {
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_path(props, S_MODEL_PATH, "Checkpoint / GGUF (or use split components below)", OBS_PATH_FILE, "*.safetensors *.gguf *.ckpt", sight_preferences::directory().c_str());
    // Split model components are required by some newer image families.
    obs_properties_t* components = obs_properties_create();
    obs_properties_add_path(components, S_DIFFUSION_PATH, "Diffusion model (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_HIGH_NOISE_PATH, "High-noise diffusion model (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_TAESD_PATH, "TAESD fast decoder (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_VAE_PATH, "VAE (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_CLIP_L_PATH, "CLIP-L (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_CLIP_G_PATH, "CLIP-G (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_T5_PATH, "T5-XXL (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_LLM_PATH, "Text encoder / LLM (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_path(components, S_LORA_PATH, "LoRA (optional)", OBS_PATH_FILE, "*.safetensors *.gguf", sight_preferences::directory().c_str());
    obs_properties_add_group(props, "split_model_group", "Advanced model components", OBS_GROUP_NORMAL, components);
    auto*device=obs_properties_add_list(props,"device","Inference device",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(device,"CUDA0","CUDA0");
    obs_property_list_add_string(device,"CPU (diagnostic)","cpu");
    obs_properties_add_text(props, S_PROMPT, "Prompt", OBS_TEXT_MULTILINE);
    obs_properties_add_text(props, S_NEGATIVE_PROMPT, "Negative Prompt", OBS_TEXT_DEFAULT);
    obs_properties_add_path(props, S_INIT_IMAGE, "Init Image (Optional)", OBS_PATH_FILE, "*.png *.jpg *.jpeg", NULL);
    
    obs_properties_add_int(props, S_WIDTH, "Width", 128, 1024, 64);
    obs_properties_add_int(props, S_HEIGHT, "Height", 128, 1024, 64);
    obs_properties_add_int(props, S_STEPS, "Steps", 1, 50, 1);
    obs_properties_add_float_slider(props, S_CFG_SCALE, "CFG Scale", 0.0, 20.0, 0.5);
    obs_properties_add_float(props, S_DISTILLED_GUIDANCE, "Distilled guidance (-1 = model default)", -1.0, 30.0, 0.5);
    obs_properties_add_text(props, S_SAMPLER_NAME, "Sampler (empty = model default)", OBS_TEXT_DEFAULT);
    obs_properties_add_text(props, S_SCHEDULER_NAME, "Scheduler (empty = model default)", OBS_TEXT_DEFAULT);
    obs_property_t* cache = obs_properties_add_list(props, S_CACHE_MODE, "Generation cache", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(cache, "Model default", "");
    obs_property_list_add_string(cache, "Disabled", "disabled");
    obs_property_list_add_string(cache, "EasyCache", "easycache");
    obs_property_list_add_string(cache, "UCache", "ucache");
    obs_property_list_add_string(cache, "DBCache", "dbcache");
    obs_property_list_add_string(cache, "TaylorSeer", "taylorseer");
    obs_property_list_add_string(cache, "CacheDiT", "cache_dit");
    obs_property_list_add_string(cache, "Spectrum", "spectrum");
    obs_properties_add_int(props, S_SEED, "Seed (-1 = random)", -1, 2147483647, 1);
    
    // Endless Mode Group
    obs_properties_t *grp = obs_properties_create();
    obs_properties_add_group(props, "endless_group", "Recursive prompt video", OBS_GROUP_NORMAL, grp);

    obs_properties_add_bool(grp, S_ENDLESS_MODE, "Enable recursive feedback");
    obs_properties_add_float_slider(grp, S_ZOOM, "Zoom per generated frame (1.0 = none)", 0.9, 1.1, 0.001);
    obs_properties_add_float(grp, S_PAN_X, "Pan X per generated frame (pixels)", -100.0, 100.0, 0.1);
    obs_properties_add_float(grp, S_PAN_Y, "Pan Y per generated frame (pixels)", -100.0, 100.0, 0.1);

    obs_properties_add_float_slider(grp, S_STRENGTH, "Denoising Strength (0.1=Little Change, 0.9=Big Change)", 0.0, 1.0, 0.05);

    obs_properties_add_float_slider(grp,"rotation_deg","Rotation per generated frame (degrees)",-15.,15.,.1);
    obs_properties_add_int(grp,"seed_stride","Seed increment per frame (0 = fixed)",0,1000,1);
    obs_properties_add_int(grp,"refresh_interval","Prompt-only refresh every N frames (0 = off)",0,10000,1);
    obs_properties_add_int(grp,"frame_limit","Frame limit (0 = unlimited)",0,1000000,1);
    auto*exposure=obs_properties_add_float_slider(grp,"exposure_strength","Feedback exposure anchoring",0.,1.,.05);
    obs_property_set_long_description(exposure,"Limit cumulative brightness drift toward the first image of the current prompt segment. Bounded common RGB offset; refresh or prompt change establishes a new reference.");
    obs_properties_add_bool(grp,"continuous_denoise","Continuous noise level (SD 1.x / LCM)");
    auto*preview=obs_properties_add_int_slider(grp,"preview_ms","Camera-aware preview smoothing (ms)",0,2000,50);
    obs_property_set_long_description(preview,"Display-only motion-aligned interpolation between completed consecutive frames. Duration is capped by the generation interval. Zero disables smoothing; refreshes and restarts are cuts.");
#ifdef SIGHT_HAVE_DENSE_FLOW
    auto*flow=obs_properties_add_bool(grp,"preview_flow","Compensate object motion in preview");
    obs_property_set_long_description(flow,"Estimate residual motion on a bounded CPU proxy once per generated frame. Unreliable regions fall back to camera-aware interpolation. Requires nonzero smoothing.");
#endif
    auto*schedule=obs_properties_create();
    obs_properties_add_group(grp,"animation_schedule","Animation keyframes",OBS_GROUP_NORMAL,schedule);
    obs_properties_add_text(schedule,"motion_schedule","Motion and denoise keyframes",OBS_TEXT_MULTILINE);
    obs_properties_add_text(schedule,"prompt_schedule","Prompt keyframes",OBS_TEXT_MULTILINE);
    obs_properties_add_text(schedule,"schedule_help","One keyframe per line, e.g. 0: zoom=1.005, pan_x=1, denoise=.25\n120: zoom=1.02, pan_x=-1, denoise=.35\nFields: zoom, pan_x, pan_y, rotation, denoise, control. Numeric tracks interpolate; prompts switch at their frame. Frame 0 is the first image.",OBS_TEXT_INFO);
#ifdef SIGHT_HAVE_NATIVE_SD_IMAGE_SOURCE
    auto*guide=obs_properties_add_list(grp,"feedback_guide","Feedback conditioning",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(guide,"Off","off");
    obs_property_list_add_string(guide,"Canny edges","canny");
    obs_property_list_add_string(guide,"Tile / RGB","tile");
    obs_property_list_add_string(guide,"Depth Anything V2","depth");
    obs_properties_add_path(grp,"controlnet_path","Matching ControlNet weights",OBS_PATH_FILE,"*.safetensors *.gguf",sight_preferences::directory().c_str());
    obs_properties_add_path(grp,"guide_depth_model","Depth model (ONNX)",OBS_PATH_FILE,"*.onnx",sight_preferences::directory().c_str());
    obs_properties_add_float_slider(grp,"control_strength","ControlNet strength",0.,1.,.05);
    obs_properties_add_text(grp,"guide_help","Match Canny, tile or relative depth guidance to the corresponding SD1.x ControlNet model. Guidance uses the transformed feedback image; depth runs on CPU. No guide on a prompt-only refresh.",OBS_TEXT_INFO);
#endif
    obs_properties_add_bool(grp, S_SAVE_EVERY, "Save Every Frame");
    obs_properties_add_path(grp, S_OUTPUT_DIR, "Output Directory", OBS_PATH_DIRECTORY, NULL, NULL);

    obs_properties_add_button(props, S_BUTTON_GENERATE, "Start / Restart", generate_clicked);
    obs_properties_add_button(props,"stop_generation","Stop",stop_clicked);
    
    return props;
}

static void get_defaults(obs_data_t* settings) {
    obs_data_set_default_int(settings, S_WIDTH, 512);
    obs_data_set_default_int(settings, S_HEIGHT, 512);
    obs_data_set_default_int(settings, S_STEPS, 20);
    obs_data_set_default_double(settings, S_CFG_SCALE, 7.0);
    obs_data_set_default_double(settings, S_DISTILLED_GUIDANCE, -1.0);
    obs_data_set_default_int(settings, S_SEED, -1);
    obs_data_set_default_double(settings, S_STRENGTH, 0.7);
    obs_data_set_default_double(settings, S_ZOOM, 1.0);
#ifdef SIGHT_SD_GENERATOR_CUDA_GUARD
    obs_data_set_default_string(settings,"device","CUDA0");
#else
    obs_data_set_default_string(settings,"device","cpu");
#endif
    obs_data_set_default_string(settings,"feedback_guide","off");
    obs_data_set_default_double(settings,"control_strength",.65);
    obs_data_set_default_int(settings,"preview_ms",0);
    obs_data_set_default_int(settings,"frame_limit",0);
    obs_data_set_default_double(settings,"exposure_strength",0.);
}

static uint32_t get_width(void *data) { return ((image_gen_data*)data)->width; }
static uint32_t get_height(void *data) { return ((image_gen_data*)data)->height; }

static void video_render(void *data, gs_effect_t *) {
    auto*d=static_cast<image_gen_data*>(data);
    const auto now=os_gettime_ns();
    {
        std::unique_lock lock(d->curr_mutex,std::try_to_lock);
        if(lock.owns_lock()&&d->new_texture_ready){
            const auto key=d->pending_key;
            const bool pair=d->texture&&d->preview_effect&&d->preview_ms&&d->trigger_generation&&
                key.epoch==d->epoch&&sight::sd::can_interpolate(d->displayed_key,key);
            // Upload to the spare texture first so allocation failure never
            // discards the last successfully displayed image.
            if(d->previous_texture&&(gs_texture_get_width(d->previous_texture)!=key.width||
                                     gs_texture_get_height(d->previous_texture)!=key.height)){
                gs_texture_destroy(d->previous_texture);d->previous_texture=nullptr;
            }
            if(!d->previous_texture)
                d->previous_texture=gs_texture_create(key.width,key.height,GS_RGBA,1,nullptr,GS_DYNAMIC);
            if(d->previous_texture){
                gs_texture_set_image(d->previous_texture,d->texture_data.data(),key.width*4,false);
                std::swap(d->texture,d->previous_texture);
                d->blend_duration=pair?sight::sd::preview_duration_ns(d->displayed_key,key,d->preview_ms):0;
                d->displayed_flow_coverage=0;
                if(pair&&d->preview_flow&&d->pending_flow.coverage>0){
                    if(!d->preview_flow_texture)
                        d->preview_flow_texture=gs_texture_create(SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,1,nullptr,GS_DYNAMIC);
                    if(d->preview_flow_texture){
                        gs_texture_set_image(d->preview_flow_texture,
                            reinterpret_cast<const uint8_t*>(d->pending_flow.grid.data()),
                            SIGHT_FLOW_WIDTH*4*sizeof(float),false);
                        d->displayed_flow_coverage=d->pending_flow.coverage;
                    }
                }
                d->blend_started=now;d->displayed_key=key;d->new_texture_ready=false;
            }
        }
    }
    if(!d->texture)return;
    const bool smooth=d->preview_effect&&d->previous_texture&&d->preview_ms&&
        d->trigger_generation&&d->displayed_key.epoch==d->epoch;
    if(!smooth)d->blend_duration=0;
    const float phase=sight::sd::preview_phase(now-d->blend_started,d->blend_duration);
    d->preview_progress=phase;
    d->drawn_flow_coverage=0;
    // Respect a caller-owned effect rather than nesting effect loops.
    if(auto*active=gs_get_effect()){
        if(auto*image=gs_effect_get_param_by_name(active,"image")){
            gs_effect_set_texture(image,d->texture);
            gs_draw_sprite(d->texture,0,d->width,d->height);
        }
        d->preview_progress=1;return;
    }
    if(smooth&&phase<1){
        auto*fx=d->preview_effect;
        const auto map=sight::sd::preview_mapping(d->displayed_key.transform,phase);
        vec2 size;vec2_set(&size,float(d->displayed_key.width),float(d->displayed_key.height));
        vec4 a,b;vec4_set(&a,map.old_scale,map.old_angle,map.old_pan_x,map.old_pan_y);
        vec4_set(&b,map.new_scale,map.new_angle,map.new_pan_x,map.new_pan_y);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx,"image"),d->texture);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx,"previous_image"),d->previous_texture);
        gs_effect_set_vec2(gs_effect_get_param_by_name(fx,"image_size"),&size);
        gs_effect_set_vec4(gs_effect_get_param_by_name(fx,"old_transform"),&a);
        gs_effect_set_vec4(gs_effect_get_param_by_name(fx,"new_transform"),&b);
        gs_effect_set_float(gs_effect_get_param_by_name(fx,"phase"),phase);
        const bool use_flow=d->preview_flow&&d->preview_flow_texture&&d->displayed_flow_coverage>0;
        gs_effect_set_texture(gs_effect_get_param_by_name(fx,"residual_flow"),
            use_flow?d->preview_flow_texture:d->texture);
        gs_effect_set_float(gs_effect_get_param_by_name(fx,"flow_enabled"),use_flow?1.f:0.f);

        while(gs_effect_loop(fx,"Draw"))gs_draw_sprite(d->texture,0,d->width,d->height);
        if(use_flow){
            d->drawn_flow_coverage=d->displayed_flow_coverage;
            if(d->flow_draws.fetch_add(1)==0)
                blog(LOG_INFO,"[Sight] SIGHT_SD_PREVIEW_FLOW_20261124_B49 coverage=%.3f",d->displayed_flow_coverage);
        }
        if(d->preview_draws.fetch_add(1)==0)
            blog(LOG_INFO,"[Sight] SIGHT_SD_PREVIEW_20261124_M27 camera-aware draw");
    }else{
        auto*base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(base,"image"),d->texture);
        while(gs_effect_loop(base,"Draw"))gs_draw_sprite(d->texture,0,d->width,d->height);
    }
}

struct obs_source_info image_gen_source_info = {
    .id = "ai_image_generator",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
    .get_name = get_name,
    .create = create,
    .destroy = destroy,
    .get_width = get_width,
    .get_height = get_height,
    .get_defaults = get_defaults,
    .get_properties = get_properties,
    .update = update,
    .video_render = video_render,
};
