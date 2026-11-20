#include <obs-module.h>
OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("sight-ivme-denoise", "en-US")
extern obs_source_info sight_ivme_denoise_filter_info;
bool obs_module_load(void) {
    blog(LOG_INFO,"[Sight/Ivme] IVME_REAL_ONNX_FILTER_20261028");
    obs_register_source(&sight_ivme_denoise_filter_info);
    return true;
}
const char* obs_module_name(void) {return "Sight: Camera Denoise (Ivme)";}
const char* obs_module_description(void) {return "Native noise removal using Ivme ONNX, asynchronous and local";}
