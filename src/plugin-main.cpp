#include <obs-module.h>

#include <filesystem>
#include <sys/stat.h>

#include "common/sight-log.h"

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

#ifdef ENABLE_DOCK_PANEL
#include <obs-frontend-api.h>
#include <QMainWindow>
#include <QCoreApplication>
#include "ai-style-dock.hpp"
#endif

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("live-latent-lab-sight", "en-US")

// External reference to filter definition
#ifndef SIGHT_ANIMEGAN_ONLY
extern struct obs_source_info style_transfer_filter_info;
extern struct obs_source_info image_gen_source_info; // New Source
extern struct obs_source_info latent_codec_test_filter_info; // Latent Codec Test Filter
extern struct obs_source_info privacy_filter_info; // Privacy Filter
extern struct obs_source_info parallax_filter_info; // 2.5D Parallax Filter
extern struct obs_source_info mediapipe_filter_info; // Interactive Segment Filter
#endif
extern struct obs_source_info neural_enhance_filter_info; // Portrait Sketch
#ifdef SIGHT_ANIMEGAN_ONLY
extern struct obs_source_info sight_u2net_filter_info; // U2Net subject cutout
extern struct obs_source_info sight_depth_filter_info; // Depth Anything depth camera
extern struct obs_source_info sight_unified_tracking_filter_info;
extern struct obs_source_info sight_privacy_v2_filter_info;
#ifdef SIGHT_HAVE_OCR
extern struct obs_source_info sight_text_obfuscation_filter_info;
#endif
#ifdef SIGHT_HAVE_DENSE_FLOW
extern struct obs_source_info sight_spatial_motion_filter_info;
#endif
extern struct obs_source_info sight_tracking_receiver_filter_info;
extern struct obs_source_info sight_value_router_filter_info;
extern struct obs_source_info sight_native_vrm_source_info;
#ifdef SIGHT_HAVE_LIVE2D
extern struct obs_source_info sight_native_live2d_source_info;
#endif
#endif

#ifdef ENABLE_DOCK_PANEL
#include <QPointer>
static QPointer<AIStyleDock> styleDock;

static void on_frontend_finished_loading(enum obs_frontend_event event, void *data)
{
    UNUSED_PARAMETER(data);
    
    if (event == OBS_FRONTEND_EVENT_FINISHED_LOADING) {
        QMainWindow *mainWindow = (QMainWindow *)obs_frontend_get_main_window();
        if (mainWindow) {
            styleDock = new AIStyleDock(mainWindow);
            mainWindow->addDockWidget(Qt::RightDockWidgetArea, styleDock);
            
            // Make it floating initially for a nicer first impression
            styleDock->setFloating(false);
            styleDock->show();
            
            blog(LOG_INFO, "[Live Latent Lab: Sight] AI Style dock panel created");
        }
    } else if (event == OBS_FRONTEND_EVENT_THEME_CHANGED && styleDock) {
        styleDock->refreshTheme();
    }
}
#endif

bool obs_module_load(void)
{
    // Route processor diagnostics (shared with standalone test binaries)
    // into the OBS log instead of lost stdout printf.
    sight_log_set_sink([](int level, const char *msg) { blog(level, "%s", msg); });
#ifdef SIGHT_ANIMEGAN_ONLY
    // Build identity must appear in OBS logs before testing an effect.
    // Unambiguous marker to distinguish this corrected filter in OBS logs.
    blog(LOG_INFO, "[Live Latent Lab: Sight] ACTIVE_SKETCH_CUTOUT_DEPTH_TASKS_SIGNALS_HAND_TOUCH_EVENTS_20261010");
    obs_register_source(&neural_enhance_filter_info);
    obs_register_source(&sight_u2net_filter_info);
    obs_register_source(&sight_depth_filter_info);
    obs_register_source(&sight_unified_tracking_filter_info);
    blog(LOG_INFO,"[Sight] SIGHT_UNIFIED_TRACKING_20261118_U653");
    obs_register_source(&sight_privacy_v2_filter_info);
#ifdef SIGHT_HAVE_OCR
    obs_register_source(&sight_text_obfuscation_filter_info);
#endif
#ifdef SIGHT_HAVE_DENSE_FLOW
    obs_register_source(&sight_spatial_motion_filter_info);
#endif
    obs_register_source(&sight_tracking_receiver_filter_info);
    obs_register_source(&sight_value_router_filter_info);
    obs_register_source(&sight_native_vrm_source_info);
#ifdef SIGHT_HAVE_LIVE2D
    obs_register_source(&sight_native_live2d_source_info);
    blog(LOG_INFO,"[Sight] SIGHT_NATIVE_LIVE2D_SOURCE_20261118");
#endif
#else
    blog(LOG_INFO, "DEBUG_ACTIVE_XYZ123_V3_PLUGIN_LOADING");
    obs_register_source(&style_transfer_filter_info);
    obs_register_source(&image_gen_source_info); // Register new source
    obs_register_source(&neural_enhance_filter_info); // Register neural enhance filter
    obs_register_source(&latent_codec_test_filter_info); // Register latent codec test filter
    obs_register_source(&privacy_filter_info); // Register privacy filter
    obs_register_source(&parallax_filter_info); // Register parallax filter
    obs_register_source(&mediapipe_filter_info); // Register interactive segment filter
#endif
    
#ifndef SIGHT_ANIMEGAN_ONLY
    // Experimental models require the legacy checkpoint directories.
    char *config_path = obs_module_get_config_path(obs_current_module(), "");
    if (!config_path) {
        blog(LOG_ERROR, "[Live Latent Lab: Sight] Failed to get config path, cannot create model directories");
        return false;
    }
    std::string base_path = config_path;
    bfree(config_path);
    
    std::string models_path = base_path + "/models";
    MKDIR(models_path.c_str());
    MKDIR((models_path + "/checkpoints").c_str());
    MKDIR((models_path + "/loras").c_str());
    MKDIR((models_path + "/controlnet").c_str());
    MKDIR((models_path + "/vae").c_str());
    MKDIR((models_path + "/taesd").c_str());
    
    blog(LOG_INFO, "[Live Latent Lab: Sight] Ensure model directories exist at: %s", models_path.c_str());
#endif

#ifdef ENABLE_DOCK_PANEL
    obs_frontend_add_event_callback(on_frontend_finished_loading, nullptr);
    
    // Debug: Check Qt Core Instance
    if (QCoreApplication::instance()) {
        blog(LOG_INFO, "[Live Latent Lab: Sight] QCoreApplication exists at address: %p", QCoreApplication::instance());
    } else {
        blog(LOG_WARNING, "[Live Latent Lab: Sight] CRITICAL: QCoreApplication::instance() is NULL during load! Plugin might link against wrong Qt version.");
    }
    
    blog(LOG_INFO, "[Live Latent Lab: Sight] Plugin loaded with dock panel support");
#else
    blog(LOG_INFO, "[Live Latent Lab: Sight] Plugin loaded (dock panel disabled)");
#endif
    
    return true;
}

void obs_module_unload(void)
{
#ifdef ENABLE_DOCK_PANEL
    obs_frontend_remove_event_callback(on_frontend_finished_loading, nullptr);
    if (styleDock) {
        styleDock->deleteLater();
        styleDock = nullptr;
    }
#endif
    blog(LOG_INFO, "[Live Latent Lab: Sight] Plugin unloaded");
}

const char *obs_module_name(void)
{
    return "Live Latent Lab: Sight";
}

const char *obs_module_description(void)
{
    return "Real-time AI visual suite for OBS Studio";
}
