#include <obs.h>
#include <obs-module.h>
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QDir>
#include <QSettings>
#include <cstdio>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed: %s\n",#x);return 1;}}while(false)
int main(int argc,char**argv){
 QCoreApplication app(argc,argv);CHECK(argc==3);
 QTemporaryDir tmp;CHECK(tmp.isValid());
 CHECK(QDir(tmp.path()).mkdir("models"));CHECK(QDir(tmp.path()).mkdir("runtime"));
 CHECK(obs_startup("en-US",tmp.path().toUtf8().constData(),nullptr));
 obs_module_t*m=nullptr;CHECK(obs_open_module(&m,argv[1],argv[2])==MODULE_SUCCESS&&obs_init_module(m));
 auto*baseline=obs_get_source_defaults("sight_unified_tracking_filter");CHECK(baseline);
 QString selected=QString::fromUtf8(obs_data_get_string(baseline,"face_model_path"));obs_data_release(baseline);
 char*path=obs_module_get_config_path(m,"preferences.ini");CHECK(path);
 {QSettings s(QString::fromUtf8(path),QSettings::IniFormat);s.setValue("models",tmp.filePath("models"));s.setValue("runtimes",tmp.filePath("runtime"));s.sync();CHECK(s.status()==QSettings::NoError);}
 bfree(path);
 for(auto id:{"sight_depth_filter","sight_u2net_filter","neural_enhance_filter","sight_native_vrm_source"}){
   auto*p=obs_get_source_properties(id);CHECK(p);
   CHECK(QString::fromUtf8(obs_property_path_default_path(obs_properties_get(p,"model_path")))==tmp.filePath("models"));
   obs_properties_destroy(p);
 }
 auto*p=obs_get_source_properties("sight_unified_tracking_filter");CHECK(p);
 CHECK(QString::fromUtf8(obs_property_path_default_path(obs_properties_get(p,"face_model_path")))==tmp.filePath("models"));
 CHECK(QString::fromUtf8(obs_property_path_default_path(obs_properties_get(p,"face_mediapipe_runtime")))==tmp.filePath("runtime"));
 obs_properties_destroy(p);
 auto*after=obs_get_source_defaults("sight_unified_tracking_filter");CHECK(after);
 CHECK(QString::fromUtf8(obs_data_get_string(after,"face_model_path"))==selected);
 obs_data_release(after);obs_shutdown();
 puts("SIGHT_SETTINGS_NATIVE_PICKER_PASS=1");
}
