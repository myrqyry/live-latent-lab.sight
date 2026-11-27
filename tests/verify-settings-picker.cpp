#include <obs.h>
#include <obs-module.h>
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QDir>
#include <QSettings>
#include <QFile>
#include <cstdio>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed: %s\n",#x);return 1;}}while(false)
int main(int argc,char**argv){
 QCoreApplication app(argc,argv);CHECK(argc==3);
 QTemporaryDir tmp;CHECK(tmp.isValid());
 CHECK(QDir(tmp.path()).mkdir("models"));CHECK(QDir(tmp.path()).mkdir("runtime"));
 qputenv("XDG_CONFIG_HOME",tmp.path().toUtf8());
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
 // The full development build has additional model consumers.
 auto checkPath=[&](const char*id,const char*key){
   auto*props=obs_get_source_properties(id);
   if(!props){fprintf(stdout,"optional source not registered: %s\n",id);return true;}
   auto*property=obs_properties_get(props,key);
   bool ok=property&&QString::fromUtf8(obs_property_path_default_path(property))==tmp.filePath("models");
   obs_properties_destroy(props);fprintf(stdout,"picker: %s/%s = %s\n",id,key,ok?"PASS":"FAIL");return ok;
 };
 CHECK(checkPath("sight_preprocessors_filter","depth_model_path"));
 CHECK(checkPath("sight_preprocessors_filter","teed_model_path"));
 CHECK(checkPath("sight_privacy_v2_filter","semantic_model"));
 CHECK(checkPath("sight_native_sd_live_filter","model_path"));
 CHECK(checkPath("ai_image_generator","model_path"));
 CHECK(checkPath("sight_native_live2d_source","model_path"));
 // Actual libobs property modification callbacks, not inference or file picker mocks.
 auto testDiagnostic=[&](const char*id,const char*key,const QString&file,const char*message,obs_text_info_type type){
   auto*props=obs_get_source_properties(id);
   if(!props)return false;
   const auto diagnostic=std::string(key)+"_diagnostic";
   auto*field=obs_properties_get(props,diagnostic.c_str());
   auto*settings=obs_get_source_defaults(id);
   if(!settings||!field){if(settings)obs_data_release(settings);obs_properties_destroy(props);return false;}
   obs_data_set_string(settings,key,file.toUtf8().constData());
   obs_properties_apply_settings(props,settings);
   bool ok=QString::fromUtf8(obs_property_description(field)).contains(QString::fromUtf8(message))&&obs_property_text_info_type(field)==type;
   if(!ok)fprintf(stderr,"diagnostic %s/%s: %s\n",id,key,obs_property_description(field));
   obs_data_release(settings);obs_properties_destroy(props);return ok;
 };
 const auto model=tmp.filePath("model.onnx");
 const auto wrong=tmp.filePath("wrong.txt");
 const auto tflite=tmp.filePath("invalid.tflite");
 const auto runtime=tmp.filePath("invalid.so");
 {QFile f(model);CHECK(f.open(QIODevice::WriteOnly));}
 for(const auto&path:{wrong,tflite,runtime}){QFile f(path);CHECK(f.open(QIODevice::WriteOnly));CHECK(f.write("not a model",11)==11);}
 for(auto id:{"sight_depth_filter","sight_u2net_filter","neural_enhance_filter"}){
   CHECK(testDiagnostic(id,"model_path",{},"No model selected",OBS_TEXT_INFO_WARNING));
   CHECK(testDiagnostic(id,"model_path",tmp.filePath("missing.onnx"),"not found",OBS_TEXT_INFO_ERROR));
   CHECK(testDiagnostic(id,"model_path",tmp.path(),"not a regular file",OBS_TEXT_INFO_ERROR));
   CHECK(testDiagnostic(id,"model_path",model,"empty",OBS_TEXT_INFO_ERROR));
   CHECK(testDiagnostic(id,"model_path",wrong,"Expected",OBS_TEXT_INFO_WARNING));
 }
 {QFile f(model);CHECK(f.open(QIODevice::WriteOnly));CHECK(f.write("not a valid ONNX graph",22)==22);}
 CHECK(testDiagnostic("sight_depth_filter","model_path",model,"not yet verified",OBS_TEXT_INFO_NORMAL));
 CHECK(testDiagnostic("sight_unified_tracking_filter","face_model_path",tmp.filePath("missing.task"),"not found",OBS_TEXT_INFO_ERROR));
 CHECK(testDiagnostic("sight_unified_tracking_filter","object_model_path",tflite,"TFLite identifier",OBS_TEXT_INFO_ERROR));
 CHECK(testDiagnostic("sight_unified_tracking_filter","face_mediapipe_runtime",runtime,"ELF",OBS_TEXT_INFO_ERROR));
 CHECK(testDiagnostic("sight_unified_tracking_filter","face_mediapipe_runtime",{},"No runtime selected",OBS_TEXT_INFO_WARNING));
 // A same-property change updates the existing diagnostic, and does not
 // request a needless Qt refresh when the result is unchanged.
 auto*depthProps=obs_get_source_properties("sight_depth_filter");
 auto*depthSettings=obs_get_source_defaults("sight_depth_filter");
 auto*depthPath=obs_properties_get(depthProps,"model_path");
 auto*depthDiagnostic=obs_properties_get(depthProps,"model_path_diagnostic");
 CHECK(depthPath&&depthDiagnostic);
 obs_data_set_string(depthSettings,"model_path",model.toUtf8().constData());
 CHECK(obs_property_modified(depthPath,depthSettings));
 CHECK(!obs_property_modified(depthPath,depthSettings));
 obs_data_set_string(depthSettings,"model_path",tmp.filePath("missing.onnx").toUtf8().constData());
 CHECK(obs_property_modified(depthPath,depthSettings));
 CHECK(obs_property_text_info_type(depthDiagnostic)==OBS_TEXT_INFO_ERROR);
 obs_data_release(depthSettings);obs_properties_destroy(depthProps);
 // Sparse synthetic ELF fixtures exercise the existing size gate without
 // ever loading or pretending to validate the synthetic shared library.
 const auto legacy=tmp.filePath("legacy.so");
 const auto official=tmp.filePath("official.so");
 for(const auto&pair:{qMakePair(legacy,qint64(33886744)),qMakePair(official,qint64(127908512))}){
   QFile f(pair.first);CHECK(f.open(QIODevice::WriteOnly));CHECK(f.write("\x7f" "ELF",4)==4);CHECK(f.resize(pair.second));
 }
 CHECK(testDiagnostic("sight_unified_tracking_filter","face_mediapipe_runtime",legacy,"not yet verified",OBS_TEXT_INFO_NORMAL));
 CHECK(testDiagnostic("sight_unified_tracking_filter","holistic_mediapipe_runtime",legacy,"requires MediaPipe 1.2.3",OBS_TEXT_INFO_ERROR));
 CHECK(testDiagnostic("sight_unified_tracking_filter","object_mediapipe_runtime",official,"not yet verified",OBS_TEXT_INFO_NORMAL));
 auto*trackingProps=obs_get_source_properties("sight_unified_tracking_filter");
 auto*trackingSettings=obs_get_source_defaults("sight_unified_tracking_filter");
 obs_data_set_string(trackingSettings,"tracking_mode","object");
 obs_properties_apply_settings(trackingProps,trackingSettings);
 CHECK(obs_property_visible(obs_properties_get(trackingProps,"object_model_path_diagnostic")));
 CHECK(!obs_property_visible(obs_properties_get(trackingProps,"face_model_path_diagnostic")));
 obs_data_release(trackingSettings);obs_properties_destroy(trackingProps);
 auto*after=obs_get_source_defaults("sight_unified_tracking_filter");CHECK(after);
 CHECK(QString::fromUtf8(obs_data_get_string(after,"face_model_path"))==selected);
 obs_data_release(after);obs_shutdown();
 puts("SIGHT_SETTINGS_NATIVE_PICKER_PASS=1");
}
