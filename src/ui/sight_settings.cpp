#include "plugin-settings-page.hpp"
#include "../common/preferences.hpp"
#include <obs-module.h>
#include <util/bmem.h>
#include <QSettings>
#include <QDir>
#include <QLabel>
#include <QLineEdit>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QScrollArea>
#include <QDesktopServices>
#include <QUrl>
#include <QFileInfo>

namespace {
std::unique_ptr<plugin_settings::Section> section;
QString configPath(){
    char*p=obs_module_config_path("preferences.ini");
    if(!p)return {};
    auto path=QString::fromUtf8(p);bfree(p);return path;
}
QString pref(const char*key){
    auto path=configPath();if(path.isEmpty())return {};
    return QSettings(path,QSettings::IniFormat).value(key).toString();
}
void link(QVBoxLayout*layout,QWidget*parent,const QString&title,const QString&url){
    auto*label=new QLabel(parent);label->setTextFormat(Qt::RichText);
    label->setText("<a href=\""+url.toHtmlEscaped()+"\">"+title.toHtmlEscaped()+"</a>");
    label->setTextInteractionFlags(Qt::TextBrowserInteraction);
    QObject::connect(label,&QLabel::linkActivated,label,[url](const QString&){QDesktopServices::openUrl(QUrl(url));});
    layout->addWidget(label);
}
const QString docs="https://github.com/myrqyry/live-latent-lab.sight/blob/main/docs/";
QWidget* makePage(QWidget*parent){
    auto*page=new QWidget(parent);
    auto*root=new QVBoxLayout(page);root->setContentsMargins(8,6,8,6);
    auto*scroll=new QScrollArea(page);scroll->setWidgetResizable(true);scroll->setFrameShape(QFrame::NoFrame);root->addWidget(scroll);
    auto*body=new QWidget(scroll);auto*layout=new QVBoxLayout(body);scroll->setWidget(body);
    auto*heading=new QLabel("Sight · Models and runtimes",body);auto font=heading->font();font.setBold(true);heading->setFont(font);layout->addWidget(heading);
    auto*hint=new QLabel("Choose starting folders for model and runtime file pickers. Existing sources keep their own settings. Downloads open in your browser.",body);
    hint->setWordWrap(true);layout->addWidget(hint);
    auto*form=new QFormLayout;layout->addLayout(form);
    auto picker=[&](const char*key,const QString&title){
        auto*row=new QWidget(body);auto*h=new QHBoxLayout(row);h->setContentsMargins(0,0,0,0);
        auto*edit=new QLineEdit(pref(key),row);edit->setObjectName(key);edit->setReadOnly(true);edit->setPlaceholderText("System default");
        auto*browse=new QPushButton("Browse…",row);auto*clear=new QPushButton("Clear",row);
        h->addWidget(edit,1);h->addWidget(browse);h->addWidget(clear);form->addRow(title,row);
        QObject::connect(browse,&QPushButton::clicked,row,[page,edit]{auto dir=QFileDialog::getExistingDirectory(page,"Choose folder",edit->text());if(!dir.isEmpty())edit->setText(dir);});
        QObject::connect(clear,&QPushButton::clicked,edit,&QLineEdit::clear);
        QObject::connect(edit,&QLineEdit::textChanged,page,[page]{if(section)section->markDirty(page);});
    };
    picker("models","Model library");picker("runtimes","Runtime library");
    auto*note=new QLabel("Changes apply to newly opened file pickers. Use OBS Apply or OK to save.",body);note->setWordWrap(true);note->setObjectName("sight-preferences-status");layout->addWidget(note);
    link(layout,body,"Model compatibility and licensing guide",docs+"model-downloads.md");
    link(layout,body,"Portrait Sketch · AnimeGANv3 ONNX (usage restrictions apply)","https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch");
    link(layout,body,"Subject Cutout · U²-Net","https://github.com/danielgatis/rembg/releases/tag/v0.0.0");
    link(layout,body,"Depth Camera · Depth Anything V2 Small FP32","https://huggingface.co/onnx-community/depth-anything-v2-small/blob/main/onnx/model.onnx");
    link(layout,body,"Face · MediaPipe Face Landmarker","https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker");
    link(layout,body,"Body · MediaPipe Pose Landmarker","https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker");
    link(layout,body,"Hands · MediaPipe Gesture Recognizer","https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer");
    link(layout,body,"Holistic and object tracking · qualified setup",docs+"model-downloads.md");
    link(layout,body,"MediaPipe 1.2.3 runtime · ABI and checksums",docs+"mediapipe-1.2.3.md");
    link(layout,body,"MediaPipe 1.2.3 · PyPI","https://pypi.org/project/mediapipe/1.2.3/");
#ifdef SIGHT_HAVE_RTMW
    link(layout,body,"WholeBody (RTMW GPU) · setup and qualification","https://github.com/myrqyry/live-latent-lab.sight#optional-gpu-tracking");
#endif
#ifdef SIGHT_HAVE_OCR
    link(layout,body,"OCR · Tesseract language models","https://github.com/tesseract-ocr/tessdata");
#endif
    link(layout,body,"VRM avatar · interoperability guide",docs+"vrm-interoperability.md");
    link(layout,body,"ONNX Runtime · pinned dependency",docs+"onnx-runtime.md");
    layout->addStretch();return page;
}
bool savePage(QWidget*page){
    auto path=configPath();if(path.isEmpty())return false;
    QSettings settings(path,QSettings::IniFormat);
    for(auto key:{"models","runtimes"})settings.setValue(key,page->findChild<QLineEdit*>(key)->text());
    settings.sync();
    bool ok=settings.status()==QSettings::NoError;
    page->findChild<QLabel*>("sight-preferences-status")->setText(ok?"Sight preferences saved.":"Could not save Sight preferences.");
    return ok;
}
}
namespace sight_preferences {
std::string directory(bool runtime){
    auto dir=pref(runtime?"runtimes":"models");
    return QFileInfo(dir).isDir()&&!dir.isEmpty()?QDir::cleanPath(dir).toUtf8().toStdString():std::string{};
}
}
void register_sight_settings(){
    if(section||!qobject_cast<QApplication*>(QCoreApplication::instance()))return;
    section=std::make_unique<plugin_settings::Section>("lll.sight","LLL.sight",makePage,savePage);
    section->scan();
    blog(LOG_INFO,"[Sight] Native plugin settings bridge registered");
}
void unregister_sight_settings(){section.reset();}
