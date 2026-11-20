#include "live2d/model_package.hpp"
#include "live2d/face_parameters.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace sight::live2d;
namespace fs=std::filesystem;
int main(int argc,char** argv){
    if(argc==2){
        Package pkg;std::string error;
        if(!inspect_package(argv[1],pkg,error)){std::cerr<<error<<'\n';return 1;}
        std::cout<<"LIVE2D_PACKAGE_INTAKE textures="<<pkg.textures.size()
                 <<" motions="<<pkg.motions.size()<<" expressions="<<pkg.expressions.size()
                 <<" lip_ids="<<pkg.lip_ids.size()<<" blink_ids="<<pkg.blink_ids.size()<<'\n';
        return 0;
    }
    char temp[]="/tmp/sight-live2d-intake-XXXXXX";
    assert(mkdtemp(temp));fs::path root(temp);
    struct Cleanup{fs::path p;~Cleanup(){fs::remove_all(p);}} cleanup{root};
    fs::create_directories(root/"avatar/textures");
    auto file=root/"avatar"/"test.model3.json";
    std::ofstream(root/"avatar/model.moc3",std::ios::binary)<<"MOC3fixture";
    std::ofstream(root/"avatar/textures/atlas.png",std::ios::binary)<<"texture fixture";
    std::ofstream(root/"avatar/idle.motion3.json")<<"{}";
    std::ofstream(root/"avatar/happy.exp3.json")<<"{}";
    std::ofstream(root/"outside.png")<<"not inside model directory";
    auto write=[&](std::string_view s){std::ofstream(file)<<s;};
    write(R"({"Version":3,"FileReferences":{"Moc":"model.moc3","Textures":["textures/atlas.png"],"Motions":{"Idle":[{"File":"idle.motion3.json","FadeInTime":0.5}]},"Expressions":[{"Name":"Happy","File":"happy.exp3.json"}]},"Groups":[{"Target":"Parameter","Name":"LipSync","Ids":["ParamA"]},{"Target":"Parameter","Name":"EyeBlink","Ids":["ParamEyeLOpen","ParamEyeROpen"]}]})");
    Package pkg;std::string error;
    assert(inspect_package(file,pkg,error));
    assert(pkg.textures.size()==1&&pkg.motions.size()==1&&pkg.expressions.size()==1&&pkg.lip_ids==std::vector<std::string>{"ParamA"});
    assert(pkg.motions[0].group=="Idle"&&pkg.motions[0].fade_in==.5f);
    assert(pkg.blink_ids.size()==2&&pkg.moc==fs::canonical(root/"avatar/model.moc3"));
    write(R"({"Version":3,"FileReferences":{"Moc":"model.moc3","Textures":["../outside.png"]}})");
    assert(!inspect_package(file,pkg,error)&&pkg.moc.empty()&&!error.empty());
    fs::create_symlink(root/"outside.png",root/"avatar/escape.png");
    write(R"({"Version":3,"FileReferences":{"Moc":"model.moc3","Textures":["escape.png"]}})");
    assert(!inspect_package(file,pkg,error));
    write(R"({"Version":3,"FileReferences":{"Moc":"model.moc3","Textures":["missing.png"]}})");
    assert(!inspect_package(file,pkg,error));
    write(R"({"Version":2,"FileReferences":{"Moc":"model.moc3","Textures":["textures/atlas.png"]}})");
    assert(!inspect_package(file,pkg,error));
    write(R"({"Version":3,"FileReferences":{"Moc":"model.moc3","Textures":["textures/atlas.png"]}})");
    std::ofstream(root/"avatar/model.moc3",std::ios::binary|std::ios::trunc)<<"not moc";
    assert(!inspect_package(file,pkg,error));
    sight::TrackingSignal face;face.kind=sight::TrackingKind::Face;
    face.data.landmarks.resize(478);
    face.data.blendshape_names={"eyeBlinkRight","jawOpen","mouthSmileLeft","eyeBlinkLeft","mouthSmileRight","eyeLookOutLeft","eyeLookInRight"};
    face.data.blendshapes={.25f,.8f,.6f,.75f,.4f,.4f,.4f};
    const std::vector<Parameter> rig={{"ParamEyeLOpen",0,1,1},{"ParamEyeROpen",0,1,1},
        {"ParamA",0,0,2},{"ParamMouthForm",-1,0,1},{"ParamEyeBallX",-1,0,1},
        {"ParamAngleX",-30,0,30},{"ParamAngleY",-20,0,20},{"ParamAngleZ",-10,0,10},
        {"Unrelated",0,0,1}};
    auto values=face_parameters(&face,rig,{"ParamA","Nonexistent"});
    auto close=[](float a,float b){return std::abs(a-b)<.001f;};
    assert(close(values.at("ParamEyeLOpen"),.25)&&close(values.at("ParamEyeROpen"),.75));
    assert(close(values.at("ParamA"),1.6)&&close(values.at("ParamMouthForm"),.5));
    assert(close(values.at("ParamEyeBallX"),.4));
    assert(!values.contains("Nonexistent")&&!values.contains("Unrelated")&&!values.contains("ParamAngleX"));
    const std::vector<Parameter> old_rig={{"PARAM_EYE_L_OPEN",0,1,1},
        {"PARAM_EYE_R_OPEN",0,1,1},{"PARAM_MOUTH_OPEN_Y",0,0,1},{"PARAM_MOUTH_FORM",-1,0,1}};
    auto legacy=face_parameters(&face,old_rig);
    assert(close(legacy.at("PARAM_EYE_L_OPEN"),.25f)&&close(legacy.at("PARAM_EYE_R_OPEN"),.75f));
    assert(close(legacy.at("PARAM_MOUTH_OPEN_Y"),.8f)&&close(legacy.at("PARAM_MOUTH_FORM"),.5f));
    face.data.has_head_transform=true;
    face.data.head_transform={.8660254f,0,.5f,0, 0,1,0,0, -.5f,0,.8660254f,0, 0,0,0,1};
    values=face_parameters(&face,rig,{"ParamA"});
    assert(close(values.at("ParamAngleX"),-30)&&close(values.at("ParamAngleY"),0));
    face.data.blendshapes[1]=NAN;face.data.head_transform[0]=NAN;
    values=face_parameters(&face,rig,{"ParamA"});
    assert(!values.contains("ParamA")&&!values.contains("ParamAngleX"));
    face.data.landmarks.clear();assert(face_parameters(&face,rig,{"ParamA"}).empty());
    assert(face_parameters(nullptr,rig).empty());
    std::cout<<"LIVE2D_INTAKE_TEST manifest=1 sandbox=1 missing=1 version=1 moc_header=1 named_tracking=1 bounds=1 missing_values=1\n";
}
