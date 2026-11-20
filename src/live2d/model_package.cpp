#include "model_package.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
namespace sight::live2d {
namespace fs=std::filesystem;
using Json=nlohmann::json;
namespace {
constexpr uintmax_t MiB=1024*1024;
constexpr uintmax_t MaxManifest=2*MiB,MaxFile=128*MiB,MaxTotal=512*MiB;
constexpr size_t MaxReferences=512;
using Error=std::runtime_error;
std::string nonempty(const Json& j){
    auto s=j.get<std::string>();
    if(s.empty()||s.size()>1024||s.find('\0')!=std::string::npos)throw Error("Invalid empty or oversized model field");
    return s;
}
struct References {
    fs::path root;
    std::set<fs::path> seen;
    uintmax_t total=0;
    fs::path get(const Json& j){
        auto name=nonempty(j);
        // JSON model references are local relative paths. Never resolve URI,
        // absolute, drive-qualified, or parent traversal inputs.
        if(name.find(':')!=std::string::npos||name.find('\\')!=std::string::npos)
            throw Error("Unsupported model reference: "+name);
        fs::path rel(name);
        if(rel.is_absolute()||rel.has_root_path())throw Error("Absolute model reference");
        for(const auto& part:rel)if(part=="..")throw Error("Parent traversal in model reference");
        auto path=fs::canonical(root/rel);
        auto within=path.lexically_relative(root);
        if(within.empty()||within=="."||within.is_absolute()||*within.begin()=="..")
            throw Error("Model reference escapes its directory");
        if(!fs::is_regular_file(path))throw Error("Model reference is not a regular file");
        if(seen.insert(path).second){
            auto size=fs::file_size(path);
            if(!size||size>MaxFile||size>MaxTotal-total||seen.size()>MaxReferences)
                throw Error("Model package exceeds resource limits");
            total+=size;
        }
        return path;
    }
};
float fade(const Json& value,const char* key){
    if(!value.contains(key))return -1;
    auto v=value.at(key).get<float>();
    if(!std::isfinite(v)||v<0||v>60)throw Error("Invalid motion fade time");
    return v;
}
void ids(const Json& input,std::vector<std::string>& out){
    if(!input.is_array()||input.size()>128)throw Error("Invalid parameter group");
    for(const auto& entry:input){
        auto id=nonempty(entry);
        if(std::find(out.begin(),out.end(),id)==out.end())out.push_back(std::move(id));
        if(out.size()>128)throw Error("Too many parameter IDs");
    }
}
}
bool inspect_package(const fs::path& input,Package& out,std::string& error){
    out={};error.clear();
    try{
        Package pkg;
        pkg.manifest=fs::canonical(input);
        if(!fs::is_regular_file(pkg.manifest)||!pkg.manifest.filename().string().ends_with(".model3.json"))
            throw Error("Select a local .model3.json file");
        const auto size=fs::file_size(pkg.manifest);
        if(!size||size>MaxManifest)throw Error("Invalid model manifest size");
        std::ifstream stream(pkg.manifest,std::ios::binary);
        std::string bytes(size,'\0');
        if(!stream.read(bytes.data(),std::streamsize(bytes.size())))throw Error("Cannot read model manifest");
        auto json=Json::parse(bytes);
        if(!json.is_object()||!json.at("Version").is_number_integer()||json.at("Version")!=3)
            throw Error("Only model3 manifest version 3 is supported");
        const auto& refs=json.at("FileReferences");
        if(!refs.is_object())throw Error("Invalid FileReferences");
        References paths{pkg.manifest.parent_path(),{},0};
        pkg.moc=paths.get(refs.at("Moc"));
        char magic[4]{};
        std::ifstream moc(pkg.moc,std::ios::binary);
        if(!moc.read(magic,4)||std::string_view(magic,4)!="MOC3")throw Error("Missing MOC3 signature");
        const auto& tex=refs.at("Textures");
        if(!tex.is_array()||tex.empty()||tex.size()>64)throw Error("Invalid texture list");
        for(const auto& entry:tex)pkg.textures.push_back(paths.get(entry));
        auto optional=[&](const char* key,fs::path& dest){
            if(refs.contains(key))dest=paths.get(refs.at(key));
        };
        optional("Physics",pkg.physics);optional("Pose",pkg.pose);
        fs::path unused;optional("DisplayInfo",unused);optional("UserData",unused);
        if(refs.contains("Expressions")){
            auto& list=refs.at("Expressions");
            if(!list.is_array()||list.size()>256)throw Error("Invalid expression list");
            std::set<std::string> names;
            for(auto& expr:list){
                auto name=nonempty(expr.at("Name"));
                if(!names.insert(name).second)throw Error("Duplicate expression name");
                pkg.expressions.push_back({std::move(name),paths.get(expr.at("File"))});
            }
        }
        if(refs.contains("Motions")){
            auto& groups=refs.at("Motions");
            if(!groups.is_object()||groups.size()>128)throw Error("Invalid motion groups");
            for(auto it=groups.begin();it!=groups.end();++it){
                if(it.key().empty()||it.key().size()>1024||!it.value().is_array())throw Error("Invalid motion group");
                for(auto& motion:it.value()){
                    if(pkg.motions.size()>=MaxReferences)throw Error("Too many motions");
                    MotionFile entry{it.key(),paths.get(motion.at("File")),{},fade(motion,"FadeInTime"),fade(motion,"FadeOutTime")};
                    if(motion.contains("Sound")&&motion.at("Sound")!="")entry.sound=paths.get(motion.at("Sound"));
                    pkg.motions.push_back(std::move(entry));
                }
            }
        }
        if(json.contains("Groups")){
            auto& groups=json.at("Groups");
            if(!groups.is_array()||groups.size()>128)throw Error("Invalid model groups");
            for(auto& group:groups){
                if(group.at("Target")!="Parameter")continue;
                if(group.at("Name")=="EyeBlink")ids(group.at("Ids"),pkg.blink_ids);
                else if(group.at("Name")=="LipSync")ids(group.at("Ids"),pkg.lip_ids);
            }
        }
        out=std::move(pkg);
        return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
}
