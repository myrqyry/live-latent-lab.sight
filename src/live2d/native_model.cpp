#include "native_model.hpp"
#include "vrm/vendor/stb_image.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <set>
#include <nlohmann/json.hpp>
namespace sight::live2d {
Core::~Core(){if(library)dlclose(library);}
bool Core::open(std::string& error){
    library=dlopen(SIGHT_CUBISM_CORE_PATH,RTLD_NOW|RTLD_LOCAL);
    if(!library){error=dlerror();return false;}
#define LOAD(n) n=reinterpret_cast<decltype(n)>(dlsym(library,#n));if(!n){error="Missing Cubism symbol " #n;return false;}
    SIGHT_CUBISM_API(LOAD)
#undef LOAD
    if(csmGetVersion()!=0x06000001u){error="Expected qualified Cubism Core 6.0.1 (Native SDK 5 R5)";return false;}
    return true;
}
namespace {
constexpr size_t MiB=1024*1024;
using Error=std::runtime_error;
std::vector<uint8_t> bytes(const std::filesystem::path& p,size_t limit){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    const auto n=f.tellg();
    if(n<=0||uintmax_t(n)>limit)throw Error("Unreadable or oversized resource: "+p.filename().string());
    std::vector<uint8_t> data(static_cast<size_t>(n));f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(data.data()),n))throw Error("Cannot read resource");
    return data;
}
void* aligned(std::vector<uint8_t>& memory,size_t n,size_t alignment){
    memory.resize(n+alignment-1);
    auto p=reinterpret_cast<uintptr_t>(memory.data());
    return reinterpret_cast<void*>((p+alignment-1)&~(uintptr_t(alignment)-1));
}
bool finite(csmVector2 p){return std::isfinite(p.X)&&std::isfinite(p.Y);}
}
std::unique_ptr<NativeModel> load_native(const std::filesystem::path& path,std::string& error){
    try{
        auto out=std::make_unique<NativeModel>();
        if(!inspect_package(path,out->package,error)||!out->core.open(error))return {};
        auto& c=out->core;
        auto moc=bytes(out->package.moc,128*MiB);
        if(moc.size()<64)throw Error("Truncated MOC3");
        auto* buffer=aligned(out->moc_memory,moc.size(),csmAlignofMoc);
        std::copy(moc.begin(),moc.end(),static_cast<uint8_t*>(buffer));
        auto version=c.csmGetMocVersion(buffer,unsigned(moc.size()));
        if(!version||version>csmMocVersion_53||!c.csmHasMocConsistency(buffer,unsigned(moc.size())))
            throw Error("Cubism Core rejected MOC3 consistency");
        auto* revived=c.csmReviveMocInPlace(buffer,unsigned(moc.size()));
        if(!revived)throw Error("Cubism Core could not revive model");
        unsigned size=c.csmGetSizeofModel(revived);
        if(!size||size>128*MiB)throw Error("Cubism model exceeds memory budget");
        out->model=c.csmInitializeModelInPlace(revived,aligned(out->model_memory,size,csmAlignofModel),size);
        auto* m=out->model;
        if(!m)throw Error("Cubism Core initialization failed");
        if(c.csmGetOffscreenCount(m)!=0)throw Error("This rig uses advanced Cubism offscreen compositing (not supported)");
        c.csmReadCanvasInfo(m,&out->canvas,&out->origin,&out->pixels_per_unit);
        if(!finite(out->canvas)||!finite(out->origin)||out->canvas.X<=0||out->canvas.Y<=0||
           !std::isfinite(out->pixels_per_unit)||out->pixels_per_unit<=0)
            throw Error("Invalid Cubism canvas");
        int count=c.csmGetParameterCount(m);
        if(count<0||count>4096)throw Error("Invalid parameter count");
        auto ids=c.csmGetParameterIds(m);auto min=c.csmGetParameterMinimumValues(m);
        auto max=c.csmGetParameterMaximumValues(m);auto neutral=c.csmGetParameterDefaultValues(m);
        for(int i=0;i<count;++i){
            if(!ids[i]||!std::isfinite(min[i])||!std::isfinite(max[i])||!std::isfinite(neutral[i])||
               min[i]>neutral[i]||neutral[i]>max[i])throw Error("Invalid parameter range");
            out->rig.push_back({ids[i],min[i],neutral[i],max[i]});out->current.push_back(neutral[i]);
            c.csmGetParameterValues(m)[i]=neutral[i];
        }
        const int part_count=c.csmGetPartCount(m);
        if(part_count<0||part_count>4096)throw Error("Invalid model part count");
        auto part_ids=c.csmGetPartIds(m);auto* opacities=c.csmGetPartOpacities(m);
        if(part_count)out->initial_part_opacities.assign(opacities,opacities+part_count);
        for(int i=0;i<part_count;++i)out->part_ids.emplace_back(part_ids[i]);
        size_t animation_segments=0;
        for(const auto& motion:out->package.motions){
            if(out->motions.contains(motion.group)||out->motions.size()>=32)continue;
            Motion decoded;std::string diagnostic;
            if(!read_motion(motion.file,decoded,diagnostic))continue;
            size_t segments=0;for(const auto& curve:decoded.curves)segments+=curve.segments.size()+1;
            if(segments>200000-animation_segments)continue;
            animation_segments+=segments;out->motions.emplace(motion.group,std::move(decoded));
        }
        size_t expression_values=0;
        for(const auto& expression:out->package.expressions){
            std::vector<ExpressionValue> decoded;std::string diagnostic;
            if(read_expression(expression.file,decoded,diagnostic)&&decoded.size()<=8192-expression_values){
                expression_values+=decoded.size();out->expressions.emplace(expression.name,std::move(decoded));
            }
        }
        if(!out->package.pose.empty()){
            auto raw=bytes(out->package.pose,1024*1024);
            auto pose=nlohmann::json::parse(raw);
            const auto& groups=pose.at("Groups");
            if(!groups.is_array()||groups.size()>256)throw Error("Invalid pose groups");
            auto part_index=[&](const std::string& id){
                for(int i=0;i<part_count;++i)if(id==part_ids[i])return i;
                return -1;
            };
            auto param_index=[&](const std::string& id){
                for(size_t i=0;i<out->rig.size();++i)if(id==out->rig[i].id)return int(i);
                return -1;
            };
            for(const auto& group:groups){
                if(!group.is_array()||group.empty()||group.size()>64)throw Error("Invalid pose group");
                std::vector<PosePart> entries;
                for(const auto& entry:group){
                    auto id=entry.at("Id").get<std::string>();int index=part_index(id);
                    if(index<0)continue;
                    PosePart part{id,index,param_index(id),{}};
                    if(entry.contains("Link")){
                        const auto& links=entry.at("Link");
                        if(!links.is_array()||links.size()>64)throw Error("Invalid pose links");
                        for(const auto& link:links){int linked=part_index(link.get<std::string>());if(linked>=0)part.links.push_back(linked);}
                    }
                    entries.push_back(std::move(part));
                }
                if(!entries.empty())out->pose_groups.push_back(std::move(entries));
            }
        }
        out->apply_pose();
        c.csmUpdateModel(m);
        count=c.csmGetDrawableCount(m);
        if(count<=0||count>1024)throw Error("Unsupported drawable count");
        auto textures=c.csmGetDrawableTextureIndices(m);auto blends=c.csmGetDrawableBlendModes(m);
        auto flags=c.csmGetDrawableConstantFlags(m);auto vc=c.csmGetDrawableVertexCounts(m);
        auto uv=c.csmGetDrawableVertexUvs(m);auto ic=c.csmGetDrawableIndexCounts(m);
        auto idx=c.csmGetDrawableIndices(m);auto mc=c.csmGetDrawableMaskCounts(m);
        auto masks=c.csmGetDrawableMasks(m);auto positions=c.csmGetDrawableVertexPositions(m);
        size_t total=0;
        for(int i=0;i<count;++i){
            Drawable d;
            if(textures[i]<0||size_t(textures[i])>=out->package.textures.size()||
               blends[i]<0||blends[i]>2||vc[i]<0||vc[i]>65535||ic[i]<0||ic[i]%3||
               ic[i]>250000||mc[i]<0||mc[i]>count)
                throw Error("Unsupported drawable, texture, or blend mode");
            total+=size_t(ic[i]);if(total>500000)throw Error("Drawable geometry exceeds budget");
            d.texture=textures[i];d.blend=blends[i];d.inverted=(flags[i]&csmIsInvertedMask)!=0;
            for(int j=0;j<vc[i];++j)if(!finite(uv[i][j])||!finite(positions[i][j]))throw Error("Nonfinite mesh");
            if(vc[i])d.uv.assign(uv[i],uv[i]+vc[i]);
            for(int j=0;j<ic[i];++j){if(idx[i][j]>=vc[i])throw Error("Index outside mesh");d.indices.push_back(idx[i][j]);}
            if(mc[i]){
                std::vector<int> group;
                for(int j=0;j<mc[i];++j){if(masks[i][j]<0||masks[i][j]>=count)throw Error("Invalid clipping index");group.push_back(masks[i][j]);}
                std::sort(group.begin(),group.end());group.erase(std::unique(group.begin(),group.end()),group.end());
                auto it=std::find(out->masks.begin(),out->masks.end(),group);
                if(it==out->masks.end()){if(out->masks.size()>=32)throw Error("Too many distinct clipping masks");d.mask=int(out->masks.size());out->masks.push_back(std::move(group));}
                else d.mask=int(it-out->masks.begin());
            }
            out->drawables.push_back(std::move(d));
        }
        size_t decoded=0;
        for(const auto& file:out->package.textures){
            auto raw=bytes(file,128*MiB);int w=0,h=0,n=0;
            if(!stbi_info_from_memory(raw.data(),int(raw.size()),&w,&h,&n)||w<=0||h<=0||w>8192||h>8192||
               size_t(w)*h*4>256*MiB-decoded)throw Error("Texture exceeds decoded image budget");
            std::unique_ptr<stbi_uc,decltype(&stbi_image_free)> rgba(stbi_load_from_memory(raw.data(),int(raw.size()),&w,&h,&n,4),stbi_image_free);
            if(!rgba)throw Error("Cubism texture decode failed");
            decoded+=size_t(w)*h*4;
            out->images.push_back({w,h,std::vector<uint8_t>(rgba.get(),rgba.get()+size_t(w)*h*4)});
        }
        return out;
    }catch(const std::exception& e){error=e.what();return {};}
}
void NativeModel::apply_pose(const Values& authored){
    auto* opacity=core.csmGetPartOpacities(model);
    auto* parameters=core.csmGetParameterValues(model);
    // Establish one mutually exclusive authored pose per group. For rigs
    // with explicit selector parameters, use those; otherwise choose the
    // first authored part. Do not force unrelated part alpha values.
    for(const auto& group:pose_groups){
        size_t selected=0;
        float best=.001f;
        for(size_t i=0;i<group.size();++i){
            const auto& p=group[i];
            const auto it=authored.find(p.id);
            const float score=it!=authored.end()?it->second:p.parameter>=0?parameters[p.parameter]:0.f;
            if(score>best){best=score;selected=i;}
        }
        for(size_t i=0;i<group.size();++i){
            const auto& p=group[i];const float value=i==selected?1.f:0.f;
            opacity[p.part]=value;
            for(int link:p.links)opacity[link]=value;
        }
    }
}
void NativeModel::animate(const TrackingSignal* face,float dt,bool idle,float ease_ms,const std::string& motion_group,const std::string& expression){
    dt=std::clamp(dt,0.f,.1f);elapsed+=dt;
    const auto values=face_parameters(face,rig,package.lip_ids);
    const std::string selected=idle?motion_group:"";
    if(selected!=active_motion){active_motion=selected;motion_elapsed=0;}
    motion_elapsed+=dt;
    Values authored,parts;
    if(auto it=motions.find(active_motion);it!=motions.end())it->second.sample(motion_elapsed,authored,parts);
    auto exp=expressions.find(expression);
    auto* opacity=core.csmGetPartOpacities(model);
    for(size_t i=0;i<part_ids.size();++i){
        const auto it=parts.find(part_ids[i]);
        opacity[i]=it!=parts.end()?std::clamp(it->second,0.f,1.f):initial_part_opacities[i];
    }
    const float a=ease_ms<=0?1.f:1.f-std::exp(-dt/(ease_ms*.001f));
    auto* target=core.csmGetParameterValues(model);
    const float blink=std::max(0.f,1.f-float(std::abs(std::fmod(elapsed,4.3)-2.5)/.12));
    for(size_t i=0;i<rig.size();++i){
        const auto& p=rig[i];const auto id=standard_parameter(p.id);float v=p.neutral;
        if(idle){
            if(id=="ParamBreath")v=p.neutral+float(std::sin(elapsed*1.5))*.15f*(p.maximum-p.minimum);
            else if(id=="ParamAngleZ")v=p.neutral+float(std::sin(elapsed*1.1))*2.f;
            else if(id=="ParamEyeLOpen"||id=="ParamEyeROpen"||
                    std::find(package.blink_ids.begin(),package.blink_ids.end(),p.id)!=package.blink_ids.end())
                v=p.neutral-blink*(p.neutral-p.minimum);
        }
        if(auto it=authored.find(p.id);it!=authored.end())v=it->second;
        if(exp!=expressions.end())for(const auto& e:exp->second){
            if(e.id!=p.id)continue;
            if(e.blend==ExpressionBlend::Add)v+=e.value;
            else if(e.blend==ExpressionBlend::Multiply)v*=e.value;
            else v=e.value;
        }
        if(auto it=values.find(p.id);it!=values.end())v=it->second;
        v=std::clamp(v,p.minimum,p.maximum);
        current[i]+=a*(v-current[i]);
        if(std::abs(v-current[i])<.0001f)current[i]=v;
        target[i]=current[i];
    }
    for(const auto& [id,value]:parts)authored[id]=value;
    apply_pose(authored);
    core.csmUpdateModel(model);
}
}
