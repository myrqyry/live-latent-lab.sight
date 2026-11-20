#pragma once
#include "model_package.hpp"
#include "face_parameters.hpp"
#include "animation.hpp"
#include <Live2DCubismCore.h>
#include <memory>
#include <vector>
namespace sight::live2d {
#define SIGHT_CUBISM_API(X) \
 X(csmGetVersion) X(csmGetMocVersion) X(csmHasMocConsistency) X(csmReviveMocInPlace) \
 X(csmGetSizeofModel) X(csmInitializeModelInPlace) X(csmUpdateModel) X(csmReadCanvasInfo) \
 X(csmGetParameterCount) X(csmGetParameterIds) X(csmGetParameterMinimumValues) \
 X(csmGetParameterDefaultValues) X(csmGetParameterMaximumValues) X(csmGetParameterValues) \
 X(csmGetPartCount) X(csmGetPartIds) X(csmGetPartOpacities) X(csmGetDrawableCount) X(csmGetDrawableConstantFlags) X(csmGetDrawableDynamicFlags) \
 X(csmGetDrawableBlendModes) X(csmGetDrawableTextureIndices) X(csmGetDrawableDrawOrders) \
 X(csmGetDrawableOpacities) X(csmGetDrawableMaskCounts) X(csmGetDrawableMasks) \
 X(csmGetDrawableVertexCounts) X(csmGetDrawableVertexPositions) X(csmGetDrawableVertexUvs) \
 X(csmGetDrawableIndexCounts) X(csmGetDrawableIndices) X(csmGetDrawableMultiplyColors) \
 X(csmGetDrawableScreenColors) X(csmGetOffscreenCount)
struct Core {
    void* library=nullptr;
#define MEMBER(n) decltype(&::n) n=nullptr;
    SIGHT_CUBISM_API(MEMBER)
#undef MEMBER
    Core()=default;Core(const Core&)=delete;Core& operator=(const Core&)=delete;
    ~Core();
    bool open(std::string& error);
};
struct Image {int width=0,height=0;std::vector<uint8_t> rgba;};
struct Drawable {
    int texture=0,blend=0,mask=-1;
    bool inverted=false;
    std::vector<uint16_t> indices;
    std::vector<csmVector2> uv;
};
struct PosePart {std::string id;int part=-1,parameter=-1;std::vector<int> links;};
struct NativeModel {
    Core core;
    Package package;
    std::vector<uint8_t> moc_memory,model_memory;
    csmModel* model=nullptr;
    std::vector<Image> images;
    std::vector<Drawable> drawables;
    std::vector<std::vector<int>> masks;
    std::vector<Parameter> rig;
    std::vector<float> current;
    std::vector<std::vector<PosePart>> pose_groups;
    std::vector<float> initial_part_opacities;
    std::map<std::string,Motion> motions;
    std::map<std::string,std::vector<ExpressionValue>> expressions;
    std::vector<std::string> part_ids;
    std::string active_motion;
    double motion_elapsed=0;
    void apply_pose(const Values& authored={});
    csmVector2 canvas{},origin{};
    float pixels_per_unit=0;
    double elapsed=0;
    void animate(const TrackingSignal* face,float dt,bool idle,float ease_ms,const std::string& motion_group,const std::string& expression);
};
std::unique_ptr<NativeModel> load_native(const std::filesystem::path&,std::string& error);
}
