#include "core/tasks_abi_123.hpp"
#include "core/tasks_runtime_abi.hpp"
#include <cstddef>
#include <cstdio>
#include <type_traits>
using namespace sight::mp123;
static_assert(std::is_standard_layout_v<BaseOptions>);
static_assert(sizeof(BaseOptions)==80);
static_assert(offsetof(BaseOptions,file_descriptor)==24);
static_assert(offsetof(BaseOptions,delegate)==28);
static_assert(offsetof(BaseOptions,app_id)==56);
static_assert(offsetof(BaseOptions,npu_dispatch_library_dir)==72);
static_assert(sizeof(FaceOptions)==112);
static_assert(offsetof(FaceOptions,running_mode)==80);
static_assert(offsetof(FaceOptions,result_callback)==104);
static_assert(sizeof(PoseOptions)==112);
static_assert(offsetof(PoseOptions,output_segmentation_masks)==100);
static_assert(sizeof(GestureOptions)==208);
static_assert(offsetof(GestureOptions,canned_gestures_classifier_options)==104);
static_assert(offsetof(GestureOptions,custom_gestures_classifier_options)==152);
static_assert(offsetof(GestureOptions,result_callback)==200);
int main(){
 BaseOptions base{};
 if(base.file_descriptor!=-1||base.delegate!=1)return 1;
 if(sight::tasks_runtime_abi(33886744)!=sight::TasksRuntimeAbi::Legacy)return 2;
 if(sight::tasks_runtime_abi(127908512)!=sight::TasksRuntimeAbi::V123)return 3;
 if(sight::tasks_runtime_abi(127908511)!=sight::TasksRuntimeAbi::Unknown)return 4;
 if(sight::tasks_runtime_abi(0)!=sight::TasksRuntimeAbi::Unknown)return 5;
 puts("TASKS_123_ABI_LAYOUT_PASS=1");
}
