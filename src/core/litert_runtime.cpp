#include "litert_runtime.hpp"

#include <litert/c/litert_environment.h>
#include <litert/c/litert_model.h>
#include <litert/c/litert_options.h>
#include <litert/c/litert_compiled_model.h>
#include <litert/c/litert_tensor_buffer.h>
#include <algorithm>
#include <cstring>
#include <list>
#include <dlfcn.h>

namespace lll {
namespace {
constexpr size_t MaxSessions = 4;
constexpr size_t MaxTensorBytes = 256u * 1024u * 1024u;

struct Session {
    std::filesystem::path path;
    std::filesystem::file_time_type stamp;
    uintmax_t bytes = 0;
    LiteRtDevice device;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiled = nullptr;
    std::vector<LiteRtTensorBuffer> inputs, outputs;
    ~Session() {
        for (auto b : inputs) if (b) LiteRtDestroyTensorBuffer(b);
        for (auto b : outputs) if (b) LiteRtDestroyTensorBuffer(b);
        if (compiled) LiteRtDestroyCompiledModel(compiled);
        if (model) LiteRtDestroyModel(model);
    }
};
struct Options {
    LiteRtOptions value = nullptr;
    ~Options() { if(value) LiteRtDestroyOptions(value); }
};

bool check(LiteRtStatus status, const char* operation, std::string& error) {
    if(status==kLiteRtStatusOk)return true;
    error=std::string(operation)+": LiteRT status "+std::to_string(int(status));
    return false;
}

bool buffer_size(LiteRtTensorBuffer buffer,size_t& bytes,std::string& error) {
    LiteRtRankedTensorType type{};
    if(!check(LiteRtGetTensorBufferTensorType(buffer,&type),"tensor type",error) ||
       !check(LiteRtGetTensorBufferPackedSize(buffer,&bytes),"tensor size",error))
        return false;
    if(type.element_type!=kLiteRtElementTypeFloat32 || !bytes ||
       bytes>MaxTensorBytes || bytes%sizeof(float)) {
        error="LiteRT requires bounded float32 tensor buffers";
        return false;
    }
    return true;
}

bool create_buffers(LiteRtEnvironment env,Session& session,std::string& error) {
    LiteRtSignature signature=nullptr;
    LiteRtParamIndex ni=0,no=0;
    if(!check(LiteRtGetModelSignature(session.model,0,&signature),"signature",error)||
       !check(LiteRtGetNumSignatureInputs(signature,&ni),"input count",error)||
       !check(LiteRtGetNumSignatureOutputs(signature,&no),"output count",error))
        return false;
    if(ni!=1||no==0||no>32){error="Unsupported LiteRT signature";return false;}
    session.inputs.resize(ni,nullptr);
    session.outputs.resize(no,nullptr);
    for(bool input:{true,false}) {
        auto& buffers=input?session.inputs:session.outputs;
        for(size_t i=0;i<buffers.size();++i) {
            LiteRtTensor tensor=nullptr;
            LiteRtRankedTensorType type{};
            LiteRtTensorBufferRequirements requirements=nullptr;
            if(!check(input?LiteRtGetSignatureInputTensorByIndex(signature,i,&tensor):
                            LiteRtGetSignatureOutputTensorByIndex(signature,i,&tensor),
                      "signature tensor",error)||
               !check(LiteRtGetRankedTensorType(tensor,&type),"ranked type",error)||
               !check(input?LiteRtGetCompiledModelInputBufferRequirements(session.compiled,0,i,&requirements):
                            LiteRtGetCompiledModelOutputBufferRequirements(session.compiled,0,i,&requirements),
                      "buffer requirements",error))
                return false;
            if(input && !check(LiteRtGetCompiledModelInputTensorLayout(session.compiled,0,i,&type.layout),
                              "compiled input layout",error))return false;
            if(!check(LiteRtCreateManagedTensorBufferFromRequirements(env,&type,requirements,&buffers[i]),
                      "allocate tensor buffer",error))return false;
            size_t bytes=0;
            if(!buffer_size(buffers[i],bytes,error))return false;
        }
    }
    return true;
}
}

struct RuntimeState {
    LiteRtEnvironment environment=nullptr;
    std::string runtime_directory;
    std::list<std::unique_ptr<Session>> sessions;
    size_t compilations=0;
    ~RuntimeState() {
        sessions.clear(); // model/buffers must die before environment
        if(environment)LiteRtDestroyEnvironment(environment);
    }
};

LiteRtRuntime::LiteRtRuntime() = default;
LiteRtRuntime::~LiteRtRuntime() = default;
LiteRtRuntime& LiteRtRuntime::instance() {
    static LiteRtRuntime runtime;
    return runtime;
}
bool LiteRtRuntime::initialize() {
    std::lock_guard lock(mutex_);
    if(state_)return true;
    auto state=std::make_unique<RuntimeState>();
    // Resolve the accelerator relative to the actual loaded C runtime,
    // including private staged OBS installations, not the current directory.
    Dl_info info{};
    if(!dladdr(reinterpret_cast<void*>(&LiteRtCreateEnvironment),&info)||!info.dli_fname){
        last_error_="Cannot locate loaded LiteRT runtime";return false;
    }
    state->runtime_directory=std::filesystem::path(info.dli_fname).parent_path().string();
    LiteRtEnvOption option{};
    option.tag=kLiteRtEnvOptionTagRuntimeLibraryDir;
    option.value.type=kLiteRtAnyTypeString;
    option.value.str_value=state->runtime_directory.c_str();
    if(!check(LiteRtCreateEnvironment(1,&option,&state->environment),"create environment",last_error_))
        return false;
    state_=std::move(state);
    last_error_.clear();
    return true;
}
void LiteRtRuntime::shutdown() {
    std::lock_guard lock(mutex_);
    state_.reset();
}
bool LiteRtRuntime::ready() const {
    std::lock_guard lock(mutex_);
    return bool(state_);
}
std::string LiteRtRuntime::last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}
size_t LiteRtRuntime::compilation_count() const {
    std::lock_guard lock(mutex_);
    return state_?state_->compilations:0;
}
size_t LiteRtRuntime::cached_session_count() const {
    std::lock_guard lock(mutex_);
    return state_?state_->sessions.size():0;
}
bool LiteRtRuntime::run(const std::filesystem::path& path,LiteRtDevice device,
                       const std::vector<float>& input,std::vector<float>& output,
                       const LiteRtRunOptions& options) {
    std::vector<std::vector<float>> outputs;
    if(!run_multi(path,device,input,outputs,options))return false;
    if(outputs.size()!=1){
        std::lock_guard lock(mutex_);
        last_error_="LiteRT runtime requires one output tensor";
        return false;
    }
    output=std::move(outputs.front());
    return true;
}
bool LiteRtRuntime::run_multi(const std::filesystem::path& path,LiteRtDevice device,
                             const std::vector<float>& input,std::vector<std::vector<float>>& outputs,
                             const LiteRtRunOptions& options) {
    std::lock_guard lock(mutex_);
    if(!state_){last_error_="LiteRT runtime is not initialized";return false;}
    std::error_code ec;
    const auto canonical=std::filesystem::canonical(path,ec);
    if(ec||!std::filesystem::is_regular_file(canonical,ec)||ec){
        last_error_="LiteRT model does not exist: "+path.string();return false;
    }
    if(run_once_multi(canonical,device,input,outputs))return true;
    if(device==LiteRtDevice::Gpu&&options.allow_cpu_fallback){
        const auto gpu_error=last_error_;
        if(run_once_multi(canonical,LiteRtDevice::Cpu,input,outputs))return true;
        last_error_=gpu_error+"; CPU fallback: "+last_error_;
    }
    return false;
}
bool LiteRtRuntime::run_once_multi(const std::filesystem::path& path,LiteRtDevice device,
                                  const std::vector<float>& input,std::vector<std::vector<float>>& output_values) {
    std::error_code ec;
    const auto stamp=std::filesystem::last_write_time(path,ec);
    if(ec){last_error_="Cannot stat LiteRT model";return false;}
    const auto bytes=std::filesystem::file_size(path,ec);
    if(ec){last_error_="Cannot stat LiteRT model size";return false;}
    auto& sessions=state_->sessions;
    Session* session=nullptr;
    for(auto it=sessions.begin();it!=sessions.end();) {
        auto& item=**it;
        if(item.path==path&&(item.stamp!=stamp||item.bytes!=bytes)){
            it=sessions.erase(it);continue;
        }
        if(item.path==path&&item.device==device){
            session=&item;
            sessions.splice(sessions.begin(),sessions,it);
            break;
        }
        ++it;
    }
    if(!session) {
        auto next=std::make_unique<Session>();
        next->path=path;next->stamp=stamp;next->bytes=bytes;next->device=device;
        Options opts;
        if(!check(LiteRtCreateModelFromFile(state_->environment,path.c_str(),&next->model),"load model",last_error_)||
           !check(LiteRtCreateOptions(&opts.value),"create options",last_error_)||
           !check(LiteRtSetOptionsHardwareAccelerators(opts.value,device==LiteRtDevice::Gpu?
                 kLiteRtHwAcceleratorGpu:kLiteRtHwAcceleratorCpu),"set accelerator",last_error_)||
           !check(LiteRtCreateCompiledModel(state_->environment,next->model,opts.value,&next->compiled),
                 "compile model",last_error_))return false;
        ++state_->compilations;
        if(device==LiteRtDevice::Gpu){
            bool full=false;
            if(!check(LiteRtCompiledModelIsNonCpuFullyAccelerated(next->compiled,&full),
                      "verify GPU delegation",last_error_))return false;
            if(!full){last_error_="GPU model is not fully accelerated";return false;}
        }
        if(!create_buffers(state_->environment,*next,last_error_))return false;
        while(sessions.size()>=MaxSessions)sessions.pop_back();
        sessions.push_front(std::move(next));
        session=sessions.front().get();
    }
    size_t size=0;
    const auto in=session->inputs.front();
    if(!buffer_size(in,size,last_error_))return false;
    if(input.size()!=size/sizeof(float)){last_error_="LiteRT input tensor size mismatch";return false;}
    void* mapped=nullptr;
    if(!check(LiteRtLockTensorBuffer(in,&mapped,kLiteRtTensorBufferLockModeWrite),"lock input",last_error_))
        return false;
    std::memcpy(mapped,input.data(),size);
    if(!check(LiteRtUnlockTensorBuffer(in),"unlock input",last_error_)||
       !check(LiteRtRunCompiledModel(session->compiled,0,session->inputs.size(),session->inputs.data(),
                 session->outputs.size(),session->outputs.data()),"run model",last_error_))
        return false;
    std::vector<std::vector<float>> result;
    result.reserve(session->outputs.size());
    for(auto buffer:session->outputs){
        if(!buffer_size(buffer,size,last_error_))return false;
        result.emplace_back(size/sizeof(float));
        if(!check(LiteRtLockTensorBuffer(buffer,&mapped,kLiteRtTensorBufferLockModeRead),"lock output",last_error_))
            return false;
        std::memcpy(result.back().data(),mapped,size);
        if(!check(LiteRtUnlockTensorBuffer(buffer),"unlock output",last_error_))return false;
    }
    output_values=std::move(result);
    last_error_.clear();
    return true;
}
void LiteRtRuntime::set_error(std::string error){last_error_=std::move(error);}
} // namespace lll
