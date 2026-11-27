#include "core/onnx_session.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <chrono>
int main(int argc,char** argv) {
    const bool cuda=argc==2&&std::string_view(argv[1])=="--cuda";
    if(argc!=1&&!cuda)return 2;
    namespace fs=std::filesystem;
    const auto work=fs::temp_directory_path()/("sight-ort-smoke-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {fs::path path;~Cleanup(){std::error_code e;fs::remove_all(path,e);}} cleanup{work};
    try {
        fs::create_directory(work);
        const auto model=work/fs::u8path(u8"model-é-日本語.onnx");
        const unsigned char bytes[]={8,8,58,65,10,16,10,1,120,18,1,121,34,8,73,100,101,110,116,105,116,121,18,11,115,105,103,104,116,45,115,109,111,107,101,90,15,10,1,120,18,10,10,8,8,1,18,4,10,2,8,1,98,15,10,1,121,18,10,10,8,8,1,18,4,10,2,8,1,66,2,16,17};
        const unsigned char matmul[]={8,8,58,100,10,17,10,1,120,10,1,121,18,1,122,34,6,77,97,116,77,117,108,18,16,115,105,103,104,116,45,99,117,100,97,45,115,109,111,107,101,90,19,10,1,120,18,14,10,12,8,1,18,8,10,2,8,2,10,2,8,2,90,19,10,1,121,18,14,10,12,8,1,18,8,10,2,8,2,10,2,8,2,98,19,10,1,122,18,14,10,12,8,1,18,8,10,2,8,2,10,2,8,2,66,4,10,0,16,17};
        {std::ofstream out(model,std::ios::binary);if(cuda)out.write(reinterpret_cast<const char*>(matmul),sizeof(matmul));else out.write(reinterpret_cast<const char*>(bytes),sizeof(bytes));}
        const auto u8=model.u8string();
        const std::string path(u8.begin(),u8.end());
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"SightNativeSmoke");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        const char* provider=nullptr;
        if(cuda)options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
        auto session=lll::open_onnx_session(env,path.c_str(),options,cuda,&provider,cuda);
        auto info=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        if(cuda) {
            std::array<float,4> a{1,2,3,4},b{5,6,7,8};
            std::array<int64_t,2> shape{2,2};
            std::array<Ort::Value,2> tensors{
                Ort::Value::CreateTensor<float>(info,a.data(),a.size(),shape.data(),shape.size()),
                Ort::Value::CreateTensor<float>(info,b.data(),b.size(),shape.data(),shape.size())};
            const char* names[]={"x","y"},*outputs[]={"z"};
            auto result=session->Run(Ort::RunOptions{nullptr},names,tensors.data(),tensors.size(),outputs,1);
            const std::array<float,4> expected{19,22,43,50};
            if(result.size()!=1||!provider||std::string_view(provider).find("CUDA")==std::string_view::npos||
                !std::equal(expected.begin(),expected.end(),result[0].GetTensorData<float>()))return 2;
            std::cout<<"Native ONNX Runtime "<<Ort::GetVersionString()<<" CUDA-only MatMul passed (CPU execution fallback disabled)\n";
            return 0;
        }
        float input=42.25f;
        std::array<int64_t,1> shape{1};
        auto tensor=Ort::Value::CreateTensor<float>(info,&input,1,shape.data(),shape.size());
        const char* inputs[]={"x"},*outputs[]={"y"};
        auto result=session->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
        if(result.size()!=1||result[0].GetTensorData<float>()[0]!=input||!provider)return 2;
        std::cout<<"Native ONNX Runtime "<<Ort::GetVersionString()<<" identity inference passed on UTF-8 model path\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
