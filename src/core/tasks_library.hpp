#pragma once
#include "tasks_runtime_abi.hpp"
#include <string>
#if defined(__linux__) && (defined(__x86_64__) || defined(__amd64__))
#include <dlfcn.h>
#endif
namespace sight {
// Keep the existing verified Linux runtime mapped until process shutdown.
// Other hosts cannot load the verified Linux ABI, regardless of file size.
inline void* open_tasks_library(const std::string& path,std::string& error) {
#if defined(__linux__) && (defined(__x86_64__) || defined(__amd64__))
    auto* library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);
    if(!library) {
        const char* message=dlerror();
        error=message?message:"MediaPipe library could not be loaded";
    }
    return library;
#else
    (void)path;
    error="No ABI-qualified native MediaPipe runtime for this host";
    return nullptr;
#endif
}
inline void close_tasks_library(void* library) {
#if defined(__linux__) && (defined(__x86_64__) || defined(__amd64__))
    if(library)dlclose(library);
#else
    (void)library;
#endif
}
inline void* tasks_library_symbol(void* library,const char* name) {
#if defined(__linux__) && (defined(__x86_64__) || defined(__amd64__))
    return dlsym(library,name);
#else
    (void)library;(void)name;return nullptr;
#endif
}
}
