#pragma once
#include <string>
namespace sight_preferences {
#ifdef SIGHT_WITH_SETTINGS_PAGE
std::string directory(bool runtime=false);
#else
inline std::string directory(bool=false){return {};}
#endif
inline std::string browse(std::string fallback={},bool runtime=false){
    auto configured=directory(runtime);
    return configured.empty()?fallback:configured;
}
}
