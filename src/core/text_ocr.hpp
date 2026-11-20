#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <tesseract/baseapi.h>
namespace sight {
struct TextLine { int x0=0,y0=0,x1=0,y1=0; float confidence=0; std::string text; };
struct TextResult { bool valid=false; std::vector<TextLine> lines; };
class TextOcr {
    std::unique_ptr<tesseract::TessBaseAPI> api_;
public:
    bool load(const std::string& path,const std::string& language);
    TextResult recognize(const std::vector<uint8_t>& gray,unsigned w,unsigned h,int mode);
};
}
