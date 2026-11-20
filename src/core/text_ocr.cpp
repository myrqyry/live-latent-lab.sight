#include "text_ocr.hpp"
#include "sight_text_v1.h"
#include <tesseract/resultiterator.h>
#include <cmath>
#include <algorithm>
namespace sight {
bool TextOcr::load(const std::string& path,const std::string& language) {
    api_.reset();
    if(path.empty()||language.empty())return false;
    auto next=std::make_unique<tesseract::TessBaseAPI>();
    if(next->Init(path.c_str(),language.c_str(),tesseract::OEM_LSTM_ONLY)!=0)return false;
    api_=std::move(next);
    return true;
}
TextResult TextOcr::recognize(const std::vector<uint8_t>& gray,unsigned w,unsigned h,int mode) {
    TextResult out;
    if(!api_||!w||!h||w>2048||h>2048||gray.size()!=size_t(w)*h)return out;
    if(mode!=3&&mode!=6&&mode!=7&&mode!=11)return out;
    api_->SetPageSegMode(static_cast<tesseract::PageSegMode>(mode));
    api_->SetImage(gray.data(),int(w),int(h),1,int(w));
    api_->SetSourceResolution(150);
    if(api_->Recognize(nullptr)!=0) {api_->Clear();return out;}
    bool overflow=false;
    {
        std::unique_ptr<tesseract::ResultIterator> it(api_->GetIterator());
        if(it)do {
            TextLine line;
            std::unique_ptr<char[]> text(it->GetUTF8Text(tesseract::RIL_TEXTLINE));
            if(!text||!text[0]||!it->BoundingBox(tesseract::RIL_TEXTLINE,&line.x0,&line.y0,&line.x1,&line.y1))continue;
            if(line.x0<0||line.y0<0||line.x1>int(w)||line.y1>int(h)||line.x0>=line.x1||line.y0>=line.y1)continue;
            line.confidence=it->Confidence(tesseract::RIL_TEXTLINE);
            if(!std::isfinite(line.confidence))continue;
            line.text=text.get();
            out.lines.push_back(std::move(line));
            // Bounded work; an over-capacity result is not silently valid.
            if(out.lines.size()>SIGHT_TEXT_MAX_LINES){out.lines.clear();overflow=true;break;}
        } while(it->Next(tesseract::RIL_TEXTLINE));
        // Empty pages are valid, including an empty iterator.
        out.valid=!overflow;
    }
    api_->Clear();
    return out;
}
}
