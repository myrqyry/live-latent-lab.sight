#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <optional>
#include <string_view>
#include <vector>
namespace sight {
struct OcrRegion {unsigned x,y,w,h;};
inline OcrRegion ocr_region(unsigned w,unsigned h,int x,int y,int width,int height){
 if(!w||!h)return {};
 x=std::clamp(x,0,99);y=std::clamp(y,0,99);
 width=std::clamp(width,1,100-x);height=std::clamp(height,1,100-y);
 unsigned l=unsigned(uint64_t(w)*x/100),t=unsigned(uint64_t(h)*y/100);
 unsigned r=std::max(l+1,unsigned((uint64_t(w)*(x+width)+99)/100));
 unsigned b=std::max(t+1,unsigned((uint64_t(h)*(y+height)+99)/100));
 return {l,t,std::min(w,r)-l,std::min(h,b)-t};
}
inline std::vector<uint8_t> ocr_crop(const std::vector<uint8_t>& src,unsigned w,unsigned h,OcrRegion r){
 if(src.size()!=size_t(w)*h||!r.w||!r.h||r.x>w||r.y>h||r.w>w-r.x||r.h>h-r.y)return {};
 std::vector<uint8_t> out(size_t(r.w)*r.h);
 for(unsigned y=0;y<r.h;++y)std::copy_n(src.begin()+size_t(r.y+y)*w+r.x,r.w,out.begin()+size_t(y)*r.w);
 return out;
}
inline std::string ocr_clean(std::string_view s){
 std::string out;out.reserve(s.size());
 for(size_t i=0;i<s.size();++i){if(s[i]=='\r'){if(i+1<s.size()&&s[i+1]=='\n')++i;out+='\n';}else out+=s[i];}
 auto first=out.find_first_not_of(" \t\n");if(first==std::string::npos)return {};
 return out.substr(first,out.find_last_not_of(" \t\n")-first+1);
}
inline std::optional<std::string> ocr_format(std::string_view format,std::string_view region,std::string_view text){
 if(text.empty())return std::string{};
 std::string out;
 for(size_t i=0;i<format.size();){
  if(format.substr(i).starts_with("{text}")){out+=text;i+=6;}
  else if(format.substr(i).starts_with("{region}")){out+=region;i+=8;}
  else out+=format[i++];
  if(out.size()>16384)return std::nullopt;
 }
 return out;
}
// A run of matching complete UTF-8 strings, not character-wise averaging
// (which can invent scores or corrupt multibyte text). Empty is a reading.
class OcrConsensus {
 std::string candidate_,value_;
 unsigned run_=0;
public:
 void reset(){candidate_.clear();value_.clear();run_=0;}
 bool observe(const std::string& text,unsigned required){
  required=std::clamp(required,1u,5u);
  if(!run_||candidate_!=text){candidate_=text;run_=1;}
  else if(run_<5)++run_;
  if(run_<required)return false;
  value_=candidate_;return true;
 }
 const std::string& value() const{return value_;}
};
}
