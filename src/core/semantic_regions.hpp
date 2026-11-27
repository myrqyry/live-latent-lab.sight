#pragma once
#include "privacy_coverage.hpp"
#include <limits>
namespace sight::semantic {
struct Letterbox {unsigned input_w=0,input_h=0,width=0,height=0,left=0,top=0;};
inline Letterbox letterbox(unsigned w,unsigned h,unsigned iw,unsigned ih) {
 if(!w||!h||!iw||!ih)return {};
 const double scale=std::min(double(iw)/w,double(ih)/h);
 const auto rw=std::clamp(unsigned(std::lround(w*scale)),1u,iw);
 const auto rh=std::clamp(unsigned(std::lround(h*scale)),1u,ih);
 return {iw,ih,rw,rh,(iw-rw)/2,(ih-rh)/2};
}
inline bool parse_ids(std::string_view s,std::vector<int>& out) {
 std::vector<int> ids;
 s=privacy::trim_zone(s);
 while(!s.empty()) {
  auto split=s.find(',');auto part=privacy::trim_zone(s.substr(0,split));
  int id=-1;auto [end,ec]=std::from_chars(part.data(),part.data()+part.size(),id);
  if(part.empty()||ec!=std::errc{}||end!=part.data()+part.size()||id<0||id>1023)return false;
  ids.push_back(id);
  if(split==std::string_view::npos)break;
  s=privacy::trim_zone(s.substr(split+1));if(s.empty())return false;
 }
 std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
 out=std::move(ids);return true;
}
inline std::vector<float> prepare(const uint8_t* rgba,unsigned w,unsigned h,const Letterbox& t) {
 if(!rgba||!w||!h||!t.width||!t.height)return {};
 const size_t plane=size_t(t.input_w)*t.input_h;
 std::vector<float> result(3*plane,114.f/255.f);
 for(unsigned y=0;y<t.height;++y)for(unsigned x=0;x<t.width;++x) {
  const float sx=std::clamp((x+.5f)*w/t.width-.5f,0.f,float(w-1));
  const float sy=std::clamp((y+.5f)*h/t.height-.5f,0.f,float(h-1));
  const unsigned x0=unsigned(sx),y0=unsigned(sy),x1=std::min(x0+1,w-1),y1=std::min(y0+1,h-1);
  const float dx=sx-x0,dy=sy-y0;
  const size_t dest=size_t(y+t.top)*t.input_w+x+t.left;
  for(unsigned c=0;c<3;++c) {
   const auto at=[&](unsigned xx,unsigned yy){return float(rgba[(size_t(yy)*w+xx)*4+c]);};
   result[c*plane+dest]=((at(x0,y0)*(1-dx)+at(x1,y0)*dx)*(1-dy)+
                         (at(x0,y1)*(1-dx)+at(x1,y1)*dx)*dy)/255.f;
  }
 }
 return result;
}
struct Threshold {int category;float value;bool operator==(const Threshold&) const=default;};
struct Region {privacy::Rect box;int category;float score;uint64_t id=0;bool held=false;};
inline bool parse_thresholds(std::string_view s,std::vector<Threshold>& out) {
 std::vector<Threshold> next;
 s=privacy::trim_zone(s);
 while(!s.empty()) {
  auto comma=s.find(',');auto pair=privacy::trim_zone(s.substr(0,comma));
  auto colon=pair.find(':');if(colon==std::string_view::npos)return false;
  auto a=privacy::trim_zone(pair.substr(0,colon)),b=privacy::trim_zone(pair.substr(colon+1));
  int id=-1;float value=0;
  auto [ae,ac]=std::from_chars(a.data(),a.data()+a.size(),id);
  auto [be,bc]=std::from_chars(b.data(),b.data()+b.size(),value);
  if(ac!=std::errc{}||ae!=a.data()+a.size()||bc!=std::errc{}||be!=b.data()+b.size()||
     id<0||id>1023||!std::isfinite(value)||value<.01f||value>1.f||
     std::any_of(next.begin(),next.end(),[&](auto v){return v.category==id;}))return false;
  next.push_back({id,value});
  if(comma==std::string_view::npos)break;
  s=privacy::trim_zone(s.substr(comma+1));if(s.empty())return false;
 }
 out=std::move(next);return true;
}
inline std::vector<privacy::Rect> boxes_of(const std::vector<Region>& regions) {
 std::vector<privacy::Rect> boxes;boxes.reserve(regions.size());
 for(const auto& r:regions)boxes.push_back(r.box);
 return boxes;
}
// Raw YOLO channel-major output: [1,4+classes,N], decoded boxes in input pixels.
// Masks represent the union of all selected boxes; NMS is unnecessary for coverage.
inline bool decode_regions(const float* data,size_t channels,size_t n,const Letterbox& t,
                   const std::vector<int>& ids,float threshold,float padding,
                   const std::vector<Threshold>& overrides,std::vector<Region>& out) {
 out.clear();
 if(!data||channels<5||channels>1028||!n||n>100000||!t.width||!t.height||
    !std::isfinite(threshold)||threshold<=0||threshold>1||
    !std::isfinite(padding)||padding<0||padding>1)return false;
 for(int id:ids)if(id<0||size_t(id)>=channels-4)return false;
 std::vector<float> thresholds(channels-4,threshold);
 for(auto rule:overrides) {
  if(rule.category<0||size_t(rule.category)>=channels-4||!std::isfinite(rule.value)||rule.value<.01f||rule.value>1.f)return false;
  thresholds[rule.category]=rule.value;
 }
 std::vector<Region> next;
 for(size_t i=0;i<n;++i) {
  float best=0;int category=-1;
  for(size_t c=4;c<channels;++c) {
   const float score=data[c*n+i];
   if(!std::isfinite(score)||score<0||score>1)return false;
   if((ids.empty()||std::find(ids.begin(),ids.end(),int(c-4))!=ids.end())&&score>=thresholds[c-4]&&score>best){best=score;category=int(c-4);}
  }
  if(category<0)continue;
  const float cx=data[i],cy=data[n+i],w=data[2*n+i],h=data[3*n+i];
  if(!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0)return false;
  // Reject boxes entirely outside the image before expanding coverage.
  const float x0=(cx-w*.5f-t.left)/t.width,x1=(cx+w*.5f-t.left)/t.width;
  const float y0=(cy-h*.5f-t.top)/t.height,y1=(cy+h*.5f-t.top)/t.height;
  if(x1<=0||y1<=0||x0>=1||y0>=1)continue;
  privacy::Rect r{std::clamp(x0-padding*w/t.width,0.f,1.f),
                  std::clamp(y0-padding*h/t.height,0.f,1.f),
                  std::clamp(x1+padding*w/t.width,0.f,1.f),
                  std::clamp(y1+padding*h/t.height,0.f,1.f)};
  if(privacy::valid_rect(r))next.push_back({r,category,best});
 }
 out=std::move(next);return true;
}
inline bool decode(const float* data,size_t channels,size_t n,const Letterbox& t,
                   const std::vector<int>& ids,float threshold,float padding,
                   std::vector<privacy::Rect>& out) {
 std::vector<Region> regions;
 const bool ok=decode_regions(data,channels,n,t,ids,threshold,padding,{},regions);
 out=ok?boxes_of(regions):std::vector<privacy::Rect>{};return ok;
}
}
