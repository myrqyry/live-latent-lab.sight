#pragma once
#include <algorithm>
#include <cmath>
#include <optional>
namespace sight {
struct OcrEditorRect {
 int x=0,y=0,w=100,h=100;
 bool operator==(const OcrEditorRect&) const=default;
};
struct OcrEditorFit {double x=0,y=0,w=0,h=0;};
inline OcrEditorFit ocr_editor_fit(double w,double h,unsigned sw,unsigned sh){
 if(w<=0||h<=0||!sw||!sh)return {};
 double scale=std::min(w/sw,h/sh);
 double fw=sw*scale,fh=sh*scale;
 return {(w-fw)/2,(h-fh)/2,fw,fh};
}
inline OcrEditorRect ocr_editor_clamp(OcrEditorRect r){
 r.x=std::clamp(r.x,0,99);r.y=std::clamp(r.y,0,99);
 r.w=std::clamp(r.w,1,100-r.x);r.h=std::clamp(r.h,1,100-r.y);
 return r;
}
inline std::optional<OcrEditorRect> ocr_editor_drag(OcrEditorFit f,double ax,double ay,double bx,double by){
 if(f.w<=0||f.h<=0||ax<f.x||ax>f.x+f.w||ay<f.y||ay>f.y+f.h)return {};
 bx=std::clamp(bx,f.x,f.x+f.w);by=std::clamp(by,f.y,f.y+f.h);
 if(std::abs(ax-bx)<3||std::abs(ay-by)<3)return {};
 auto xp=[&](double x){return int(std::lround((x-f.x)*100/f.w));};
 auto yp=[&](double y){return int(std::lround((y-f.y)*100/f.h));};
 int l=xp(std::min(ax,bx)),r=xp(std::max(ax,bx)),t=yp(std::min(ay,by)),b=yp(std::max(ay,by));
 return ocr_editor_clamp({l,t,r-l,b-t});
}
}
