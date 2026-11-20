#include "tracking_overlay.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace sight {
namespace {
struct Canvas {
    std::vector<uint8_t>& rgba;
    int w, h;
    std::array<uint8_t,3> color;
    void pixel(int x, int y, uint8_t alpha) {
        if(x<0||y<0||x>=w||y>=h)return;
        const size_t idx=(static_cast<size_t>(y)*w+x)*4;
        // Premultiplied alpha for shader composition; strongest marker wins.
        if(alpha<=rgba[idx+3])return;
        for(size_t c=0;c<3;c++) rgba[idx+c]=
            static_cast<uint8_t>((unsigned(color[c])*alpha+127)/255);
        rgba[idx+3]=alpha;
    }
    void dot(int x,int y,int radius,uint8_t alpha){
        for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++)
            if(dx*dx+dy*dy<=radius*radius)pixel(x+dx,y+dy,alpha);
    }
    void line(int ax,int ay,int bx,int by,uint8_t alpha){
        const int dx=std::abs(bx-ax),dy=std::abs(by-ay);
        // Guard against malformed large coordinates on invalid model outputs.
        if(dx>2*w||dy>2*h)return;
        const int sx=ax<bx?1:-1,sy=ay<by?1:-1;
        int err=dx-dy;
        for(int t=0;t<=dx+dy+1;t++){
            pixel(ax,ay,alpha);
            if(ax==bx&&ay==by)break;
            int e=2*err;
            if(e>-dy){err-=dy;ax+=sx;}
            if(e<dx){err+=dx;ay+=sy;}
        }
    }
};
constexpr std::array<std::pair<int,int>,21> hand_bones{{
    {0,1},{1,2},{2,3},{3,4},
    {0,5},{5,6},{6,7},{7,8},
    {5,9},{9,10},{10,11},{11,12},
    {9,13},{13,14},{14,15},{15,16},
    {13,17},{17,18},{18,19},{19,20},{0,17}
}};
constexpr std::array<std::pair<int,int>,26> pose_bones{{
    {11,12},{11,13},{13,15},{12,14},{14,16},
    {11,23},{12,24},{23,24},{23,25},{25,27},{24,26},{26,28},
    {27,29},{29,31},{28,30},{30,32},
    {0,1},{1,2},{2,3},{3,7},{0,4},{4,5},{5,6},{6,8},
    {15,17},{16,18}
}};
}
std::vector<uint8_t> draw_tracking_overlay(uint32_t width,uint32_t height,
    TrackingKind kind,const TrackingData& result,bool confidence_alpha)
{
    if(width<1||height<1||width>2048||height>2048)return {};
    std::vector<uint8_t> rgba(static_cast<size_t>(width)*height*4,0);
    if(kind==TrackingKind::WholeBody) {
        const auto& points=result.landmarks;
        if(points.size()!=133)return rgba;
        Canvas c{rgba,int(width),int(height),{171,221,245}};
        const auto valid=[&](unsigned i){
            const auto& p=points[i];
            return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.confidence)&&
                p.confidence>=.45f&&p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1;
        };
        const auto pos=[&](unsigned i){return std::pair{
            int(std::lround(points[i].x*(width-1))),
            int(std::lround(points[i].y*(height-1)))};};
        const auto bone=[&](unsigned a,unsigned b){
            if(!valid(a)||!valid(b))return;
            const auto [ax,ay]=pos(a);const auto [bx,by]=pos(b);
            c.line(ax,ay,bx,by,195);
        };
        // COCO WholeBody: body 0..16, feet 17..22, face 23..90,
        // left/right hands 91..111 and 112..132. No MediaPipe indices.
        constexpr std::pair<unsigned,unsigned> bones[]={
            {0,1},{0,2},{1,3},{2,4},{5,6},{5,7},{7,9},{6,8},{8,10},
            {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16},
            {15,17},{17,18},{15,19},{16,20},{20,21},{16,22},
            {9,91},{10,112}};
        for(auto [a,b]:bones)bone(a,b);
        for(unsigned base:{91u,112u})for(auto [a,b]:hand_bones)bone(base+a,base+b);
        for(unsigned i=0;i<133;++i)if(valid(i)){
            auto [x,y]=pos(i);c.dot(x,y,i>=23&&i<91?1:2,230);
        }
        return rgba;
    }
    if(kind==TrackingKind::Hand) {
        // Draw BOTH hands; the main landmark alias is just the first hand.
        for(size_t h=0;h<result.hands.size()&&h<2;++h) {
            const auto& hand=result.hands[h].landmarks;
            if(hand.size()!=21)continue;
            Canvas c{rgba,static_cast<int>(width),static_cast<int>(height),
                     h==0?std::array<uint8_t,3>{244,187,136}:
                          std::array<uint8_t,3>{154,216,214}};
            const auto available=[&](int i){
                return std::isfinite(hand[i].x)&&std::isfinite(hand[i].y) &&
                    hand[i].x>=0&&hand[i].x<=1&&hand[i].y>=0&&hand[i].y<=1 &&
                    (!confidence_alpha ||
                     (std::isfinite(hand[i].confidence) &&
                      hand[i].confidence>.02f));
            };
            const auto alpha=[&](int i,int base){
                if(!confidence_alpha)return uint8_t(base);
                return static_cast<uint8_t>(std::lround(
                    base*std::clamp(hand[i].confidence,0.f,1.f)));
            };
            const auto pixel=[&](int i){
                return std::pair{int(std::lround(hand[i].x*(width-1))),
                                 int(std::lround(hand[i].y*(height-1)))};
            };
            for(const auto [a,b]:hand_bones)if(available(a)&&available(b)){
                const auto [ax,ay]=pixel(a);const auto [bx,by]=pixel(b);
                c.line(ax,ay,bx,by,
                    std::min(alpha(a,195),alpha(b,195)));
            }
            for(int i=0;i<21;++i)if(available(i)){
                const auto [x,y]=pixel(i);
                c.dot(x,y,(i==4||i==8||i==12||i==16||i==20)?3:2,
                      alpha(i,240));
            }
        }
        return rgba;
    }
    if(result.landmarks.size() != (kind==TrackingKind::Face?478u:33u))return rgba;
    Canvas c{rgba,static_cast<int>(width),static_cast<int>(height),
             kind==TrackingKind::Face?std::array<uint8_t,3>{250,196,129}:
                                      std::array<uint8_t,3>{165,222,185}};
    const auto usable=[&](int i){
        const auto& p=result.landmarks[i];
        return std::isfinite(p.x)&&std::isfinite(p.y)&&
               p.x>=0&&p.y>=0&&p.x<=1&&p.y<=1 &&
               p.confidence>(confidence_alpha?.02f:.45f);
    };
    const auto alpha=[&](int i,int base){
        if(!confidence_alpha)return uint8_t(base);
        return static_cast<uint8_t>(std::lround(
            base*std::clamp(result.landmarks[size_t(i)].confidence,0.f,1.f)));
    };
    const auto point=[&](int i){
        return std::pair{static_cast<int>(std::lround(result.landmarks[i].x*(width-1))),
                         static_cast<int>(std::lround(result.landmarks[i].y*(height-1)))};
    };
    if(kind==TrackingKind::Body) {
        for(const auto& [a,b]:pose_bones)if(usable(a)&&usable(b)){
            const auto [ax,ay]=point(a);const auto [bx,by]=point(b);
            c.line(ax,ay,bx,by,std::min(alpha(a,185),alpha(b,185)));
        }
    }
    for(size_t i=0;i<result.landmarks.size();++i){
        if(kind==TrackingKind::Face && i%2==1)continue; // declutter 478-point mesh
        if(!usable(static_cast<int>(i)))continue;
        const auto [x,y]=point(static_cast<int>(i));
        c.dot(x,y,kind==TrackingKind::Face?1:2,alpha(int(i),230));
    }
    return rgba;
}

// Compose premultiplied RGBA layers using normal source-over blending.
// Face -> Body -> Hands keeps hands readable when silhouettes overlap.
// The same renderer is used by each existing Sight tracking filter.
std::vector<uint8_t> draw_holistic_overlay(
    uint32_t width, uint32_t height,
    const TrackingData& face, const TrackingData& body,
    const TrackingData& hands, HolisticOverlayMode mode,
    bool confidence_alpha)
{
    if(width<1||height<1||width>2048||height>2048)return {};
    std::vector<uint8_t> combined(size_t(width)*height*4,0);
    const auto append=[&](TrackingKind kind,const TrackingData& result){
        // Preserve historical Face/Body/Hand standalone rendering.
        // Alpha-based fading is restricted to the optional Holistic
        // stabilization preview, not raw tracking consumers.
        auto layer=draw_tracking_overlay(width,height,kind,result,
                          confidence_alpha && kind!=TrackingKind::Face);
        if(layer.size()!=combined.size())return;
        for(size_t i=0;i<combined.size();i+=4) {
            const unsigned src_a=layer[i+3];
            if(!src_a)continue;
            const unsigned inv=255-src_a;
            for(size_t k=0;k<3;++k)
                combined[i+k]=static_cast<uint8_t>(
                    std::min(255u,unsigned(layer[i+k])+
                        (unsigned(combined[i+k])*inv+127)/255));
            combined[i+3]=static_cast<uint8_t>(
                std::min(255u,src_a+
                    (unsigned(combined[i+3])*inv+127)/255));
        }
    };
    if(mode==HolisticOverlayMode::All||mode==HolisticOverlayMode::Face)
        append(TrackingKind::Face,face);
    if(mode==HolisticOverlayMode::All||mode==HolisticOverlayMode::Body)
        append(TrackingKind::Body,body);
    if(mode==HolisticOverlayMode::All||mode==HolisticOverlayMode::Hands)
        append(TrackingKind::Hand,hands);
    return combined;
}
} // namespace sight
