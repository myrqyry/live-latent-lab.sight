#include "object_overlay.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>
namespace sight {
namespace {
// 3x5 glyphs. Supports COCO category names, digits and ID tags without
// introducing a font or Qt dependency on Sight's rendering worker.
std::array<unsigned char,5> glyph(char ch){
    switch(std::tolower(static_cast<unsigned char>(ch))){
      case 'a':return {2,5,7,5,5}; case 'b':return {6,5,6,5,6};
      case 'c':return {3,4,4,4,3}; case 'd':return {6,5,5,5,6};
      case 'e':return {7,4,6,4,7}; case 'f':return {7,4,6,4,4};
      case 'g':return {3,4,5,5,3}; case 'h':return {5,5,7,5,5};
      case 'i':return {7,2,2,2,7}; case 'j':return {3,1,1,5,2};
      case 'k':return {5,5,6,5,5}; case 'l':return {4,4,4,4,7};
      case 'm':return {5,7,7,5,5}; case 'n':return {5,7,7,7,5};
      case 'o':return {2,5,5,5,2}; case 'p':return {6,5,6,4,4};
      case 'q':return {2,5,5,3,1}; case 'r':return {6,5,6,5,5};
      case 's':return {3,4,2,1,6}; case 't':return {7,2,2,2,2};
      case 'u':return {5,5,5,5,7}; case 'v':return {5,5,5,5,2};
      case 'w':return {5,5,7,7,5}; case 'x':return {5,5,2,5,5};
      case 'y':return {5,5,2,2,2}; case 'z':return {7,1,2,4,7};
      case '0':return {7,5,5,5,7}; case '1':return {2,6,2,2,7};
      case '2':return {7,1,7,4,7}; case '3':return {7,1,3,1,7};
      case '4':return {5,5,7,1,1}; case '5':return {7,4,7,1,7};
      case '6':return {7,4,7,5,7}; case '7':return {7,1,2,2,2};
      case '8':return {7,5,7,5,7}; case '9':return {7,5,7,1,7};
      case '#':return {5,7,5,7,5}; case '-':return {0,0,7,0,0};
      case ':':return {0,2,0,2,0}; case ' ':return {0,0,0,0,0};
      default:return {0,0,2,0,0};
    }
}
struct Canvas{
    std::vector<uint8_t>& data;
    int width,height;
    void put(int x,int y,int r,int g,int b,int a) {
        if(x<0||y<0||x>=width||y>=height)return;
        auto i=(size_t(y)*width+size_t(x))*4;
        data[i]=static_cast<uint8_t>((r*a+127)/255);
        data[i+1]=static_cast<uint8_t>((g*a+127)/255);
        data[i+2]=static_cast<uint8_t>((b*a+127)/255);
        data[i+3]=uint8_t(a);
    }
    void label(int x,int y,std::string name,int id){
        name.resize(std::min<size_t>(name.size(),22));
        name+=" #"+std::to_string(id);
        // 3x5 at 2x for legibility, plus 2px margins.
        const int bar_width=int(name.size())*8+4;
        const int x0=std::clamp(x,0,std::max(0,width-bar_width));
        const int y0=y>=15?y-15:std::min(height-14,y+4);
        for(int yy=y0;yy<y0+14;++yy)
            for(int xx=x0;xx<x0+bar_width;++xx)
                put(xx,yy,23,35,39,210);
        for(size_t j=0;j<name.size();++j){
            const auto rows=glyph(name[j]);
            for(int py=0;py<5;++py)
                for(int px=0;px<3;++px)
                    if(rows[size_t(py)]&(1<<(2-px)))
                        for(int iy=0;iy<2;++iy)
                            for(int ix=0;ix<2;++ix)
                                put(x0+2+int(j)*8+px*2+ix,
                                    y0+2+py*2+iy,255,255,255,240);
        }
    }
};
} // namespace
std::vector<uint8_t> draw_object_overlay(uint32_t width,uint32_t height,
                                        const std::vector<ObjectDetection>& objects){
    if(width<1||height<1||width>2048||height>2048)return {};
    std::vector<uint8_t> rgba(size_t(width)*height*4,0);
    Canvas c{rgba,int(width),int(height)};
    const std::array<std::array<int,3>,5> palette{{
        {244,188,137},{139,219,201},{177,195,249},{240,178,203},{245,217,134}
    }};
    for(const auto& o:objects) {
        if(!valid_object(o)||o.id<=0)continue;
        const auto color=palette[size_t(o.id)%palette.size()];
        const int x0=int(std::lround(o.x*width)),y0=int(std::lround(o.y*height));
        const int x1=int(std::lround((o.x+o.w)*width))-1;
        const int y1=int(std::lround((o.y+o.h)*height))-1;
        if(x1<=x0||y1<=y0)continue;
        for(int k=0;k<2;++k) {
            for(int x=x0;x<=x1;++x){
                c.put(x,y0+k,color[0],color[1],color[2],230);
                c.put(x,y1-k,color[0],color[1],color[2],230);
            }
            for(int y=y0;y<=y1;++y){
                c.put(x0+k,y,color[0],color[1],color[2],230);
                c.put(x1-k,y,color[0],color[1],color[2],230);
            }
        }
        c.label(x0,y0,o.label,o.id);
    }
    return rgba;
}

} // namespace sight
