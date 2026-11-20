#include "core/face_quality.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
using namespace sight;
static TrackingData face(float shift=0) {
    TrackingData d;d.landmarks.resize(478,{.5f+shift,.5f,0,1});
    d.landmarks[10]={.5f+shift,.2f,0,1};
    d.landmarks[152]={.5f+shift,.8f,0,1};
    const int a[]={133,173,157,158,159,160,161,246,33};
    const int b[]={362,398,384,385,386,387,388,466,263};
    for(int i=0;i<9;++i){
        float x=.06f+.016f*i;
        d.landmarks[a[i]]={.5f-x+shift,.4f-.029f*std::sin(i*3.14159265f/8),0,1};
        d.landmarks[b[i]]={.5f+x+shift,.4f-.029f*std::sin(i*3.14159265f/8),0,1};
    }
    d.landmarks[145]={.376f+shift,.416f,0,1};
    d.landmarks[374]={.624f+shift,.416f,0,1};
    d.blendshape_names={"jawOpen"};d.blendshapes={.6f};
    return d;
}
int main(){
    constexpr unsigned W=512,H=512;
    std::vector<uint8_t> rgba(W*H*4,200);
    FaceImage img{rgba.data(),W,H,rgba.size()};
    FaceQuality q;
    auto d=face();assert(q.apply(d,img,1,FaceQualityMode::Adaptive,.7f));
    assert(d.landmarks[0].x==.5f);
    double sum=0,raw=0;
    for(int i=1;i<=40;++i){
        float dx=i%2?.002f:-.002f;d=face(dx);
        assert(q.apply(d,img,1+i/30.,FaceQualityMode::Adaptive,.7f));
        if(i>10){sum+=std::abs(d.landmarks[0].x-.5f);raw+=std::abs(dx);}
    }
    assert(sum<raw*.6);
    d=face(.05f);assert(q.apply(d,img,1+41/30.,FaceQualityMode::Adaptive,.7f));
    assert(d.landmarks[0].x>.53f&&d.landmarks[0].x<=.55f);
    d=face(-.10f);assert(q.apply(d,img,4,FaceQualityMode::Adaptive,.7f));
    assert(std::abs(d.landmarks[0].x-.4f)<.00001f);
    d={};assert(!q.apply(d,img,4.03,FaceQualityMode::Adaptive,.7f));
    d=face(.1f);assert(q.apply(d,img,4.06,FaceQualityMode::Adaptive,.7f));
    assert(d.landmarks[0].x==.6f&&d.blendshapes[0]==.6f);
    d.landmarks[468].x=std::numeric_limits<float>::quiet_NaN();
    assert(!q.apply(d,img,4.1,FaceQualityMode::Adaptive,.7f)&&d.landmarks.empty());
    d=face();assert(q.apply(d,img,4.13,FaceQualityMode::Adaptive,0.f));
    // Paint a dark strip 2px below the predicted upper lids.
    const int ids[]={173,157,158,159,160,161,246,398,384,385,386,387,388,466};
    for(int id:ids){auto p=d.landmarks[id];int x=int(p.x*W),y=int(p.y*H)+2;
        for(int yy=y-1;yy<=y+1;++yy)for(int xx=x-5;xx<=x+5;++xx)
            rgba[(yy*W+xx)*4]=rgba[(yy*W+xx)*4+1]=rgba[(yy*W+xx)*4+2]=25;
    }
    q.reset();auto refined=face();assert(q.apply(refined,img,5,FaceQualityMode::Precision,0.f));
    assert(refined.landmarks[159].y>d.landmarks[159].y+.001f);
    assert(refined.landmarks[133].x==d.landmarks[133].x);
    assert(refined.landmarks[468].y==d.landmarks[468].y);
    std::fill(rgba.begin(),rgba.end(),200);q.reset();
    refined=face();assert(q.apply(refined,img,5,FaceQualityMode::Precision,0.f));
    assert(refined.landmarks[159].y==d.landmarks[159].y);
    auto closed=face();closed.landmarks[145].y=closed.landmarks[159].y;
    q.reset();assert(q.apply(closed,img,5,FaceQualityMode::Precision,0.f));
    assert(closed.landmarks[159].y==d.landmarks[159].y);
    std::cout<<"FACE_QUALITY jitter_ratio="<<sum/raw<<" fast_motion=1 reset=1 finite=1 image_refinement=1 low_contrast=1 closed=1 metadata=1\n";
}
