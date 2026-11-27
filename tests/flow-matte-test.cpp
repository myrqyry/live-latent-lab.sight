#include "core/flow-matte.hpp"
#include <cstdio>
#include <limits>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed: %s\n",#x);return 1;}}while(false)
int main(){
    constexpr unsigned w=8,h=4;
    std::vector<uint8_t> prev(w*h,0),cur(w*h,0),prgb(w*h*3,0),rgb=prgb;
    for(unsigned y=0;y<h;++y)for(unsigned x=2;x<4;++x){
        prev[y*w+x]=200;cur[y*w+x+2]=160;
        for(unsigned c=0;c<3;++c){prgb[(y*w+x)*3+c]=180;rgb[(y*w+x+2)*3+c]=180;}
    }
    std::vector<float> flow(8,0);
    std::fill_n(flow.begin(),4,-1.f); // backward dx=-1 base pixel = -2 source pixels
    auto result=cur;
    CHECK(sight::warp_matte(result,rgb,prev,prgb,w,h,flow,2,2,4,2,.75f)>0);
    CHECK(result[4]>cur[4]&&result[4]<=200&&result[2]==0&&result[0]==0);
    auto scene=rgb;std::fill(scene.begin(),scene.end(),255);
    result=cur;CHECK(sight::warp_matte(result,scene,prev,prgb,w,h,flow,2,2,4,2,.75f)==0&&result==cur);
    result=cur;flow[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK(sight::warp_matte(result,rgb,prev,prgb,w,h,flow,2,2,4,2,.75f)==0&&result==cur);
    flow.assign(8,0);std::fill_n(flow.begin(),4,100.f);
    CHECK(sight::warp_matte(result,rgb,prev,prgb,w,h,flow,2,2,4,2,.75f)==0&&result==cur);
    CHECK(sight::warp_matte(result,rgb,prev,prgb,w,h,flow,2,2,4,2,0)==0&&result==cur);
    // A uniform lighting adjustment should not trigger an unnecessary
    // motion warp; translated texture should.
    constexpr unsigned W=96,H=64;
    std::vector<uint8_t> a(W*H*3),lit(a.size()),shifted(a.size());
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x)for(unsigned c=0;c<3;++c){
        const auto i=(size_t(y)*W+x)*3+c;
        a[i]=uint8_t(30+(x*13+y*17+c*29)%170);
        lit[i]=uint8_t(std::lround(a[i]*1.06f+3));
    }
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x)for(unsigned c=0;c<3;++c)
        shifted[(size_t(y)*W+x)*3+c]=a[(size_t(y)*W+(x+W-2)%W)*3+c];
    CHECK(sight::flow_stationary(a,lit,W,H));
    CHECK(sight::flow_stationary(a,a,W,H));
    CHECK(!sight::flow_stationary(a,shifted,W,H));
    CHECK(!sight::flow_stationary(a,lit,0,H));
    CHECK(!sight::flow_stationary(a,std::vector<uint8_t>{},W,H));
    puts("SIGHT_FLOW_MATTE_UNIT_PASS");
}
