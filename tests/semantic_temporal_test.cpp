#include "core/semantic_tracks.hpp"
#include <iostream>
using namespace sight::semantic;
int main() {
 std::vector<Threshold> thresholds;
 if(!parse_thresholds("0:0.7, 1:0.2",thresholds)||thresholds.size()!=2||
    parse_thresholds("1:0.2,1:0.4",thresholds)||parse_thresholds("2:nan",thresholds))return 1;
 auto t=letterbox(320,320,320,320);
 const float data[]={100,220, 100,220, 40,40, 40,40, .6f,.1f, .1f,.3f};
 std::vector<Region> raw;
 if(!decode_regions(data,6,2,t,{},.5f,0.f,thresholds,raw)||raw.size()!=1||raw[0].category!=1)return 2;
 if(decode_regions(data,6,2,t,{},.5f,0.f,{{2,.5f}},raw))return 3;
 Tracker tr;
 auto a=tr.update({{{.1f,.2f,.3f,.4f},1,.9f},{{.11f,.2f,.31f,.4f},1,.8f}},100,200);
 if(a.size()!=1||a[0].id==0||a[0].box.x0>.1f||a[0].box.x1<.31f)return 4;
 const auto id=a[0].id;
 a=tr.update({{{.15f,.2f,.35f,.4f},1,.9f}},150,200);
 if(a.size()!=1||a[0].id!=id||a[0].box.x0>.15f||a[0].box.x1<.35f)return 5;
 a=tr.update({},220,200);
 if(a.size()!=1||a[0].id!=id||!a[0].held||a[0].box.x1<.35f)return 6;
 a=tr.update({},351,200);
 if(!a.empty())return 7;
 a=tr.update({{{.1f,.1f,.2f,.2f},0,.9f},{{.1f,.1f,.2f,.2f},1,.9f}},400,200);
 if(a.size()!=2||a[0].id==a[1].id)return 8;
 tr.reset();if(!tr.update({},401,200).empty())return 9;
 // A moving region is predicted forward through a brief detector miss.
 Tracker motion;
 auto m=motion.update({{{.1f,.2f,.2f,.3f},1,.9f}},100,200);
 auto moving_id=m[0].id;
 m=motion.update({{{.14f,.2f,.24f,.3f},1,.9f}},140,200);
 m=motion.update({},180,200);
 if(m.size()!=1||m[0].id!=moving_id||!m[0].held||m[0].box.x1<.275f)return 10;
 m=motion.update({{{.22f,.2f,.32f,.3f},1,.9f}},220,200);
 if(m.size()!=1||m[0].id!=moving_id||m[0].box.x1<.32f)return 11;
 // No persistence means no predicted mask on an empty detection.
 if(!motion.update({},240,0).empty())return 12;
 // Prediction must remain clipped and expire, never accumulate a trail.
 Tracker edge;
 edge.update({{{.82f,.2f,.92f,.3f},2,.9f}},100,180);
 edge.update({{{.87f,.2f,.97f,.3f},2,.9f}},140,180);
 m=edge.update({},220,180);
 if(m.size()!=1||m[0].box.x1!=1.f||!sight::privacy::valid_rect(m[0].box))return 13;
 if(!edge.update({},321,180).empty())return 14;
 std::cout<<"PASS: class overrides, first-frame coverage, duplicate union, tracking identity, bounded hold, reset\n";
}
