#include "core/semantic_regions.hpp"
#include <cmath>
#include <iostream>
int main() {
 using namespace sight::semantic;
 // 2:1 input letterboxed to a square: vertical padding is 80 px.
 const auto t=letterbox(640,320,320,320);
 if(t.width!=320||t.height!=160||t.left!=0||t.top!=80)return 1;
 // Raw YOLO channel-major output, two categories and three candidates.
 const float data[]={160,50,160, 160,30,160, 80,40,80, 40,20,40, .9f,.99f,.2f, .1f,.01f,.85f};
 std::vector<sight::privacy::Rect> boxes;
 if(!decode(data,6,3,t,{0},.5f,0.f,boxes)||boxes.size()!=1)return 2;
 if(std::abs(boxes[0].x0-.375f)>1e-5f||std::abs(boxes[0].y0-.375f)>1e-5f||
    std::abs(boxes[0].x1-.625f)>1e-5f||std::abs(boxes[0].y1-.625f)>1e-5f)return 3;
 if(!decode(data,6,3,t,{1},.5f,.2f,boxes)||boxes.size()!=1||boxes[0].x0>=.375f)return 4;
 if(decode(data,6,3,t,{2},.5f,0.f,boxes))return 5;
 if(!decode(data,6,3,t,{},.995f,0.f,boxes)||!boxes.empty())return 6;
 std::vector<int> ids;
 if(!parse_ids("0, 3,5",ids)||ids!=std::vector<int>({0,3,5})||parse_ids("1,bad",ids))return 7;
 std::vector<uint8_t> rgba(2*1*4,255);rgba[0]=255;rgba[1]=0;rgba[2]=0;
 rgba[4]=0;rgba[5]=255;rgba[6]=0;
 auto input=prepare(rgba.data(),2,1,letterbox(2,1,2,2));
 if(input.size()!=12||input[0]!=1.f||input[1]!=0.f||input[4]!=0.f||input[5]!=1.f||
    std::abs(input[2]-114.f/255.f)>1e-6f)return 8;
 std::cout<<"PASS: semantic detection geometry, class filtering, padding, zero results, RGB letterbox\n";
}
