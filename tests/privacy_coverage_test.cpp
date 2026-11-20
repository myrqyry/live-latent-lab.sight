#include "core/privacy_coverage.hpp"
#include <iostream>
using namespace sight;
int main() {
 std::vector<FastFace> faces={
     {.score=.9f,.x=12,.y=20,.w=10,.h=8},
     {.score=.9f,.x=72,.y=52,.w=10,.h=8}
 };
 auto boxes=privacy::coverage(faces,100,100);
 if(boxes.size()!=2 || boxes[0].x0>=.12f || boxes[1].x1<=.82f)return 1;
 std::vector<uint8_t> mask;
 privacy::paint_mask(mask,128,72,boxes);
 const auto at=[&](int x,int y){return mask[y*128+x];};
 if(at(22,18)!=255||at(95,40)!=255||at(64,4)!=0)return 2;
 if(!privacy::hide_entire_frame(false,false,false,100,0))return 3;
 if(!privacy::hide_entire_frame(true,true,false,100,100))return 4;
 if(privacy::hide_entire_frame(true,true,true,800,100))return 5;
 if(privacy::hide_entire_frame(true,true,true,1900,100))return 6;
 if(!privacy::hide_entire_frame(true,true,true,1901,100))return 7;
 if(!privacy::hide_entire_frame(true,true,true,100,0))return 8;
 if(!privacy::hide_entire_frame(true,true,true,99,100))return 9;
 if(!privacy::detection_result_usable(true,1100,1000))return 10;
 if(privacy::detection_result_usable(false,1100,1000))return 11;
 if(privacy::detection_result_usable(true,5100,1000))return 12;
 if(privacy::detection_result_usable(true,1100,0))return 13;
 std::vector<privacy::Rect> manual;
 if(!privacy::parse_text_zones("0.1,0.2,0.3,0.4; 0.60,0.05,0.85,0.20",manual)
    ||manual.size()!=2||manual[0].x0!=.1f||manual[1].y1!=.20f)return 14;
 privacy::paint_mask(mask,100,100,manual);
 if(mask[30*100+20]!=255 || mask[10*100+70]!=255 || mask[85*100+50]!=0)return 15;
 const auto original=manual;
 for(const auto* bad:{"-1,0,1,1","0,0,2,1","0,0,0,1",
                      "nan,0,1,1","0,0,1,1,","0,0,1","1,1,0,0",
                      "0,0,1,1;not-a-zone"}) {
    if(privacy::parse_text_zones(bad,manual)||manual.size()!=original.size())return 16;
 }
 if(!privacy::parse_text_zones("",manual)||!manual.empty())return 17;
 // A model span is text, NOT a pixel box. The OCR line's box is the
 // conservative region concealed when a valid detected span overlaps it.
 const std::vector<privacy::TextEvidence> lines={
    {"Hi from Alice Smith", {0.1f,0.2f,0.8f,0.3f}, {{8,19}}},
    {"Nothing sensitive here", {0.1f,0.5f,0.8f,0.6f}, {}},
 };
 const auto textBoxes=privacy::text_coverage(lines);
 if(textBoxes.size()!=1 || textBoxes[0].x0>=.1f || textBoxes[0].x1<=.8f)return 18;
 // Out-of-bounds spans must not trigger the wrong concealment.
 if(!privacy::text_coverage({{"abc",{0,0,1,1},{{3,4}}}}).empty())return 19;
 std::cout<<"PASS: multi-face coverage, text regions and conservative OCR-to-mask geometry\n";
}
