#include "core/ocr_output.hpp"
#include "core/ocr_processing.hpp"
#include "core/ocr_editor_geometry.hpp"
#include <iostream>
#define CHECK(x) do{if(!(x)){std::cerr<<__LINE__<<": "<<#x<<"\n";return 1;}}while(0)
int main(){
 using namespace sight;
 auto r=ocr_region(800,400,0,50,100,50);
 CHECK(r.x==0&&r.y==200&&r.w==800&&r.h==200);
 auto edge=ocr_region(800,400,90,90,100,100);
 CHECK(edge.x==720&&edge.y==360&&edge.w==80&&edge.h==40);
 std::vector<uint8_t> img{0,1,2,3,4,5,6,7};
 CHECK(ocr_crop(img,4,2,{1,0,2,2})==std::vector<uint8_t>({1,2,5,6}));
 CHECK(ocr_clean("  A\r\nB \n")=="A\nB");
 CHECK(ocr_format("[{region}] {text}","score","12")=="[score] 12");
 CHECK(ocr_format("{text}","x","{region}")=="{region}");
 CHECK(!ocr_format("{text}{text}{text}","x",std::string(8192,'x')));
 CHECK(ocr_format("{text}","x","")==std::string{});
 OcrConsensus c;
 CHECK(!c.observe("ONE",2));CHECK(c.observe("ONE",2)&&c.value()=="ONE");
 CHECK(!c.observe("TWO",2));CHECK(c.value()=="ONE");
 CHECK(!c.observe("ONE",2));CHECK(c.observe("ONE",2));CHECK(!c.observe("",2));CHECK(c.observe("",2)&&c.value().empty());
 c.reset();CHECK(!c.observe("ONE",2));CHECK(c.observe("TWO",1)&&c.value()=="TWO");
 CHECK(ocr_preprocess({10,20,30},3,1,1)==std::vector<uint8_t>({245,235,225}));
 CHECK(ocr_preprocess({20,60,100},3,1,2)==std::vector<uint8_t>({0,127,255}));
 CHECK(ocr_preprocess({40,40},2,1,2)==std::vector<uint8_t>({40,40}));
 CHECK(ocr_preprocess({10,10,240,240},4,1,3)==std::vector<uint8_t>({0,0,255,255}));
 CHECK(ocr_preprocess({1,2},3,1,3).empty());
 // Two dark glyphs under substantially different local illumination.
 std::vector<uint8_t> uneven(80*40);
 for(unsigned y=0;y<40;++y)for(unsigned x=0;x<80;++x)
    uneven[y*80+x]=x<40?75:220;
 uneven[20*80+15]=35;uneven[20*80+65]=180;
 auto local=ocr_preprocess(uneven,80,40,4);
 CHECK(local[20*80+15]==0&&local[20*80+65]==0);
 CHECK(local[20*80+10]==255&&local[20*80+70]==255);
 CHECK(ocr_preprocess(std::vector<uint8_t>(100,40),10,10,4)==std::vector<uint8_t>(100,255));
 OcrChangeGate gate;
 std::vector<uint8_t> a{1,2,3}, b{1,2,4};
 CHECK(!gate.reuse(a,100));gate.confirm(a,100);
 CHECK(gate.reuse(a,150));CHECK(!gate.reuse(b,150));
 CHECK(!gate.reuse(a,2100));gate.reset();CHECK(!gate.reuse(a,151));
 CHECK(ocr_numeric("CODE 12345")==12345);
 CHECK(ocr_numeric("HP -12.5")==-12.5);
 CHECK(!ocr_numeric("HP 12 / 20"));CHECK(!ocr_numeric("abc123"));
 CHECK(!ocr_numeric("1,234"));CHECK(!ocr_numeric("1e3"));
 CHECK(ocr_match("LEVEL UP","LEVEL",false)==1.0);
 CHECK(ocr_match("LEVEL UP","LEVEL",true)==0.0);
 CHECK(!ocr_match("LEVEL UP","",false));
 auto fit=ocr_editor_fit(800,600,1920,1080);
 CHECK(fit.x==0&&fit.y==75&&fit.w==800&&fit.h==450);
 auto selection=ocr_editor_drag(fit,600,412.5,200,187.5);
 CHECK((selection&&*selection==OcrEditorRect{25,25,50,50}));
 CHECK(!ocr_editor_drag(fit,10,20,600,100));
 auto clipped=ocr_editor_drag(fit,200,187.5,900,900);
 CHECK((clipped&&*clipped==OcrEditorRect{25,25,75,75}));
 CHECK(!ocr_editor_drag(fit,200,200,201,201));
 auto portrait=ocr_editor_fit(800,600,400,800);
 CHECK(portrait.x==250&&portrait.y==0&&portrait.w==300&&portrait.h==600);
 CHECK(!ocr_editor_drag(ocr_editor_fit(0,600,400,800),0,0,100,100));
 std::cout<<"SIGHT_OCR_OUTPUT_UNIT_PASS=1\n";
}
