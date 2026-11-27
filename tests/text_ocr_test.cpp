#include "core/text_ocr.hpp"
#include "core/ocr_processing.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::ifstream file(argv[2],std::ios::binary);
    std::vector<uint8_t> pixels((std::istreambuf_iterator<char>(file)),{});
    sight::TextOcr ocr;
    if(!ocr.load(argv[1],"eng")) {std::cerr<<"OCR engine load failed\n";return 1;}
    auto r=ocr.recognize(pixels,800,400,6);
    bool first=false,second=false;
    for(auto& l:r.lines) {
        first|=l.text.find("SIGHT TEST")!=std::string::npos&&l.x0>=50&&l.x1<=650&&l.y0>=30&&l.y1<160;
        second|=l.text.find("CODE 12345")!=std::string::npos&&l.x0>=50&&l.x1<=650&&l.y0>=200&&l.y1<340;
    }
    const auto sparse=ocr.recognize(pixels,800,400,11);
    bool sparse_first=false,sparse_second=false;
    for(const auto& line:sparse.lines) {
        sparse_first|=line.text.find("SIGHT TEST")!=std::string::npos;
        sparse_second|=line.text.find("CODE 12345")!=std::string::npos;
    }
    const bool default_layout=sparse.valid&&sparse_first&&sparse_second;
    const auto adaptive=ocr.recognize(sight::ocr_preprocess(pixels,800,400,4),800,400,6);
    bool adaptive_first=false,adaptive_second=false;
    for(const auto& line:adaptive.lines){
        adaptive_first|=line.text.find("SIGHT TEST")!=std::string::npos;
        adaptive_second|=line.text.find("CODE 12345")!=std::string::npos;
    }
    const bool adaptive_ok=adaptive.valid&&adaptive_first&&adaptive_second;
    std::cout<<"SIGHT_OCR_ADAPTIVE_REAL="<<adaptive_ok<<"\n";
    auto blank=ocr.recognize(std::vector<uint8_t>(800*400,255),800,400,6);
    auto bad=ocr.recognize({},800,400,6);
    std::cout<<"SIGHT_OCR_REAL valid="<<r.valid<<" first="<<first<<" second="<<second<<" empty="<<(blank.valid&&blank.lines.empty())<<" sparse="<<default_layout<<" malformed="<<!bad.valid<<"\n";
    return !(r.valid&&first&&second&&default_layout&&adaptive_ok&&blank.valid&&blank.lines.empty()&&!bad.valid);
}
