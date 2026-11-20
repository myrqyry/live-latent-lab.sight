#include "live2d/animation.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace sight::live2d;
int main(){
    char temp[]="/tmp/sight-live2d-motion-XXXXXX";assert(mkdtemp(temp));
    std::filesystem::path root(temp);
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{root};
    auto path=root/"motion.motion3.json";
    auto write=[&](const std::string& s){std::ofstream(path)<<s;};
    write(R"({"Version":3,"Meta":{"Duration":4},"Curves":[
        {"Target":"Parameter","Id":"linear","Segments":[0,0,0,1,2,2,2,4,3,3,6,1,3.25,6,3.75,10,4,10]},
        {"Target":"PartOpacity","Id":"part","Segments":[0,0,0,4,1]}]})");
    Motion m;std::string error;
    assert(read_motion(path,m,error));
    std::map<std::string,float> p,parts;
    auto close=[](float a,float b){return std::abs(a-b)<.005f;};
    m.sample(.5,p,parts);assert(close(p.at("linear"),1)&&close(parts.at("part"),.125));
    m.sample(1.5,p,parts);assert(close(p.at("linear"),2));
    m.sample(2.5,p,parts);assert(close(p.at("linear"),6));
    m.sample(3.5,p,parts);assert(close(p.at("linear"),8));
    m.sample(4.5,p,parts);assert(close(p.at("linear"),1));
    write(R"({"Version":3,"Meta":{"Duration":2},"Curves":[{"Target":"Parameter","Id":"bad","Segments":[0,0,1,1,0]}]})");
    assert(!read_motion(path,m,error)&&m.curves.empty());
    write(R"({"Version":3,"Meta":{"Duration":2},"Curves":[{"Target":"Parameter","Id":"bad","Segments":[0,0,0,-1,1]}]})");
    assert(!read_motion(path,m,error));
    write(R"({"Type":"Live2D Expression","Parameters":[{"Id":"a","Value":.3,"Blend":"Add"}]})");
    std::vector<ExpressionValue> e;
    assert(!read_expression(path,e,error)); // invalid JSON decimal
    write(R"({"Type":"Live2D Expression","Parameters":[{"Id":"a","Value":0.3,"Blend":"Add"},{"Id":"b","Value":0.5,"Blend":"Multiply"},{"Id":"c","Value":1,"Blend":"Overwrite"}]})");
    assert(read_expression(path,e,error)&&e.size()==3&&e[1].blend==ExpressionBlend::Multiply&&e[2].blend==ExpressionBlend::Overwrite);
    std::cout<<"LIVE2D_ANIMATION linear=1 bezier=1 step=1 inverse_step=1 loop=1 invalid=1 expression=1\n";
}
