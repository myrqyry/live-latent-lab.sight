#include "core/neural_render_depth.hpp"
#include <iostream>
#include <memory>
using namespace sight::nr;
struct Producer {
    bool fresh=true,wrong_size=false,bad_dimensions=false;
    uint64_t sequence=7;
    int copies=0;
};
void publish(void*opaque,calldata_t*args){
    auto& p=*static_cast<Producer*>(opaque);
    calldata_set_bool(args,"available",false);
    auto* s=static_cast<sight_depth_v1*>(calldata_ptr(args,"snapshot"));
    if(!s||!p.fresh||calldata_int(args,"max_age_ms")!=150||
       s->abi_version!=SIGHT_DEPTH_ABI_V1||s->struct_size!=sizeof(*s))return;
    s->sequence=p.sequence;s->width=p.bad_dimensions?999:2;s->height=2;
    s->source_width=p.wrong_size?800:640;s->source_height=360;
    if(uint64_t(calldata_int(args,"known_sequence"))!=p.sequence){
        ++p.copies;s->pixels[0]=200;s->pixels[1]=40;s->pixels[2]=200;s->pixels[3]=40;
    }
    calldata_set_bool(args,"available",true);
}
int main(){
    auto snapshot=std::make_unique<sight_depth_v1>();
    auto* proc=proc_handler_create();Producer p;
    proc_handler_add(proc,"void sight_depth_read_v1(in int max_age_ms, in int known_sequence, in ptr snapshot, out bool available)",publish,&p);
    bool first=read_depth_snapshot(proc,*snapshot,640,360,0)&&snapshot->sequence==7&&snapshot->pixels[0]==200;
    bool known=read_depth_snapshot(proc,*snapshot,640,360,7)&&p.copies==1;
    p.fresh=false;bool stale=!read_depth_snapshot(proc,*snapshot,640,360,7);
    p.fresh=true;p.wrong_size=true;bool mismatch=!read_depth_snapshot(proc,*snapshot,640,360,0);
    p.wrong_size=false;p.bad_dimensions=true;bool bounds=!read_depth_snapshot(proc,*snapshot,640,360,0);
    p.bad_dimensions=false;p.sequence=8;
    bool recovered=read_depth_snapshot(proc,*snapshot,640,360,0)&&snapshot->sequence==8;
    bool missing=!read_depth_snapshot(nullptr,*snapshot,640,360,0);
    proc_handler_destroy(proc);
    std::cout<<"fresh="<<first<<" known="<<known<<" stale="<<stale<<" mismatch="<<mismatch<<" bounds="<<bounds<<" recovered="<<recovered<<" missing="<<missing<<"\n";
    return first&&known&&stale&&mismatch&&bounds&&recovered&&missing?0:1;
}
