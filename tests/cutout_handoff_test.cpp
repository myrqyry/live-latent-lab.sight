#include "../src/u2net-filter.cpp"
#include <future>
#include <iostream>
OBS_DECLARE_MODULE()
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL: "<<#x<<"\n";return 1;}}while(0)
int main(){
    using namespace std::chrono_literals;
    U2NetFilter f;f.model_ready=true;f.model_generation=1;
    const auto now=std::chrono::steady_clock::now();
    f.capture_scratch.assign(16,17);
    auto nonblocking=[&](std::mutex&m,auto task){
        std::unique_lock guard(m);
        auto result=std::async(std::launch::async,task);
        bool ready=result.wait_for(500ms)==std::future_status::ready;
        guard.unlock();return ready&&result.get();
    };
    CHECK(nonblocking(f.input_mutex,[&]{return !try_submit(&f,1,320,2,2,2,2,now);}));
    CHECK(!f.frame_pending&&f.capture_scratch.size()==16);
    CHECK(!try_submit(&f,0,320,2,2,2,2,now));
    CHECK(try_submit(&f,1,320,2,2,2,2,now)&&f.frame_pending&&f.input_rgba[0]==17);
    CHECK(!try_submit(&f,1,320,2,2,2,2,now));
    f.output_mask.assign(4,90);f.raw_mask.assign(4,20);f.output_width=f.output_height=2;
    f.output_generation=1;f.new_output=true;
    uint32_t w=0,h=0;uint64_t generation=0;
    CHECK(nonblocking(f.output_mutex,[&]{return !try_copy_matte(&f,false,w,h,generation);}));
    CHECK(f.new_output&&f.upload_scratch.empty());
    CHECK(try_copy_matte(&f,false,w,h,generation)&&w==2&&h==2&&generation==1);
    CHECK(f.output_mask[0]==90&&f.upload_scratch[0]==90);
    f.new_output=true;
    CHECK(try_copy_matte(&f,true,w,h,generation)&&f.upload_scratch[0]==20&&f.raw_mask[0]==20);
    f.model_generation=2;f.new_output=true;
    CHECK(!try_copy_matte(&f,false,w,h,generation));
    std::cout<<"SIGHT_CUTOUT_HANDOFF_PASS=1\n";
}
