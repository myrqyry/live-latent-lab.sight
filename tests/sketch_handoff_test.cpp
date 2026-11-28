// Exercise the actual filter helpers without adding test hooks to the module.
#include "../src/animegan-filter.cpp"
#include <future>
#include <iostream>
OBS_DECLARE_MODULE()
#define CHECK(x) do { if(!(x)){std::cerr<<"FAIL: "<<#x<<"\n";return 1;} }while(0)
int main(){
    using namespace std::chrono_literals;
    AnimeGanFilter f;
    f.model_ready=true;
    f.model_generation=1;
    // Hold the mutex on the control thread. A render-side attempt must return
    // before we release it, including the post-readback submission path.
    auto nonblocking=[&](std::mutex& m,auto task){
        std::unique_lock guard(m);
        auto future=std::async(std::launch::async,task);
        bool returned=future.wait_for(500ms)==std::future_status::ready;
        guard.unlock();
        return returned&&future.get();
    };
    uint64_t generation=0;
    CHECK(nonblocking(f.input_mutex,[&]{return !try_capture(&f,generation);}));
    CHECK(try_capture(&f,generation)&&generation==1);
    f.capture_scratch.assign(16,7);
    CHECK(nonblocking(f.input_mutex,[&]{return !try_submit(&f,generation,2,2);}));
    CHECK(!f.capture_in_flight&&!f.frame_pending);
    CHECK(try_submit(&f,generation,2,2));
    CHECK(f.capture_in_flight&&f.frame_pending&&f.input_rgba.size()==16);
    CHECK(!try_capture(&f,generation));
    f.capture_in_flight=f.frame_pending=false;
    ++f.model_generation;
    CHECK(!try_submit(&f,1,2,2));
    CHECK(try_capture(&f,generation)&&generation==2);
    // Published pixels remain owned by the render thread after the swap.
    f.output_rgba.assign(16,42);f.output_width=f.output_height=2;
    f.output_generation=2;f.new_output=true;
    CHECK(nonblocking(f.output_mutex,[&]{return !try_take_output(&f);}));
    CHECK(f.new_output&&f.upload_rgba.empty());
    CHECK(try_take_output(&f)&&!f.new_output&&f.upload_rgba[0]==42);
    f.output_rgba.assign(16,99);f.output_generation=1;f.new_output=true;
    CHECK(!try_take_output(&f)&&f.upload_rgba[0]==42);
    f.output_generation=2;f.new_output=true;
    CHECK(try_take_output(&f)&&f.upload_rgba[0]==99);
    std::cout<<"SKETCH_HANDOFF_PASS input_contention=1 submission_contention=1 output_contention=1 generation=1 ownership=1\n";
}
