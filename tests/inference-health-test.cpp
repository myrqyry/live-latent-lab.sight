#include "../src/common/inference-health.hpp"
#include <cstdio>
#include <cmath>
using namespace std::chrono_literals;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed: %s\n",#x);return 1;}}while(false)
int main(){
 using H=sight_backend::InferenceHealth;
 const auto t=H::Clock::time_point{}+10s;
 H h;
 CHECK(h.state(t)==H::State::Waiting);
 h.record(true,25,t);
 CHECK(h.state(t)==H::State::Recent&&h.successes==1&&h.last_ms==25&&h.rate(t)==0);
 h.record(true,35,t+200ms);
 CHECK(std::abs(h.rate(t+200ms)-5)<.001&&h.age(t+300ms)==100);
 CHECK(h.state(t+5200ms)==H::State::Quiet&&h.rate(t+5200ms)==0);
 h.record(false,40,t+5300ms);
 CHECK(h.state(t+5300ms)==H::State::Error&&h.failures==1&&h.successes==2);
 h.record(true,30,t+5500ms);
 CHECK(h.state(t+5500ms)==H::State::Recent&&h.rate(t+5500ms)==0&&h.failures==1);
 h.record(true,20,t+5600ms);
 CHECK(std::abs(h.rate(t+5600ms)-10)<.001);
 h.record(true,20,t+12000ms);
 CHECK(h.rate(t+12000ms)==0);
 h={};
 CHECK(h.state(t)==H::State::Waiting&&h.successes==0&&h.failures==0);
 puts("SIGHT_INFERENCE_HEALTH_UNIT_PASS");
}
