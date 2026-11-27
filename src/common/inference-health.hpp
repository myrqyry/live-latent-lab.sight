#pragma once
#include <chrono>
#include <cstdint>
#include <cmath>
namespace sight_backend {
// Fixed-size, worker-owned observations, guarded by the existing input
// mutex. No allocations, timers, device queries or render-side work.
struct InferenceHealth {
    using Clock=std::chrono::steady_clock;
    enum class State {Waiting,Recent,Quiet,Error};
    static constexpr auto quiet_after=std::chrono::seconds(5);
    uint64_t successes=0,failures=0;
    bool last_failed=false;
    double last_ms=0,interval_ms=0;
    Clock::time_point last_success{};
    void record(bool ok,double ms,Clock::time_point now){
        last_ms=std::isfinite(ms)&&ms>=0?ms:0;
        if(!ok){++failures;last_failed=true;interval_ms=0;return;}
        const auto gap=now-last_success;
        if(successes&&!last_failed&&gap>Clock::duration::zero()&&gap<quiet_after){
            const double elapsed=std::chrono::duration<double,std::milli>(gap).count();
            interval_ms=interval_ms>0?.75*interval_ms+.25*elapsed:elapsed;
        }else interval_ms=0;
        last_success=now;++successes;last_failed=false;
    }
    State state(Clock::time_point now)const{
        if(last_failed)return State::Error;
        if(!successes)return State::Waiting;
        return now-last_success>=quiet_after?State::Quiet:State::Recent;
    }
    int64_t age(Clock::time_point now)const{
        return successes?std::chrono::duration_cast<std::chrono::milliseconds>(now-last_success).count():0;
    }
    double rate(Clock::time_point now)const{
        return state(now)==State::Recent&&interval_ms>0?1000./interval_ms:0.;
    }
};
}
