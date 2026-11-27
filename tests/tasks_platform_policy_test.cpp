#include "core/tasks_runtime_abi.hpp"
#include <iostream>
int main() {
    using sight::TasksRuntimeAbi;
    if(sight::tasks_runtime_for_host(33886744,false)!=TasksRuntimeAbi::Unknown ||
       sight::tasks_runtime_for_host(127908512,false)!=TasksRuntimeAbi::Unknown ||
       sight::tasks_runtime_for_host(33886744,true)!=TasksRuntimeAbi::Legacy ||
       sight::tasks_runtime_for_host(127908512,true)!=TasksRuntimeAbi::V123 ||
       sight::tasks_runtime_for_host(127908511,true)!=TasksRuntimeAbi::Unknown) {
        std::cerr<<"MediaPipe host ABI policy failed\n";return 1;
    }
    std::cout<<"MediaPipe platform ABI boundaries verified\n";
}
