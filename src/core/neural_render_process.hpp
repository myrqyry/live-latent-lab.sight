#pragma once
// Linux-only, private socket transport to a trusted local Python worker.
// No shell, network, scene-controlled executable, or inherited stdin/stdout.
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
extern char **environ;

namespace sight::nr {
class Process {
    int fd_=-1;
    pid_t pid_=-1;
public:
    Process()=default;
    Process(const Process&)=delete;
    Process& operator=(const Process&)=delete;
    ~Process(){close();}
    void close() {
        if(fd_>=0){::close(fd_);fd_=-1;}
        if(pid_<=0)return;
        auto reap=[&](){
            int status=0;
            pid_t r;
            do {r=waitpid(pid_,&status,WNOHANG);}while(r<0&&errno==EINTR);
            if(r==pid_ || (r<0&&errno==ECHILD)){pid_=-1;return true;}
            return false;
        };
        for(int i=0;i<10&&!reap();++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(pid_<=0)return;
        kill(pid_,SIGTERM);
        for(int i=0;i<10&&!reap();++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(pid_<=0)return;
        kill(pid_,SIGKILL);
        int status=0;while(waitpid(pid_,&status,0)<0&&errno==EINTR){}
        pid_=-1;
    }
    bool start(std::vector<std::string> args) {
        close();
        if(args.empty()||args[0].empty()||args[0][0]!='/')return false;
        int pair[2];
        if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,pair))return false;
        if(pair[0]<3||pair[1]<3){::close(pair[0]);::close(pair[1]);return false;}
        posix_spawn_file_actions_t actions;
        if(posix_spawn_file_actions_init(&actions)){::close(pair[0]);::close(pair[1]);return false;}
        int rc=posix_spawn_file_actions_adddup2(&actions,pair[1],STDIN_FILENO);
        if(!rc)rc=posix_spawn_file_actions_adddup2(&actions,pair[1],STDOUT_FILENO);
        if(!rc)rc=posix_spawn_file_actions_addclose(&actions,pair[0]);
        if(!rc)rc=posix_spawn_file_actions_addclose(&actions,pair[1]);
        std::vector<char*> argv;
        for(auto &a:args)argv.push_back(a.data());
        argv.push_back(nullptr);
        if(!rc)rc=posix_spawn(&pid_,argv[0],&actions,nullptr,argv.data(),environ);
        posix_spawn_file_actions_destroy(&actions);
        ::close(pair[1]);
        if(rc){pid_=-1;::close(pair[0]);return false;}
        fd_=pair[0];
        int flags=fcntl(fd_,F_GETFL);
        if(flags<0||fcntl(fd_,F_SETFL,flags|O_NONBLOCK)<0){close();return false;}
        return true;
    }
    template<class Cancel>
    bool exchange(const std::vector<uint8_t>& in,std::vector<uint8_t>& out,
                  int timeout_ms,Cancel cancelled,size_t response_size=0) {
        if(fd_<0||in.empty())return false;
        auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        auto transfer=[&](const uint8_t *sendbuf,uint8_t *recvbuf,size_t n) {
            size_t offset=0;
            while(offset<n) {
                if(cancelled()||std::chrono::steady_clock::now()>=deadline)return false;
                pollfd p{fd_,short(sendbuf?POLLOUT:POLLIN),0};
                int r=poll(&p,1,50);
                if(r<0&&errno==EINTR)continue;
                if(r<0)return false;
                if(!r)continue;
                if(!(p.revents&p.events))return false;
                ssize_t amount=sendbuf?
                    send(fd_,sendbuf+offset,n-offset,MSG_NOSIGNAL):
                    recv(fd_,recvbuf+offset,n-offset,0);
                if(amount<0&&(errno==EINTR||errno==EAGAIN||errno==EWOULDBLOCK))continue;
                if(amount<=0)return false;
                offset+=size_t(amount);
            }
            return true;
        };
        out.resize(response_size?response_size:in.size());
        return transfer(in.data(),nullptr,in.size())&&transfer(nullptr,out.data(),out.size());
    }
};
} // namespace sight::nr
