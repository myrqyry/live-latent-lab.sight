#include <obs.h>
#include <obs-module.h>
#include <cstdio>
#include <cstring>
#include <string>
int main(int argc,char**argv){
 if(argc!=4||!obs_startup("en-US",argv[3],nullptr))return 2;
 obs_module_t*m=nullptr;
 if(obs_open_module(&m,argv[1],argv[2])!=MODULE_SUCCESS||!obs_init_module(m))return 3;
 bool ok=true;size_t n=0;const char*id=nullptr;
 for(size_t i=0;obs_enum_filter_types(i,&id);++i){
  const char*name=obs_source_get_display_name(id);bool pass=name&&!strncmp(name,"lll.sight ",10);
  printf("SIGHT_FILTER id=%s name=%s pass=%d\n",id,name?name:"",pass);ok&=pass;++n;
 }
 for(size_t i=0;obs_enum_input_types(i,&id);++i){
  if(!strcmp(id,"audio_line"))continue;
  const char*name=obs_source_get_display_name(id);bool pass=name&&!strncmp(name,"lll.sight ",10);
  printf("SIGHT_SOURCE id=%s name=%s pass=%d\n",id,name?name:"",pass);ok&=pass;++n;
 }
 ok&=obs_get_module_name(m)&&std::string(obs_get_module_name(m))=="lll.sight";
 printf("SIGHT_LOWERCASE_PASS=%d count=%zu module=%s\n",ok&&n>=8,n,obs_get_module_name(m));
 obs_shutdown();return ok&&n>=8?0:1;
}
