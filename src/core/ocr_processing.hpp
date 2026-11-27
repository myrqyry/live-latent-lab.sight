#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>
namespace sight {
// No geometry-changing preprocessing. Otsu is global; no extra dependency.
inline std::vector<uint8_t> ocr_preprocess(std::vector<uint8_t> p,unsigned w,unsigned h,int mode){
 if(!w||!h||p.size()!=size_t(w)*h)return {};
 if(mode==1){for(auto& v:p)v=255-v;}
 else if(mode==2){
  auto [lo,hi]=std::minmax_element(p.begin(),p.end());int low=*lo,high=*hi;
  if(high-low>=8)for(auto& v:p)v=uint8_t((int(v)-low)*255/(high-low));
 }else if(mode==3){
  std::array<uint64_t,256> hist{};uint64_t sum=0;
  for(auto v:p){++hist[v];sum+=v;}
  uint64_t n=0,s=0;double best=0;int threshold=0;
  for(int t=0;t<255;++t){
   n+=hist[t];s+=hist[t]*t;if(!n||n==p.size())continue;
   double delta=double(s)/n-double(sum-s)/(p.size()-n);
   double score=double(n)*(p.size()-n)*delta*delta;
   if(score>best){best=score;threshold=t;}
  }
  if(best>0)for(auto& v:p)v=v<=threshold?0:255;
 }else if(mode==4){
  // Local mean threshold, bounded rolling column sums; O(w*h), O(w)
  // auxiliary storage beyond the output. Dark glyphs on lighter paper.
  const unsigned radius=std::clamp(std::min(w,h)/24,5u,32u);
  std::vector<unsigned> column(w,0);
  for(unsigned y=0;y<std::min(h,radius+1);++y)
   for(unsigned x=0;x<w;++x)column[x]+=p[size_t(y)*w+x];
  std::vector<uint8_t> binary(p.size());
  for(unsigned y=0;y<h;++y){
   unsigned top=y>radius?y-radius:0,bottom=std::min(h-1,y+radius);
   unsigned sum=0;
   for(unsigned x=0;x<std::min(w,radius+1);++x)sum+=column[x];
   for(unsigned x=0;x<w;++x){
    unsigned left=x>radius?x-radius:0,right=std::min(w-1,x+radius);
    unsigned count=(right-left+1)*(bottom-top+1);
    binary[size_t(y)*w+x]=(unsigned(p[size_t(y)*w+x])+8)*count<sum?0:255;
    if(x>=radius)sum-=column[x-radius];
    if(x+radius+1<w)sum+=column[x+radius+1];
   }
   if(y>=radius)for(unsigned x=0;x<w;++x)column[x]-=p[size_t(y-radius)*w+x];
   if(y+radius+1<h)for(unsigned x=0;x<w;++x)column[x]+=p[size_t(y+radius+1)*w+x];
  }
  return binary;
 }
 return p;
}
class OcrChangeGate {
 std::vector<uint8_t> confirmed_;
 int64_t at_=0;
public:
 void reset(){confirmed_.clear();at_=0;}
 bool reuse(const std::vector<uint8_t>& pixels,int64_t now) const{
  return !confirmed_.empty()&&now>=at_&&now-at_<2000&&pixels==confirmed_;
 }
 void confirm(const std::vector<uint8_t>& pixels,int64_t now){confirmed_=pixels;at_=now;}
};
// Exactly one standalone ASCII decimal token. Reject ambiguous readings,
// grouped numbers, ratios, scientific notation and attached word digits.
inline std::optional<double> ocr_numeric(std::string_view text){
 auto digit=[](char c){return c>='0'&&c<='9';};
 auto word=[&](char c){return digit(c)||(c>='A'&&c<='Z')||(c>='a'&&c<='z')||c=='_'||static_cast<unsigned char>(c)>=128;};
 std::optional<double> found;
 for(size_t i=0;i<text.size();){
  if(!digit(text[i])&&text[i]!='-'&&text[i]!='+'){++i;continue;}
  size_t start=i;
  if(text[i]=='-'||text[i]=='+'){if(i+1==text.size()||!digit(text[i+1])){++i;continue;}++i;}
  while(i<text.size()&&digit(text[i]))++i;
  if(i+1<text.size()&&text[i]=='.'&&digit(text[i+1])){++i;while(i<text.size()&&digit(text[i]))++i;}
  if((start&&(word(text[start-1])||text[start-1]=='.'||text[start-1]==','))||
     (i<text.size()&&(word(text[i])||text[i]==','||text[i]=='.'))||found)return {};
  double value=0;
  auto* first=text.data()+start+(text[start]=='+');
  auto r=std::from_chars(first,text.data()+i,value,std::chars_format::fixed);
  if(r.ec!=std::errc{}||r.ptr!=text.data()+i||!std::isfinite(value))return {};
  found=value;
 }
 return found;
}
inline std::optional<double> ocr_match(std::string_view text,std::string_view query,bool exact){
 if(query.empty())return {};
 return (exact?text==query:text.find(query)!=std::string_view::npos)?1.0:0.0;
}
}
