extern "C" __global__ void sketch_pack(const unsigned char* rgba,float* tensor,unsigned w,unsigned h,unsigned pw,unsigned ph){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pw*ph)return;
    unsigned x=i%pw,y=i/pw;
    x=x<w?x:w-1;y=y<h?y:h-1;
    const unsigned char* p=rgba+(y*w+x)*4;
    for(unsigned c=0;c<3;++c)tensor[i*3+c]=float(p[c])*(2.0f/255.0f)-1.0f;
}
extern "C" __global__ void sketch_unpack(const float* tensor,unsigned char* rgba,unsigned w,unsigned h,unsigned pw){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h)return;
    unsigned j=(i/w*pw+i%w)*3;
    for(unsigned c=0;c<3;++c){
        float v=(tensor[j+c]+1.0f)*0.5f;
        rgba[i*4+c]=static_cast<unsigned char>(fminf(fmaxf(v,0.0f),1.0f)*255.0f);
    }
    rgba[i*4+3]=255;
}
