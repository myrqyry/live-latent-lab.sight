// Original, model-independent FP8 E4M3FN -> FP16 and SM75 WMMA primitives.
// One-time dequantization; not an implementation of NVIDIA's network.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstddef>
#include <cstdint>

namespace turing {
__constant__ uint16_t fp8_lut[256];

inline uint16_t e4m3fn_to_half_bits(unsigned b) {
    const unsigned s = (b & 128u) << 8;
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 15 && m == 7) return s | 0x7e00u;
    if (e == 0) {
        if (m == 0) return s;
        if (m >= 4) return s | (8u << 10) | ((m - 4u) << 8);
        if (m >= 2) return s | (7u << 10) | ((m - 2u) << 9);
        return s | (6u << 10);
    }
    return s | ((e + 8u) << 10) | (m << 7);
}

inline cudaError_t init_lut() {
    uint16_t host[256];
    for (unsigned i=0;i<256;++i) host[i]=e4m3fn_to_half_bits(i);
    return cudaMemcpyToSymbol(fp8_lut,host,sizeof(host));
}

// Scale in FP32 before a single rounding to half. Bounds-check every element.
__global__ void dequantize(const unsigned char* input, __half* output,
                           size_t count, float scale) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < count) output[i] = __float2half_rn(
        __half2float(__ushort_as_half(fp8_lut[input[i]]))*scale);
}

// Contract: one warp per block, M/N/K padded to multiples of 16;
// A row-major, B column-major, C row-major FP32. CUDA allocations are aligned.
// Keep accumulator fragments in FP32; conversion (if needed) is a separate op.
__global__ void gemm_tiles(const __half* A,const __half* B,float* C,int N,int K) {
    using namespace nvcuda;
    const int row = blockIdx.y*16, col = blockIdx.x*16;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(c,0.f);
    for (int k=0;k<K;k+=16) {
        wmma::load_matrix_sync(a,A+row*K+k,K);
        wmma::load_matrix_sync(b,B+col*K+k,K);
        wmma::mma_sync(c,a,b,c);
    }
    wmma::store_matrix_sync(C+row*N+col,c,N,wmma::mem_row_major);
}
} // namespace turing
