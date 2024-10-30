#include "reduction.h"
#include <stdio.h>
#define BLOCK_SIZE 512 // must be power of 2
#define TN 8           // number of elements per thread

__inline__ __device__ float warp_reduce_sum(float val)
{
    unsigned int mask = __activemask();
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    return val;
}
// 避免多次分配共享內存：在 __device__ 函數內，若共享內存變量不加 static，
// 每次調用該函數時都會分配新的共享內存，而加上 static 後，
// 編譯器確保只分配一次共享內存。在這段代碼中，block_reduce_sum 可能會被多個 thread 調用，
// static 確保所有 thread 共享同一塊內存，不重複分配，避免內存浪費。
// 確保變量唯一性：static __shared__ 的變量在每個 block 中是唯一的，
// 因此即使這個函數被多次調用，它們都會共享這個唯一的 shared 變量。
// 這樣，所有的 thread 都能在該變量上執行同步和數據交換操作，達到正確的數據共享。
// 優化性能：避免在每次函數調用時重複分配共享內存，可以減少內存分配和管理的開銷，提高性能，
// 這對於 CUDA 程序來說尤為重要。
__inline__ __device__ float block_reduce_sum(float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int warpIdx = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int numWarps = blockDim.x / warpSize;
    
    // Each warp performs partial reduction
    val = warp_reduce_sum(val); 

    // Write reduced value of each warp to shared memory
    if (lane == 0) shared[warpIdx] = val; 

    // Wait for all partial reductions
    __syncthreads(); 

    //read from shared memory only if that warp existed
    if (warpIdx == 0) {
        val = (threadIdx.x < numWarps) ? shared[lane] : 0;
        val = warp_reduce_sum(val); //Final reduce within first warp
    }

    return val;
}

__global__ void reduction_warp_sync_kernel(float *d_output, const float *d_input, const int numElements) {

    // global index
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (numElements + TN - 1) / TN;

    // Load the first data_stride into shared memory
    float4 input = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll
    for (int i = globalIdx; i < size; i += blockDim.x * gridDim.x) {
        #pragma unroll
        for (int j = 0; j < TN; j+=4) {
            float4 temp = reinterpret_cast<const float4*>(&d_input[i * TN + j])[0];
            input.x += temp.x;
            input.y += temp.y;
            input.z += temp.z;
            input.w += temp.w;
        }
    }

    float inputSum = input.x + input.y + input.z + input.w;
    inputSum = block_reduce_sum(inputSum);

    // output 
    if (threadIdx.x == 0) d_output[blockIdx.x] = inputSum;
}

void reduction_warp_sync(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_warp_sync_kernel, BLOCK_SIZE, 0);
    int tileSize = BLOCK_SIZE * TN;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (tileSize) - 1) / (tileSize));

    // launch kernel
    reduction_warp_sync_kernel<<<gridSize, BLOCK_SIZE>>>(d_output, d_input, numElements);
    reduction_warp_sync_kernel<<<1, BLOCK_SIZE>>>(d_output, d_output, gridSize);
}