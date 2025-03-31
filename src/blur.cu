#include "blur.cuh"
#include "ErrorCheck.cuh"
#include <cstdio>
#include <cuda_runtime.h>

__device__
void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.
}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // TODO: Compute the current thread index.
    uint thread_index;

    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    while (false) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        // TODO: Update the thread index
    }
}


float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    checkCuda(cudaEventCreate(&start_gpu));
    checkCuda(cudaEventCreate(&stop_gpu));
    checkCuda(cudaEventRecord(start_gpu));

    // TODO: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data. The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.
    float* gpu_raw_data;
    // ALWAYS use checkCuda (or your own error checking function) for every cuda call

    // TODO: Allocate GPU memory for the impulse signal (for now global GPU
    //       memory is fine. The data is of type float and has blur_v_size
    //       elements. Then copy the data in blur_v into the GPU memory you
    //       allocated.
    float* gpu_blur_v;

    // TODO: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    
    // TODO: Appropriately call the kernel function.

    // Check for errors on kernel call
    // Always include an error check after every kernel call
    checkCuda(cudaGetLastError());

    // TODO: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)

    // TODO: Now that we have finished our computations on the GPU, free the
    //       GPU resources.

    // Stop the recording timer and return the computation time
    checkCuda(cudaEventRecord(stop_gpu));
    checkCuda(cudaEventSynchronize(stop_gpu));
    checkCuda(cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu));
    return time_milli;
}
