#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    const int size = 16*1024;
    float *h_data; // Host data
    float *d_data; // Device data

    cudaStream_t stream1;
    cudaError_t result;
    result = cudaStreamCreate(&stream1);

    // // Allocate host memory as pageable memory
    h_data = (float*)malloc(size * sizeof(float));

    // Allocate host memory as pinned memory
    // cudaHostAlloc((void**)&h_data, size * sizeof(float), cudaHostAllocDefault);

    // Allocate device memory
    cudaMalloc((void**)&d_data, size * sizeof(float));

    // Initialize host data
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }

    cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    // Constantly copy data from host to device and back
    while (1) {
        cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
        // cudaMemcpyAsync(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    }

    // Free memory
    cudaFree(d_data);
    free(h_data);
    result = cudaStreamDestroy(stream1);

    return 0;
}