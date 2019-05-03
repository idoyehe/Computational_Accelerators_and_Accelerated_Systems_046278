/* compile with: nvcc -O3 hw1.cu -o hw1 */

#include <stdio.h>
#include <sys/time.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define N_IMAGES 10000
#define HISTOGRAM_SIZE 256

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (IMG_WIDTH * IMG_HEIGHT - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

long long int distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    long long int distance_sqr = 0;
    for (int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int array_min_positive(int *arr, int len){
    int tid = threadIdx.x;
    __shared__ uchar min_arr[HISTOGRAM_SIZE];
    if (tid < len) {
        min_arr[tid] = arr[tid]; //copy the arr to preserve it
    }
    int half_size = len /2;
    while (half_size >=1){
        if (tid < half_size) {
            bool change_flag = (min_arr[tid + half_size] > 0 && min_arr[tid]
                    > min_arr[tid + half_size] || min_arr[tid] == 0);
            min_arr[tid] = change_flag * min_arr[tid + half_size] +
                           (!change_flag) * min_arr[tid];
        }
        __syncthreads();
        half_size /=2;
    }
    return min_arr[0];
}

__device__ void prefix_sum(int *arr, int len){
    int tid = threadIdx.x;
    int increment;
    for (int stride = 1; stride < len; stride *= 2) {
        if (tid < len && tid >= stride) { // in case # threads bigger than array length
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid < len && tid >= stride) { // in case # threads bigger than array length
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return;
}

__global__ void process_image_kernel(uchar *in, uchar *out) {
    int tid = threadIdx.x;
    int imageStartIndex = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;
    __shared__ int hist_shared[HISTOGRAM_SIZE];
    if (tid < HISTOGRAM_SIZE) {
        hist_shared[tid] = 0;
    }

    for(int startOffset = 0; startOffset < IMG_WIDTH * IMG_HEIGHT; startOffset += blockDim.x){
        int pixelValue = in[imageStartIndex + startOffset + tid];
        atomicAdd(hist_shared + pixelValue, 1);
    }
    prefix_sum(hist_shared, HISTOGRAM_SIZE);
    __syncthreads();
    int cdfMin = array_min_positive(hist_shared, HISTOGRAM_SIZE);
    __syncthreads();
    if(tid ==0) {
        printf("hist_shared[0] = %d\n", hist_shared[tid]);
        printf("min = %d\n", cdfMin);
    }
    __syncthreads();
    if (tid < HISTOGRAM_SIZE) {
        printf("hist_shared[%d] = %d\n", tid,hist_shared[tid]);
    }
    return;
}

int main() {
///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
    uchar *images_in;
    uchar *images_out_cpu; //output of CPU computation. In CPU memory.
    uchar *images_out_gpu_serial; //output of GPU task serial computation. In CPU memory.
    uchar *images_out_gpu_bulk; //output of GPU bulk computation. In CPU memory.
    CUDA_CHECK( cudaHostAlloc(&images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_cpu, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_serial, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out_gpu_bulk, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, 0) );

    /* instead of loading real images, we'll load the arrays with random data */
    srand(0);
    for (long long int i = 0; i < N_IMAGES * IMG_WIDTH * IMG_HEIGHT; i++) {
        images_in[i] = rand() % 256;
    }

    double t_start, t_finish;

    // CPU computation. For reference. Do not change
    printf("\n=== CPU ===\n");
    t_start = get_time_msec();
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in = &images_in[i * IMG_WIDTH * IMG_HEIGHT];
        uchar *img_out = &images_out_cpu[i * IMG_WIDTH * IMG_HEIGHT];
        process_image(img_in, img_out);
    }
    t_finish = get_time_msec();
    printf("total time %f [msec]\n", t_finish - t_start);

    long long int distance_sqr;
///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // GPU task serial computation
    printf("\n=== GPU Task Serial ===\n"); //Do not change
    uchar *image_in_device_serial, *image_out_device_serial;
    CUDA_CHECK(cudaMalloc((void **)&image_in_device_serial,IMG_HEIGHT * IMG_WIDTH ));
    CUDA_CHECK(cudaMalloc((void **)&image_out_device_serial,IMG_HEIGHT * IMG_WIDTH ));
    t_start = get_time_msec(); //Do not change
    CUDA_CHECK( cudaMemcpy(image_in_device_serial,images_in,IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice));
    process_image_kernel<<<1,1024 >>>(image_in_device_serial,image_out_device_serial);
    int *temp_out =(int*)malloc(IMG_HEIGHT * IMG_WIDTH);
    CUDA_CHECK( cudaMemcpy(temp_out,image_out_device_serial,(IMG_HEIGHT * IMG_WIDTH),cudaMemcpyDeviceToHost));


    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
//    t_finish = get_time_msec(); //Do not change
//    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_serial); // Do not change
//    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not change
//
//    // GPU bulk
//    printf("\n=== GPU Bulk ===\n"); //Do not change
//    //TODO: allocate GPU memory for a all input images and all output images
//    t_start = get_time_msec(); //Do not change
//    //TODO: copy all input images from images_in to the GPU memory you allocated
//    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
//    //TODO: copy output images from GPU memory to images_out_gpu_bulk
//    t_finish = get_time_msec(); //Do not change
//    distance_sqr = distance_sqr_between_image_arrays(images_out_cpu, images_out_gpu_bulk); // Do not change
//    printf("total time %f [msec]  distance from baseline %lld (should be zero)\n", t_finish - t_start, distance_sqr); //Do not chhange

    return 0;
}
